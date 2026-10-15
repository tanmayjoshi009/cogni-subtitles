#!/usr/bin/env python3
"""
Live Screen-Audio Translator (v2 - GPU auto-detect, low latency)
----------------------------------------------------------------
Captures whatever your computer is playing, transcribes it with Whisper,
translates it, and shows movie-style subtitles in a transparent overlay.

    python translator.py                    # auto GPU/CPU, English subtitles
    python translator.py --target hi        # Hindi subtitles
    python translator.py --model medium     # more accurate (GPU recommended)
    python translator.py --device cpu       # force CPU
    python translator.py --list-devices
    python translator.py --no-overlay       # print subtitles in the terminal
"""

import warnings
warnings.filterwarnings("ignore", message="data discontinuity in recording")
warnings.filterwarnings("ignore", message=".*symlinks.*")
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse
import collections
import ctypes
import glob
import platform
import queue
import site
import sys
import threading
import time

import numpy as np

# Make pip-installed NVIDIA CUDA libraries (cublas / cudnn) visible on Windows.
# Must run BEFORE faster_whisper / ctranslate2 are imported.
if os.name == "nt":
    for _sp in site.getsitepackages():
        for _d in glob.glob(os.path.join(_sp, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(_d)
            except Exception:
                pass
            os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")

import soundcard as sc
from faster_whisper import WhisperModel

CAPTURE_SR = 48000      # capture sample rate
WHISPER_SR = 16000      # Whisper needs 16 kHz mono
BLOCK_SEC = 0.1         # audio block size pushed through the pipeline
MAX_BACKLOG_SEC = 5.0   # if we fall further behind than this, skip ahead

HALLUCINATIONS = {
    "you", "thank you.", "thank you", "thanks for watching!", "thanks for watching.",
    "bye.", "bye!", ".", "...", "okay.", "subtitles by the amara.org community",
}


# --------------------------------------------------------------------------- #
# 1. AUDIO CAPTURE (system loopback)
# --------------------------------------------------------------------------- #
def list_devices():
    print("Available capture devices (loopback = what you hear):")
    for m in sc.all_microphones(include_loopback=True):
        tag = "LOOPBACK" if getattr(m, "isloopback", False) else "INPUT"
        print(f"  [{tag}] {m.name}")
    print(f"\nDefault speaker: {sc.default_speaker().name}")


def pick_capture_device(name_filter):
    mics = sc.all_microphones(include_loopback=True)
    if name_filter:
        for m in mics:
            if name_filter.lower() in m.name.lower():
                return m
        sys.exit(f"No capture device matching '{name_filter}'. Use --list-devices.")
    speaker = sc.default_speaker()
    try:
        return sc.get_microphone(id=str(speaker.name), include_loopback=True)
    except Exception:
        for m in mics:
            if getattr(m, "isloopback", False):
                return m
    sys.exit("No loopback device found. On macOS install BlackHole and pass "
             "--audio-device BlackHole (see README).")


def capture_loop(device, audio_q, stop):
    frames = int(CAPTURE_SR * BLOCK_SEC)
    factor = CAPTURE_SR // WHISPER_SR
    with device.recorder(samplerate=CAPTURE_SR, channels=2) as rec:
        while not stop.is_set():
            data = rec.record(numframes=frames)
            mono = data.mean(axis=1).astype(np.float32)
            n = (len(mono) // factor) * factor
            audio_q.put(mono[:n].reshape(-1, factor).mean(axis=1))


# --------------------------------------------------------------------------- #
# 2. TRANSLATION (for targets other than English)
# --------------------------------------------------------------------------- #
class Translator:
    def __init__(self, engine, target):
        self.engine, self.target = engine, target
        self._tried = set()
        if engine == "google":
            from deep_translator import GoogleTranslator
            self._g = GoogleTranslator(source="auto", target=target)

    def _argos_install(self, src, tgt):
        import argostranslate.package as pkg
        if (src, tgt) in self._tried:
            return
        self._tried.add((src, tgt))
        pkg.update_package_index()
        found = next((p for p in pkg.get_available_packages()
                      if p.from_code == src and p.to_code == tgt), None)
        if found:
            pkg.install_from_path(found.download())

    def __call__(self, text, src):
        try:
            if self.engine == "google":
                return self._g.translate(text)
            import argostranslate.translate as tr
            self._argos_install(src, self.target)
            self._argos_install(src, "en")
            self._argos_install("en", self.target)
            return tr.translate(text, src, self.target)
        except Exception as e:
            print(f"[translate error] {e}", file=sys.stderr)
            return text


# --------------------------------------------------------------------------- #
# 3. MODEL LOADING (GPU with automatic CPU fallback)
# --------------------------------------------------------------------------- #
def load_model(args):
    device = args.device
    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"

    if device == "cuda":
        name = args.model or "small"
        compute = args.compute_type or "float16"
        try:
            print(f"Loading Whisper '{name}' on GPU (CUDA)...")
            model = WhisperModel(name, device="cuda", compute_type=compute)
            # warm-up forces the CUDA libraries to load now, not mid-video
            list(model.transcribe(np.zeros(16000, dtype=np.float32), beam_size=1)[0])
            print("GPU ready.")
            return model, "cuda"
        except Exception as e:
            print(f"[warn] GPU not usable ({e}).\n"
                  f"       Did you run: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 ?\n"
                  f"       Falling back to CPU.", file=sys.stderr)

    name = args.model or "base"
    compute = args.compute_type if args.compute_type in ("int8", "float32") else "int8"
    print(f"Loading Whisper '{name}' on CPU...")
    model = WhisperModel(name, device="cpu", compute_type=compute,
                         cpu_threads=max(2, (os.cpu_count() or 4) - 2))
    return model, "cpu"


# --------------------------------------------------------------------------- #
# 4. SPEECH -> TEXT -> TRANSLATION WORKER
# --------------------------------------------------------------------------- #
def processor_loop(args, audio_q, text_q, stop):
    model, device = load_model(args)
    translator = None if args.target == "en" else Translator(args.engine, args.target)
    print(f"\nReady (running on {device.upper()}). Play something with speech!\n")

    max_backlog = int(MAX_BACKLOG_SEC / BLOCK_SEC)
    buf, voiced, silence, total = [], 0.0, 0.0, 0.0

    def handle(audio):
        task = "translate" if args.target == "en" else "transcribe"
        segments, info = model.transcribe(
            audio,
            task=task,
            language=args.source,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            vad_filter=True,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            without_timestamps=True,
        )
        good = [s for s in segments if s.avg_logprob > -1.0 and s.no_speech_prob < 0.6]
        text = " ".join(s.text.strip() for s in good).strip()
        if not text or text.lower() in HALLUCINATIONS:
            return
        if translator and info.language != args.target:
            text = translator(text, info.language)
        text_q.put(text)

    while not stop.is_set():
        # Fell behind real time? Skip stale audio so subtitles stay current.
        if audio_q.qsize() > max_backlog:
            with audio_q.mutex:
                audio_q.queue.clear()
            buf, voiced, silence, total = [], 0.0, 0.0, 0.0
            print("[info] Skipped ahead to stay in sync (try a smaller --model).",
                  file=sys.stderr)

        try:
            block = audio_q.get(timeout=0.2)
        except queue.Empty:
            continue

        dur = len(block) / WHISPER_SR
        rms = float(np.sqrt(np.mean(block ** 2)))
        is_speech = rms > args.threshold

        buf.append(block)
        total += dur
        if is_speech:
            voiced += dur
            silence = 0.0
        else:
            silence += dur

        if voiced == 0 and total > 0.6:  # only silence: keep a short pre-roll
            while total > 0.4 and len(buf) > 1:
                total -= len(buf.pop(0)) / WHISPER_SR
            continue

        if (voiced >= 0.4 and silence >= args.pause) or total >= args.max_seg:
            audio = np.concatenate(buf)
            buf, voiced, silence, total = [], 0.0, 0.0, 0.0
            handle(audio)


# --------------------------------------------------------------------------- #
# 5. SUBTITLE OVERLAY (tkinter)
# --------------------------------------------------------------------------- #
class Overlay:
    def __init__(self, text_q, stop, font_size=28, hold=3.0, bottom_margin=90):
        import tkinter as tk
        self.tk, self.text_q, self.stop = tk, text_q, stop
        self.hold, self.bottom_margin = hold, bottom_margin
        self.lines = collections.deque(maxlen=2)

        self.root = tk.Tk()
        self.root.title("Live Subtitles")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-alpha", 0.82)
        except tk.TclError:
            pass
        self.root.configure(bg="black")

        self.sw = self.root.winfo_screenwidth()
        self.sh = self.root.winfo_screenheight()
        self.label = tk.Label(
            self.root, text="", fg="white", bg="black",
            font=("Segoe UI" if platform.system() == "Windows" else "Helvetica",
                  font_size, "bold"),
            wraplength=int(self.sw * 0.8), justify="center", padx=18, pady=8,
        )
        self.label.pack()
        self.root.withdraw()
        self.visible = False
        self.root.update_idletasks()
        self._make_click_through()
        self.root.after(100, self.poll)

    def _make_click_through(self):
        if platform.system() != "Windows":
            return
        try:
            self.root.update()
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id()) or self.root.winfo_id()
            style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
            style |= 0x00080000 | 0x00000020 | 0x00000080 | 0x08000000
            ctypes.windll.user32.SetWindowLongW(hwnd, -20, style)
        except Exception:
            pass

    def _place(self):
        self.root.update_idletasks()
        w, h = self.root.winfo_reqwidth(), self.root.winfo_reqheight()
        self.root.geometry(f"{w}x{h}+{(self.sw - w) // 2}+{self.sh - h - self.bottom_margin}")

    def poll(self):
        if self.stop.is_set():
            self.root.destroy()
            return
        now = time.time()
        changed = False
        try:
            while True:
                text = self.text_q.get_nowait()
                self.lines.append((text, now + max(self.hold, len(text) * 0.06)))
                changed = True
        except queue.Empty:
            pass

        before = len(self.lines)
        while self.lines and self.lines[0][1] < now:
            self.lines.popleft()
        changed = changed or len(self.lines) != before

        if changed:
            if self.lines:
                self.label.config(text="\n".join(t for t, _ in self.lines))
                self._place()
                if not self.visible:
                    self.root.deiconify()
                    self.visible = True
                self.root.lift()
            elif self.visible:
                self.root.withdraw()
                self.visible = False
        self.root.after(100, self.poll)

    def run(self):
        self.root.mainloop()


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Real-time screen-audio subtitle translator")
    p.add_argument("--target", default="en", help="Subtitle language (en, hi, es, fr, ja, de, zh-CN ...)")
    p.add_argument("--source", default=None, help="Spoken language code; omit to auto-detect")
    p.add_argument("--model", default=None,
                   help="tiny/base/small/medium/large-v3. Default: small on GPU, base on CPU")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="auto = use NVIDIA GPU if available (default)")
    p.add_argument("--compute-type", default=None, help="float16 (GPU) / int8 (CPU)")
    p.add_argument("--engine", default="google", choices=["google", "argos"],
                   help="Translator for non-English targets: google (online) or argos (offline)")
    p.add_argument("--audio-device", dest="audio_device", default=None,
                   help="Capture device name substring")
    p.add_argument("--threshold", type=float, default=0.008, help="Loudness threshold for speech")
    p.add_argument("--pause", type=float, default=0.5, help="Seconds of silence that end a sentence")
    p.add_argument("--max-seg", type=float, default=5.0, help="Max seconds of audio per chunk (lower = less lag)")
    p.add_argument("--font-size", type=int, default=28)
    p.add_argument("--hold", type=float, default=3.0, help="Minimum seconds each subtitle stays")
    p.add_argument("--no-overlay", action="store_true", help="Print subtitles in the terminal only")
    p.add_argument("--list-devices", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.list_devices:
        list_devices()
        return

    device = pick_capture_device(args.audio_device)
    print(f"Capturing audio from: {device.name}")

    audio_q, text_q, stop = queue.Queue(), queue.Queue(), threading.Event()
    for fn, a in ((capture_loop, (device, audio_q, stop)),
                  (processor_loop, (args, audio_q, text_q, stop))):
        threading.Thread(target=fn, args=a, daemon=True).start()

    try:
        if args.no_overlay:
            while True:
                print(">>", text_q.get())
        else:
            Overlay(text_q, stop, font_size=args.font_size, hold=args.hold).run()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        print("\nStopped.")


if __name__ == "__main__":
    main()
