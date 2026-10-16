#!/usr/bin/env python3
"""
Live Translator - GUI app
=========================
Listens to your PC's audio (games, videos, streams) and shows live movie-style
subtitles in a transparent overlay. Run it and use the settings window:

    python translator.py            (or double-click the shortcut / .bat)

Streaming design: while someone is speaking, the subtitle updates continuously
(live captions, ~0.5s behind), then locks in when they pause.
"""

import warnings
warnings.filterwarnings("ignore")
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import sys
import json
import glob
import site
import queue
import ctypes
import platform
import threading
import traceback
import collections
import time

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(APP_DIR, "settings.json")
LOG_FILE = os.path.join(APP_DIR, "translator.log")

# When started with pythonw (no console) stdout/stderr are None -> log to a file
if sys.stdout is None or sys.stderr is None:
    _f = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or _f
    sys.stderr = sys.stderr or _f

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

import tkinter as tk
from tkinter import ttk, messagebox

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
CAPTURE_SR = 48000
WHISPER_SR = 16000
BLOCK_SEC = 0.1
MAX_BACKLOG_SEC = 2.5

HALLUCINATIONS = {
    "you", "thank you.", "thank you", "thanks for watching!", "thanks for watching.",
    "bye.", "bye!", ".", "...", "okay.", "subtitles by the amara.org community",
}

# name -> code used by the translator (Google codes)
TARGET_LANGS = {
    "English": "en", "Hindi": "hi", "Spanish": "es", "French": "fr", "German": "de",
    "Portuguese": "pt", "Italian": "it", "Russian": "ru", "Japanese": "ja",
    "Korean": "ko", "Chinese (Simplified)": "zh-CN", "Arabic": "ar", "Turkish": "tr",
    "Indonesian": "id", "Vietnamese": "vi", "Thai": "th", "Bengali": "bn",
    "Tamil": "ta", "Telugu": "te", "Marathi": "mr", "Urdu": "ur", "Polish": "pl",
    "Dutch": "nl", "Ukrainian": "uk", "Persian": "fa", "Swedish": "sv",
}
# name -> Whisper language code (None = auto-detect)
SOURCE_LANGS = {"Auto-detect": None}
SOURCE_LANGS.update({
    "English": "en", "Hindi": "hi", "Spanish": "es", "French": "fr", "German": "de",
    "Portuguese": "pt", "Italian": "it", "Russian": "ru", "Japanese": "ja",
    "Korean": "ko", "Chinese": "zh", "Arabic": "ar", "Turkish": "tr",
    "Indonesian": "id", "Vietnamese": "vi", "Thai": "th", "Bengali": "bn",
    "Tamil": "ta", "Telugu": "te", "Marathi": "mr", "Urdu": "ur", "Polish": "pl",
    "Dutch": "nl", "Ukrainian": "uk", "Persian": "fa", "Swedish": "sv",
})

LATENCY_PRESETS = {
    "Instant - live captions (~0.5s)": dict(update=0.4, pause=0.35, max_seg=7.0),
    "Fast (~1s)":                      dict(update=0.8, pause=0.4, max_seg=6.0),
    "Balanced":                        dict(update=1.2, pause=0.5, max_seg=8.0),
    "Accurate - sentence by sentence": dict(update=0.0, pause=0.7, max_seg=10.0),
}

MODELS = ["Auto", "tiny", "base", "small", "medium", "large-v3-turbo"]
DEVICES = ["Auto (use GPU if available)", "GPU (NVIDIA CUDA)", "CPU"]
ENGINES = ["Online (Google Translate)", "Offline (Argos)"]
POSITIONS = ["Bottom", "Top"]
DEFAULT_AUDIO = "Default (what my speakers play)"

DEFAULT_SETTINGS = {
    "target": "English",
    "source": "Auto-detect",
    "latency": "Instant - live captions (~0.5s)",
    "model": "Auto",
    "device": DEVICES[0],
    "engine": ENGINES[0],
    "audio_device": DEFAULT_AUDIO,
    "noise": 4,
    "font": 30,
    "position": "Bottom",
    "opacity": 85,
}


def load_settings():
    s = dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            s.update(json.load(f))
    except Exception:
        pass
    return s


def save_settings(s):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Audio capture
# --------------------------------------------------------------------------- #
def list_capture_devices():
    try:
        import soundcard as sc
        return [m.name for m in sc.all_microphones(include_loopback=True)]
    except Exception:
        return []


def pick_capture_device(name):
    import soundcard as sc
    mics = sc.all_microphones(include_loopback=True)
    if name and name != DEFAULT_AUDIO:
        for m in mics:
            if m.name == name:
                return m
        for m in mics:
            if name.lower() in m.name.lower():
                return m
    speaker = sc.default_speaker()
    try:
        return sc.get_microphone(id=str(speaker.name), include_loopback=True)
    except Exception:
        for m in mics:
            if getattr(m, "isloopback", False):
                return m
    raise RuntimeError(
        "No loopback audio device found. Pick one in 'Audio source'. "
        "(On macOS install BlackHole.)")


# --------------------------------------------------------------------------- #
# Translation (non-English targets)
# --------------------------------------------------------------------------- #
class Translator:
    def __init__(self, engine, target):
        self.engine, self.target = engine, target
        self._tried = set()
        self._cache = {}
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
        key = (text, src)
        if key in self._cache:
            return self._cache[key]
        try:
            if self.engine == "google":
                out = self._g.translate(text) or text
            else:
                import argostranslate.translate as tr
                tgt = self.target.split("-")[0]
                self._argos_install(src, tgt)
                self._argos_install(src, "en")
                self._argos_install("en", tgt)
                out = tr.translate(text, src, tgt)
        except Exception as e:
            print(f"[translate error] {e}", file=sys.stderr)
            return text
        if len(self._cache) > 300:
            self._cache.clear()
        self._cache[key] = out
        return out


# --------------------------------------------------------------------------- #
# Model loading (cached, GPU with CPU fallback)
# --------------------------------------------------------------------------- #
_MODEL_CACHE = {}


def get_model(cfg, log):
    from faster_whisper import WhisperModel

    dev_choice = cfg["device"]
    if dev_choice.startswith("CPU"):
        device = "cpu"
    elif dev_choice.startswith("GPU"):
        device = "cuda"
    else:
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"

    wanted = cfg["model"]
    if device == "cuda":
        name = "small" if wanted == "Auto" else wanted
        key = (name, "cuda")
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key], "cuda"
        try:
            log(f"Loading '{name}' model on GPU (first run downloads it)...")
            m = WhisperModel(name, device="cuda", compute_type="float16")
            list(m.transcribe(np.zeros(16000, dtype=np.float32), beam_size=1)[0])  # warm-up
            _MODEL_CACHE[key] = m
            return m, "cuda"
        except Exception as e:
            log(f"GPU not usable ({e}). Falling back to CPU. "
                f"Tip: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12")
            device = "cpu"

    name = "base" if wanted == "Auto" else wanted
    key = (name, "cpu")
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key], "cpu"
    log(f"Loading '{name}' model on CPU (first run downloads it)...")
    m = WhisperModel(name, device="cpu", compute_type="int8",
                     cpu_threads=max(2, (os.cpu_count() or 4) - 2))
    _MODEL_CACHE[key] = m
    return m, "cpu"


# --------------------------------------------------------------------------- #
# Engine: capture -> streaming speech recognition -> (translate) -> UI queue
# --------------------------------------------------------------------------- #
class Engine:
    def __init__(self, cfg, ui_q):
        self.cfg, self.ui_q = cfg, ui_q
        self.stop = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def request_stop(self):
        self.stop.set()

    def status(self, text):
        self.ui_q.put(("status", text))

    def log(self, text):
        self.ui_q.put(("log", text))

    # ---- capture ----
    def _capture(self, audio_q):
        try:
            device = pick_capture_device(self.cfg["audio_device"])
            self.log(f"Capturing audio from: {device.name}")
            frames = int(CAPTURE_SR * BLOCK_SEC)
            factor = CAPTURE_SR // WHISPER_SR
            with device.recorder(samplerate=CAPTURE_SR, channels=2) as rec:
                while not self.stop.is_set():
                    data = rec.record(numframes=frames)
                    mono = data.mean(axis=1).astype(np.float32)
                    n = (len(mono) // factor) * factor
                    audio_q.put(mono[:n].reshape(-1, factor).mean(axis=1))
        except Exception as e:
            self.ui_q.put(("error", f"Audio capture failed: {e}"))
            self.stop.set()

    # ---- translate thread ----
    def _translate_loop(self, tq):
        try:
            tr = Translator("google" if self.cfg["engine"].startswith("Online") else "argos",
                            self.cfg["target_code"])
        except Exception as e:
            self.ui_q.put(("error", f"Translator failed to start: {e}"))
            self.stop.set()
            return
        tgt_base = self.cfg["target_code"].split("-")[0]
        while not self.stop.is_set():
            try:
                items = [tq.get(timeout=0.2)]
            except queue.Empty:
                continue
            while True:
                try:
                    items.append(tq.get_nowait())
                except queue.Empty:
                    break
            # keep every final (in order) + the newest partial after the last final
            keep = [i for i in items if i[0] == "final"]
            if items[-1][0] == "partial":
                keep.append(items[-1])
            for kind, text, lang in keep:
                if text and lang and lang != tgt_base:
                    text = tr(text, lang)
                self.ui_q.put(("sub", kind, text))

    # ---- main worker ----
    def _run(self):
        cfg = self.cfg
        try:
            self.status("Loading speech model...")
            model, dev = get_model(cfg, self.log)
            if self.stop.is_set():
                return
            self.status(f"Listening  |  running on {dev.upper()}")
            self.log(f"Ready on {dev.upper()}. Play something with speech!")

            audio_q = queue.Queue()
            threading.Thread(target=self._capture, args=(audio_q,), daemon=True).start()

            if cfg["target_code"] == "en":
                emit = lambda kind, text, lang: self.ui_q.put(("sub", kind, text))
            else:
                tq = queue.Queue()
                threading.Thread(target=self._translate_loop, args=(tq,), daemon=True).start()
                emit = lambda kind, text, lang: tq.put((kind, text, lang))

            self._process(model, audio_q, emit)
        except Exception as e:
            traceback.print_exc()
            self.ui_q.put(("error", f"{type(e).__name__}: {e}"))
        finally:
            self.stop.set()
            self.ui_q.put(("stopped",))

    def _process(self, model, audio_q, emit):
        cfg = self.cfg
        preset = LATENCY_PRESETS.get(cfg["latency"], LATENCY_PRESETS["Instant - live captions (~0.5s)"])
        update, pause, max_seg = preset["update"], preset["pause"], preset["max_seg"]
        threshold = 0.003 * (1.35 ** (int(cfg["noise"]) - 1))
        max_backlog = int(MAX_BACKLOG_SEC / BLOCK_SEC)
        source = cfg["source_code"]
        task = "translate" if cfg["target_code"] == "en" else "transcribe"

        buf, voiced, silence, total = [], 0.0, 0.0, 0.0
        sticky = None
        last_decode = 0.0

        def decode(audio, final):
            nonlocal sticky
            lang = source if source else (None if final else sticky)
            segments, info = model.transcribe(
                audio, task=task, language=lang, beam_size=1, best_of=1,
                temperature=0.0, vad_filter=True, condition_on_previous_text=False,
                no_speech_threshold=0.6, without_timestamps=True)
            good = [s for s in segments if s.avg_logprob > -1.0 and s.no_speech_prob < 0.6]
            text = " ".join(s.text.strip() for s in good).strip()
            if final and not source and info.language_probability > 0.7:
                sticky = info.language
            if text.lower() in HALLUCINATIONS:
                text = ""
            return text, info.language

        while not self.stop.is_set():
            if audio_q.qsize() > max_backlog:   # fell behind: skip ahead
                with audio_q.mutex:
                    audio_q.queue.clear()
                buf, voiced, silence, total = [], 0.0, 0.0, 0.0

            try:
                block = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue

            dur = len(block) / WHISPER_SR
            speech = float(np.sqrt(np.mean(block ** 2))) > threshold
            buf.append(block)
            total += dur
            if speech:
                voiced += dur
                silence = 0.0
            else:
                silence += dur

            if voiced == 0:
                while total > 0.5 and len(buf) > 1:      # keep short pre-roll only
                    total -= len(buf.pop(0)) / WHISPER_SR
                continue

            if (voiced >= 0.3 and silence >= pause) or total >= max_seg:
                audio = np.concatenate(buf)
                buf, voiced, silence, total = [], 0.0, 0.0, 0.0
                text, lang = decode(audio, True)
                emit("final", text, lang)             # empty text clears the live line
                last_decode = time.time()
            elif (update > 0 and voiced >= 0.4 and silence < pause
                  and time.time() - last_decode >= update and audio_q.qsize() <= 2):
                text, lang = decode(np.concatenate(buf), False)
                last_decode = time.time()
                if text:
                    emit("partial", text, lang)


# --------------------------------------------------------------------------- #
# Subtitle overlay window
# --------------------------------------------------------------------------- #
class Overlay:
    def __init__(self, root):
        self.root = root
        self.top = tk.Toplevel(root)
        self.top.overrideredirect(True)
        self.top.attributes("-topmost", True)
        self.top.configure(bg="black")
        self.sw = root.winfo_screenwidth()
        self.sh = root.winfo_screenheight()
        self.font_size, self.position, self.margin = 30, "Bottom", 90
        self.label = tk.Label(self.top, text="", fg="white", bg="black",
                              wraplength=int(self.sw * 0.8), justify="center",
                              padx=18, pady=8)
        self.label.pack()
        self._set_font()
        self.top.withdraw()
        self.visible = False
        self.done = collections.deque(maxlen=1)   # (text, expiry)
        self.live, self.live_t, self.shown = "", 0.0, ""
        self.top.update_idletasks()
        self._click_through()

    def _set_font(self):
        fam = "Segoe UI" if platform.system() == "Windows" else "Helvetica"
        self.label.config(font=(fam, self.font_size, "bold"))

    def _click_through(self):
        if platform.system() != "Windows":
            return
        try:
            self.top.update()
            hwnd = ctypes.windll.user32.GetParent(self.top.winfo_id()) or self.top.winfo_id()
            style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
            style |= 0x00080000 | 0x00000020 | 0x00000080 | 0x08000000
            ctypes.windll.user32.SetWindowLongW(hwnd, -20, style)
        except Exception:
            pass

    def configure(self, font_size, position, opacity):
        self.font_size, self.position = font_size, position
        self._set_font()
        try:
            self.top.attributes("-alpha", max(0.2, min(1.0, opacity / 100.0)))
        except tk.TclError:
            pass
        self.shown = ""   # force re-layout
        self.render()

    def partial(self, text):
        self.live, self.live_t = text, time.time()
        self.render()

    def commit(self, text):
        if text:
            self.done.append((text, time.time() + max(2.5, len(text) * 0.06)))
        self.live = ""
        self.render()

    def clear(self):
        self.done.clear()
        self.live, self.shown = "", ""
        self.render()

    def tick(self):
        self.render()

    def render(self):
        now = time.time()
        while self.done and self.done[0][1] < now:
            self.done.popleft()
        if self.live and now - self.live_t > 4:
            self.live = ""
        lines = [t for t, _ in self.done] + ([self.live] if self.live else [])
        text = "\n".join(lines)
        if text == self.shown:
            return
        self.shown = text
        if text:
            self.label.config(text=text)
            self.top.update_idletasks()
            w, h = self.top.winfo_reqwidth(), self.top.winfo_reqheight()
            x = (self.sw - w) // 2
            y = self.margin if self.position == "Top" else self.sh - h - self.margin
            self.top.geometry(f"{w}x{h}+{x}+{y}")
            if not self.visible:
                self.top.deiconify()
                self.visible = True
            self.top.lift()
        elif self.visible:
            self.top.withdraw()
            self.visible = False


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
class App:
    BG = "#14161c"
    PANEL = "#1d2029"
    FG = "#e8eaf0"
    MUTED = "#9aa1b2"
    ACCENT = "#4f8cff"
    GOOD = "#35c46b"
    BAD = "#ff5c5c"

    def __init__(self, root):
        self.root = root
        self.cfg = load_settings()
        self.ui_q = queue.Queue()
        self.engine = None
        self.running = False

        root.title("Live Translator")
        root.geometry("580x800")
        root.minsize(540, 720)
        root.configure(bg=self.BG)
        self._style()
        self.overlay = Overlay(root)
        self._build()
        self._apply_appearance()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.report_callback_exception = self._on_tk_error
        self.poll()

    # ---- styling ----
    def _style(self):
        st = ttk.Style()
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure(".", background=self.BG, foreground=self.FG, font=("Segoe UI", 10))
        st.configure("TFrame", background=self.BG)
        st.configure("Panel.TLabelframe", background=self.PANEL, foreground=self.ACCENT,
                     bordercolor="#2a2e3a", relief="solid")
        st.configure("Panel.TLabelframe.Label", background=self.PANEL, foreground=self.ACCENT,
                     font=("Segoe UI", 10, "bold"))
        st.configure("Panel.TFrame", background=self.PANEL)
        st.configure("TLabel", background=self.PANEL, foreground=self.FG)
        st.configure("Muted.TLabel", background=self.PANEL, foreground=self.MUTED, font=("Segoe UI", 9))
        st.configure("Title.TLabel", background=self.BG, foreground=self.FG, font=("Segoe UI", 18, "bold"))
        st.configure("Sub.TLabel", background=self.BG, foreground=self.MUTED)
        st.configure("Status.TLabel", background=self.BG, foreground=self.MUTED)
        st.configure("TCombobox", fieldbackground="#262a36", background="#262a36",
                     foreground=self.FG, arrowcolor=self.FG, bordercolor="#2a2e3a")
        st.map("TCombobox", fieldbackground=[("readonly", "#262a36")],
               foreground=[("readonly", self.FG)])
        self.root.option_add("*TCombobox*Listbox.background", "#262a36")
        self.root.option_add("*TCombobox*Listbox.foreground", self.FG)
        self.root.option_add("*TCombobox*Listbox.selectBackground", self.ACCENT)
        st.configure("Horizontal.TScale", background=self.PANEL, troughcolor="#262a36")
        st.configure("TSpinbox", fieldbackground="#262a36", foreground=self.FG, arrowcolor=self.FG)
        st.configure("Start.TButton", font=("Segoe UI", 12, "bold"), padding=10,
                     background=self.ACCENT, foreground="white", borderwidth=0)
        st.map("Start.TButton", background=[("active", "#6aa0ff"), ("disabled", "#3a4256")])
        st.configure("Stop.TButton", font=("Segoe UI", 12, "bold"), padding=10,
                     background=self.BAD, foreground="white", borderwidth=0)
        st.map("Stop.TButton", background=[("active", "#ff7878"), ("disabled", "#3a4256")])
        st.configure("Small.TButton", padding=5, background="#2b3040", foreground=self.FG,
                     borderwidth=0)
        st.map("Small.TButton", background=[("active", "#3a4256")])

    # ---- layout ----
    def _combo(self, parent, row, label, var, values, hint=None, width=30):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(12, 8), pady=(8, 0))
        cb = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=width)
        cb.grid(row=row, column=1, sticky="ew", padx=(0, 12), pady=(8, 0))
        if hint:
            ttk.Label(parent, text=hint, style="Muted.TLabel", wraplength=330, justify="left").grid(
                row=row + 1, column=1, sticky="w", padx=(0, 12))
        return cb

    def _build(self):
        c = self.cfg
        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True, padx=16, pady=12)

        ttk.Label(outer, text="Live Translator", style="Title.TLabel").pack(anchor="w")
        ttk.Label(outer, text="Real-time subtitles for any audio playing on your PC",
                  style="Sub.TLabel").pack(anchor="w", pady=(0, 8))

        # Language
        f1 = ttk.LabelFrame(outer, text=" Language ", style="Panel.TLabelframe")
        f1.pack(fill="x", pady=5)
        f1.columnconfigure(1, weight=1)
        self.v_target = tk.StringVar(value=c["target"] if c["target"] in TARGET_LANGS else "English")
        self.v_source = tk.StringVar(value=c["source"] if c["source"] in SOURCE_LANGS else "Auto-detect")
        self._combo(f1, 0, "Subtitles in", self.v_target, list(TARGET_LANGS))
        self._combo(f1, 1, "Spoken language", self.v_source, list(SOURCE_LANGS),
                    "Pick the real language if you know it - more accurate than auto-detect.")
        ttk.Label(f1, text="").grid(row=3, column=0, pady=2)

        # Speed
        f2 = ttk.LabelFrame(outer, text=" Speed & accuracy ", style="Panel.TLabelframe")
        f2.pack(fill="x", pady=5)
        f2.columnconfigure(1, weight=1)
        self.v_latency = tk.StringVar(value=c["latency"] if c["latency"] in LATENCY_PRESETS else DEFAULT_SETTINGS["latency"])
        self.v_model = tk.StringVar(value=c["model"] if c["model"] in MODELS else "Auto")
        self.v_device = tk.StringVar(value=c["device"] if c["device"] in DEVICES else DEVICES[0])
        self.v_engine = tk.StringVar(value=c["engine"] if c["engine"] in ENGINES else ENGINES[0])
        self._combo(f2, 0, "Latency", self.v_latency, list(LATENCY_PRESETS),
                    "Instant shows words as they are spoken, then settles. 'Accurate' waits for full sentences.")
        self._combo(f2, 2, "Model", self.v_model, MODELS,
                    "Auto = small on GPU, base on CPU. Bigger = more accurate but slower.")
        self._combo(f2, 4, "Run on", self.v_device, DEVICES)
        self._combo(f2, 5, "Translator", self.v_engine, ENGINES,
                    "Used when subtitles are not English. Offline downloads language packs.")
        ttk.Label(f2, text="").grid(row=7, column=0, pady=2)

        # Appearance
        f3 = ttk.LabelFrame(outer, text=" Subtitle appearance ", style="Panel.TLabelframe")
        f3.pack(fill="x", pady=5)
        f3.columnconfigure(1, weight=1)
        self.v_font = tk.IntVar(value=int(c["font"]))
        self.v_pos = tk.StringVar(value=c["position"] if c["position"] in POSITIONS else "Bottom")
        self.v_opacity = tk.IntVar(value=int(c["opacity"]))
        ttk.Label(f3, text="Text size").grid(row=0, column=0, sticky="w", padx=(12, 8), pady=(8, 0))
        ttk.Spinbox(f3, from_=14, to=72, textvariable=self.v_font, width=6,
                    command=self._apply_appearance).grid(row=0, column=1, sticky="w", pady=(8, 0))
        self._combo(f3, 1, "Position", self.v_pos, POSITIONS, width=12)
        self.v_pos_cb = None
        ttk.Label(f3, text="Background opacity").grid(row=2, column=0, sticky="w", padx=(12, 8), pady=(8, 0))
        ttk.Scale(f3, from_=30, to=100, variable=self.v_opacity, orient="horizontal",
                  command=lambda _=None: self._apply_appearance()).grid(row=2, column=1, sticky="ew", padx=(0, 12), pady=(8, 0))
        bt = ttk.Frame(f3, style="Panel.TFrame")
        bt.grid(row=3, column=1, sticky="w", pady=(8, 10))
        ttk.Button(bt, text="Show test subtitle", style="Small.TButton",
                   command=self.test_subtitle).pack(side="left")
        self.v_font.trace_add("write", lambda *_: self._apply_appearance())
        self.v_pos.trace_add("write", lambda *_: self._apply_appearance())

        # Audio
        f4 = ttk.LabelFrame(outer, text=" Audio ", style="Panel.TLabelframe")
        f4.pack(fill="x", pady=5)
        f4.columnconfigure(1, weight=1)
        devs = [DEFAULT_AUDIO] + list_capture_devices()
        self.v_audio = tk.StringVar(value=c["audio_device"] if c["audio_device"] in devs else DEFAULT_AUDIO)
        self._combo(f4, 0, "Audio source", self.v_audio, devs,
                    "Default captures whatever your speakers/headphones play.")
        self.v_noise = tk.IntVar(value=int(c["noise"]))
        ttk.Label(f4, text="Noise filter").grid(row=2, column=0, sticky="w", padx=(12, 8), pady=(8, 0))
        ttk.Scale(f4, from_=1, to=10, variable=self.v_noise, orient="horizontal").grid(
            row=2, column=1, sticky="ew", padx=(0, 12), pady=(8, 0))
        ttk.Label(f4, text="Raise if music/effects cause false subtitles; lower if speech is missed.",
                  style="Muted.TLabel", wraplength=330, justify="left").grid(
            row=3, column=1, sticky="w", padx=(0, 12), pady=(0, 10))

        # Start / status
        self.btn = ttk.Button(outer, text="Start", style="Start.TButton", command=self.toggle)
        self.btn.pack(fill="x", pady=(10, 4))
        self.status = ttk.Label(outer, text="Idle", style="Status.TLabel")
        self.status.pack(anchor="w")
        ttk.Label(outer, text="Language, speed and audio changes apply the next time you press Start.",
                  style="Sub.TLabel", font=("Segoe UI", 8)).pack(anchor="w")

        # Log
        lf = tk.Frame(outer, bg=self.BG)
        lf.pack(fill="both", expand=True, pady=(6, 0))
        self.log = tk.Text(lf, height=6, bg="#0f1116", fg=self.FG, relief="flat", wrap="word",
                           font=("Consolas", 10), state="disabled", padx=8, pady=6)
        sb = ttk.Scrollbar(lf, command=self.log.yview)
        self.log.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)

    # ---- helpers ----
    def collect_cfg(self):
        cfg = {
            "target": self.v_target.get(), "source": self.v_source.get(),
            "latency": self.v_latency.get(), "model": self.v_model.get(),
            "device": self.v_device.get(), "engine": self.v_engine.get(),
            "audio_device": self.v_audio.get(), "noise": int(float(self.v_noise.get())),
            "font": int(self.v_font.get()), "position": self.v_pos.get(),
            "opacity": int(float(self.v_opacity.get())),
        }
        save_settings(cfg)
        run = dict(cfg)
        run["target_code"] = TARGET_LANGS.get(cfg["target"], "en")
        run["source_code"] = SOURCE_LANGS.get(cfg["source"])
        return run

    def _apply_appearance(self):
        try:
            self.overlay.configure(int(self.v_font.get()), self.v_pos.get(),
                                   float(self.v_opacity.get()))
        except (tk.TclError, ValueError):
            pass

    def test_subtitle(self):
        self._apply_appearance()
        self.overlay.commit("This is how your subtitles will look.")

    def add_log(self, text):
        self.log.config(state="normal")
        self.log.insert("end", text + "\n")
        n = int(self.log.index("end-1c").split(".")[0])
        if n > 300:
            self.log.delete("1.0", f"{n - 300}.0")
        self.log.see("end")
        self.log.config(state="disabled")

    def set_status(self, text, color=None):
        self.status.config(text=text, foreground=color or self.MUTED)

    # ---- start / stop ----
    def toggle(self):
        if self.running:
            self.stop_engine()
        else:
            self.start_engine()

    def start_engine(self):
        cfg = self.collect_cfg()
        self.overlay.clear()
        self.engine = Engine(cfg, self.ui_q)
        self.running = True
        self.btn.config(text="Stop", style="Stop.TButton")
        self.set_status("Starting...", self.ACCENT)
        self.add_log(f"--- Start: {cfg['source']} -> {cfg['target']} | {cfg['latency']} ---")
        self.engine.start()

    def stop_engine(self):
        if self.engine:
            self.engine.request_stop()
        self.btn.config(state="disabled", text="Stopping...")
        self.set_status("Stopping...", self.MUTED)

    def _finished(self):
        self.running = False
        self.btn.config(state="normal", text="Start", style="Start.TButton")
        if not self.status.cget("text").startswith("Error"):
            self.set_status("Idle")
        self.overlay.clear()

    # ---- event pump ----
    def poll(self):
        try:
            while True:
                msg = self.ui_q.get_nowait()
                kind = msg[0]
                if kind == "sub":
                    _, k, text = msg
                    if k == "partial":
                        self.overlay.partial(text)
                    else:
                        self.overlay.commit(text)
                        if text:
                            self.add_log(time.strftime("%H:%M:%S  ") + text)
                elif kind == "status":
                    self.set_status(msg[1], self.GOOD if msg[1].startswith("Listening") else self.ACCENT)
                elif kind == "log":
                    self.add_log(msg[1])
                elif kind == "error":
                    self.add_log("ERROR: " + msg[1])
                    self.set_status("Error - see log", self.BAD)
                    if self.running:
                        messagebox.showerror("Live Translator", msg[1])
                elif kind == "stopped":
                    self._finished()
        except queue.Empty:
            pass
        self.overlay.tick()
        self.root.after(30, self.poll)

    def _on_tk_error(self, exc, val, tb):
        traceback.print_exception(exc, val, tb)
        try:
            self.add_log(f"UI error: {val}")
        except Exception:
            pass

    def on_close(self):
        try:
            if self.engine:
                self.engine.request_stop()
            self.collect_cfg()
        finally:
            self.root.destroy()


def main():
    if platform.system() == "Windows":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
