# Live Screen-Audio Translator

Listens to whatever your PC is playing (games, videos, streams, calls) and shows
movie-style subtitles in English, or any language you choose, in real time.

```
System audio ──► chunking (pause detection) ──► Whisper (speech→text, auto language detect)
                                                     │
                       English target: Whisper translates directly
                       Other target:   Google / Argos translates the transcript
                                                     ▼
                                     Transparent always-on-top subtitle overlay
```

## 1. Setup

Requires Python 3.9+.

```bash
python -m venv venv
# Windows:  venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

pip install -r requirements.txt
```

### Platform notes for capturing system audio

| OS | What to do |
|----|------------|
| **Windows** | Works out of the box (WASAPI loopback of your default speakers/headphones). |
| **Linux** | Works with PulseAudio / PipeWire-Pulse (uses the "monitor" of your output). |
| **macOS** | macOS can't capture system audio natively. Install [BlackHole 2ch](https://existential.audio/blackhole/), create a *Multi-Output Device* (speakers + BlackHole) in *Audio MIDI Setup*, set it as your output, then run with `--audio-device BlackHole`. |

### Optional: NVIDIA GPU (much faster, recommended for games)

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
python translator.py --device cuda --model medium
```

## 2. Run

```bash
python translator.py                      # English subtitles, CPU, "small" model
python translator.py --target hi          # Hindi subtitles
python translator.py --target es --engine argos   # Spanish, offline translation
python translator.py --source ja          # force Japanese input (more accurate than auto)
python translator.py --list-devices       # see capture devices
python translator.py --no-overlay         # print to terminal instead of overlay
```

Stop with **Ctrl+C** in the terminal. The overlay is click-through (Windows), so it
never blocks your game.

### Useful options

| Flag | Purpose |
|------|---------|
| `--model` | `tiny`/`base` (fast, rough) → `small` (balanced) → `medium`/`large-v3` (accurate, needs GPU) |
| `--target` | Subtitle language code: `en`, `hi`, `es`, `fr`, `de`, `ja`, `ko`, `ru`, `zh-CN`, ... |
| `--source` | Spoken language; leave out to auto-detect |
| `--threshold` | Raise (e.g. `0.02`) if loud music triggers false subtitles; lower if speech is missed |
| `--pause` | Seconds of silence that end a sentence (lower = faster subtitles, choppier) |
| `--max-seg` | Max chunk length. Lower = lower latency |
| `--font-size`, `--hold` | Subtitle size and minimum on-screen time |

## 3. Tips and limitations

- **Games must run in Borderless Windowed mode.** Overlays can't draw over exclusive fullscreen.
- **Latency** is about 1-3 seconds (it waits for a sentence to end). On CPU use `--model base` or `tiny`
  for speed; with a GPU, `medium` is a great balance.
- If you see `[warn] Too slow for real time`, switch to a smaller model or a GPU.
- Loud background music/effects reduce accuracy. Lower the game's music volume if you can.
- Non-English targets use Google Translate by default (needs internet). `--engine argos` is
  offline (it downloads language packs the first time).
- Whisper's built-in translation only outputs English, so English is the fastest and most
  accurate target (single step).
- The first run downloads the Whisper model (small is about 460 MB).
- Don't use this to bypass anything protected; capturing audio you can already hear on
  your own PC for personal subtitles is the intended use.

## 4. Troubleshooting

- **"No loopback device found"**: run `--list-devices` and pass a name with `--audio-device`.
- **`numpy.fromstring` error on Windows**: make sure `numpy<2` is installed (it's pinned in requirements).
- **CUDA / cuDNN DLL errors**: install the two `nvidia-*` packages above, or use `--device cpu`.
- **No overlay on Linux**: needs a compositing desktop; Wayland may restrict always-on-top windows.
  Try `--no-overlay` or an X11 session.
