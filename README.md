# 🤟 ASL Sign Language Detector

A real-time **American Sign Language (ASL) detection system** that runs entirely in your browser via a Streamlit web app. No deep learning training required — the system uses a **geometric rule engine** on top of MediaPipe hand landmarks to classify 24 ASL letters (A–Z, excluding J & Z which require motion).

---

## ✨ Features

- 🎥 **Live webcam** hand tracking at ~30 fps
- 🔤 **24 ASL letters** (A–Z, J & Z excluded — they are dynamic motion signs)
- 🧠 **Zero training** — pure geometric analysis of hand joint angles and finger directions
- 📖 **Word suggestions** — real-time prefix-based autocomplete from the system dictionary
- 🔊 **Text-to-speech** — cross-platform TTS (Windows, macOS, Linux)
- ✍️ **Word builder** — hold a sign steady to auto-append the letter; Space, Backspace, and Clear controls
- 🎨 **Premium dark UI** — glassmorphic design with live finger state display and confidence bars

---

## 📁 Project Structure

```
signLanguageDetector/
├── frontend/
│   └── app.py                  # Streamlit UI
├── backend/
│   ├── gesture_engine.py       # Core geometric rule engine (A–Z)
│   ├── inference.py            # Wrapper: landmarks → letter + confidence
│   ├── word_engine.py          # Prefix-based word suggestions
│   ├── tts_engine.py           # Cross-platform text-to-speech
│   ├── speak_worker.py         # Subprocess worker for pyttsx3
│   ├── models/
│   │   └── hand_landmarker.task  # MediaPipe hand landmark model
│   └── utils/
│       └── hand_tracker.py     # MediaPipe hand tracking + skeleton overlay
├── requirements.txt
└── README.md
```

---

## 🖥️ System Requirements

| Requirement | Minimum |
|---|---|
| Python | 3.9 – 3.11 |
| Webcam | Any USB or built-in camera |
| OS | Windows 10+, macOS 11+, Ubuntu 20.04+ |
| RAM | 4 GB (8 GB recommended) |

---

## 🚀 Setup & Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/Adi15Jain/signLanguageDetector.git
cd signLanguageDetector
```

### Step 2 — Create a virtual environment

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

> You should see `(venv)` appear at the start of your terminal prompt.

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs: `opencv-python`, `mediapipe`, `numpy`, `streamlit`, `pyttsx3`, and `protobuf`.

### Step 4 — Download the MediaPipe hand landmark model

The hand landmarker model file is **not included in the repository** (it is too large for GitHub).

1. Download it from the official MediaPipe page:  
   👉 https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

2. Place the downloaded file here:
   ```
   backend/models/hand_landmarker.task
   ```

### Step 5 — Run the app

```bash
streamlit run frontend/app.py
```

The app will open automatically in your browser at **http://localhost:8501**.

---

## 🎮 How to Use

1. **Start Camera** — Click the `▶ Start Camera` button.
2. **Show a sign** — Hold your hand in front of the webcam with a clear background.
3. **Hold to confirm** — Keep the same sign steady; the orange bar will fill up, then the letter is added to your word.
4. **Word suggestions** — Clickable suggestions appear below the word builder based on what you've typed.
5. **Speak** — Press `🔊 Speak` to hear the built text read aloud.
6. **Word controls** — Use `⎵ Space`, `⌫ Back`, and `🗑 Clear` to edit.

### Tips for best accuracy

- Use a **plain, contrasting background** (e.g. a white wall if you have dark skin, or a dark wall otherwise).
- Ensure **good, even lighting** — avoid strong backlight.
- Hold your hand at a natural distance — **40–70 cm** from the camera.
- J and Z are excluded because they require drawing a letter in the air (motion-based signs).

---

## 🔤 Supported Signs

| A | B | C | D | E | F |
|---|---|---|---|---|---|
| G | H | I | K | L | M |
| N | O | P | Q | R | S |
| T | U | V | W | X | Y |

> J and Z require motion detection and are not supported in this static-pose system.

---

## 🧠 How It Works

Instead of training a neural network, the system uses **geometric rules applied to 21 3D hand landmarks** detected by MediaPipe:

1. **Tip-to-wrist ratio** — Determines if each finger is extended or curled (robust to hand tilt).
2. **PIP joint angles** — Measures how sharply each finger is bent at its middle joint.
3. **Direction vectors** — Detects whether the index finger is pointing up, sideways, or down.
4. **Spread distances** — Palm-normalised distances between fingertips to detect V vs U vs R.
5. **Thumb position** — Detects A vs S vs T based on where the thumb tip sits relative to the fist.
6. **Weighted temporal buffer** — A 20-frame sliding window with linear weights smooths flickery predictions.

---

## 🌐 Cross-platform TTS

Text-to-speech uses `pyttsx3`, which automatically uses:
- **Windows** → Microsoft SAPI5 voice engine
- **macOS** → Apple NSSpeechSynthesizer
- **Linux** → espeak (install with `sudo apt install espeak`)

To test TTS independently:
```bash
python backend/speak_worker.py "Hello world"
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` with the venv active |
| Camera not opening | Check webcam permissions in System Settings / Device Manager |
| `hand_landmarker.task not found` | Download model from Step 4 and place it at `backend/models/` |
| TTS not working on Linux | Install espeak: `sudo apt install espeak` |
| Low accuracy | Improve lighting; try a plain background; hold hand ~50 cm from camera |
| Prediction flickering | Increase the Hold Duration slider in the sidebar |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [MediaPipe](https://developers.google.com/mediapipe) by Google — hand landmark detection
- [Streamlit](https://streamlit.io) — web app framework
- [pyttsx3](https://pyttsx3.readthedocs.io) — cross-platform TTS
