"""
ASL Sign Language Detector — Streamlit Frontend
Premium dark UI · Word suggestions · Text-to-Speech · 17-letter support
"""

import streamlit as st
import cv2
import numpy as np
import os
import sys
import time

sys.path.append(os.path.join(os.getcwd(), 'backend'))
from utils.hand_tracker import HandTracker
from inference import SignLanguagePredictor
from word_engine import WordEngine
from tts_engine import TTSEngine

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ASL Detector",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif!important}
.stApp{background:#0D0D18}
#MainMenu,footer,header{visibility:hidden}

.letter-display{
  font-size:9rem;font-weight:700;text-align:center;line-height:1;
  background:linear-gradient(135deg,#A78BFA,#60A5FA,#34D399);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  filter:drop-shadow(0 0 30px rgba(139,92,246,.5));
}
.letter-none{font-size:6rem;font-weight:300;text-align:center;color:#2E2E55;line-height:1}
.letter-nohand{font-size:1.1rem;text-align:center;color:#2E2E55;padding:2rem 0}

.bar-track{background:#1E1E38;border-radius:999px;height:10px;overflow:hidden;margin:5px 0}
.bar-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,#7C3AED,#3B82F6,#10B981);transition:width .15s ease}
.hold-track{background:#1E1E38;border-radius:999px;height:14px;overflow:hidden;margin:5px 0;border:1px solid #2E2E55}
.hold-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,#F59E0B,#EF4444)}

.word-box{
  font-size:2rem;font-weight:600;letter-spacing:.15em;text-align:center;
  color:#E2E8F0;background:#12122A;border:1px solid #2E2E55;border-radius:12px;
  padding:16px 22px;min-height:66px;word-break:break-all;line-height:1.3;
}
.word-cursor{display:inline-block;width:3px;height:.9em;background:#7C3AED;
  margin-left:4px;vertical-align:middle;animation:blink 1s step-end infinite}
@keyframes blink{50%{opacity:0}}

.finger-row{display:flex;gap:6px;justify-content:center;margin:8px 0}
.fp{display:flex;flex-direction:column;align-items:center;gap:3px;
  padding:7px 9px;border-radius:10px;font-size:.65rem;font-weight:600;
  letter-spacing:.06em;text-transform:uppercase;flex:1}
.fp-on{background:rgba(109,40,217,.3);border:1px solid #7C3AED;color:#C4B5FD}
.fp-off{background:#16162B;border:1px solid #2E2E55;color:#3A3A5C}
.fd{width:10px;height:10px;border-radius:50%}
.fd-on{background:#7C3AED;box-shadow:0 0 7px #7C3AED}
.fd-off{background:#2E2E55}

.sec{font-size:.68rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;
  color:#4A4A6A;margin:12px 0 5px}

div.stButton>button{
  background:linear-gradient(135deg,#7C3AED,#4F46E5)!important;
  color:white!important;border:none!important;border-radius:10px!important;
  font-family:'Space Grotesk',sans-serif!important;font-weight:600!important;
  transition:all .2s!important;
}
div.stButton>button:hover{transform:translateY(-1px);
  box-shadow:0 8px 20px rgba(124,58,237,.4)!important}

/* Suggestion chip buttons — target by container id */
#sug-zone div.stButton>button{
  background:rgba(59,130,246,.15)!important;
  border:1px solid #3B82F6!important;color:#93C5FD!important;
  border-radius:20px!important;font-size:.78rem!important;padding:.3rem .7rem!important;
}
#sug-zone div.stButton>button:hover{background:rgba(59,130,246,.32)!important}

/* TTS button */
#tts-zone div.stButton>button{
  background:linear-gradient(135deg,#059669,#10B981)!important;
  box-shadow:none!important;
}
#tts-zone div.stButton>button:hover{box-shadow:0 6px 16px rgba(16,185,129,.35)!important}
#tts-stop div.stButton>button{
  background:linear-gradient(135deg,#B91C1C,#EF4444)!important;
}

[data-testid="stSidebar"]{background:#0D0D18!important;border-right:1px solid #1E1E38!important}
.sdot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:middle}
.slive{background:#10B981;animation:pulse 1.5s infinite}
.soff{background:#6B7280}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.ref-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:4px;margin-top:6px}
.ref-cell{background:#16162B;border:1px solid #2E2E55;border-radius:7px;
  text-align:center;padding:5px 2px;font-size:.8rem;color:#A78BFA;font-weight:600}
</style>
""", unsafe_allow_html=True)


# ── Renderers ─────────────────────────────────────────────────────────────────

def render_fingers(fs: dict, ph):
    pills = ""
    for name in ('thumb', 'index', 'middle', 'ring', 'pinky'):
        on = fs.get(name, False)
        pills += (f'<div class="fp fp-{"on" if on else "off"}">'
                  f'<div class="fd fd-{"on" if on else "off"}"></div>'
                  f'<span>{name[:3].upper()}</span></div>')
    ph.markdown(f'<div class="finger-row">{pills}</div>', unsafe_allow_html=True)


def render_word(word: str, ph):
    ph.markdown(f'<div class="word-box">{word}<span class="word-cursor"></span></div>',
                unsafe_allow_html=True)


def render_suggestions(suggestions: list[str], word: str):
    """Show suggestion buttons — clicking replaces the last typed partial word."""
    if not suggestions:
        st.markdown('<p style="color:#2E2E55;font-size:.75rem;">No suggestions yet…</p>',
                    unsafe_allow_html=True)
        return
    st.markdown('<div id="sug-zone">', unsafe_allow_html=True)
    cols = st.columns(min(len(suggestions), 6))
    for i, (sug, col) in enumerate(zip(suggestions, cols)):
        with col:
            if st.button(sug, key=f"sug_{i}_{sug}"):
                st.session_state.current_word = word_engine.apply(word, sug)
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in dict(run_camera=False, current_word='', last_letter=None,
                 hold_start=None, hold_progress=0.0, just_added=False).items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource
def load_tracker():   return HandTracker()

@st.cache_resource
def load_predictor(): return SignLanguagePredictor()

@st.cache_resource
def load_word_engine(): return WordEngine()

@st.cache_resource
def load_tts():       return TTSEngine()

tracker     = load_tracker()
predictor   = load_predictor()
word_engine = load_word_engine()
tts         = load_tts()


# ── Current suggestions (computed outside camera loop) ────────────────────────
def _last_partial(text: str) -> str:
    """Extract the last partial word being typed."""
    if not text or text.endswith(' '):
        return ''
    return text.rsplit(' ', 1)[-1]

current_suggestions = word_engine.suggest(_last_partial(st.session_state.current_word))


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤟 ASL Detector")
    st.markdown("---")
    dot = "slive" if st.session_state.run_camera else "soff"
    lbl = "LIVE" if st.session_state.run_camera else "STOPPED"
    st.markdown(f'<span class="sdot {dot}"></span>'
                f'<span style="color:#9CA3AF;font-size:.85rem;font-weight:600">{lbl}</span>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### ⚙️ Settings")
    conf_threshold = st.slider("Min Confidence", 0.30, 0.95, 0.50, 0.05)
    hold_dur       = st.slider("Hold Duration (s)", 0.5, 3.0, 1.5, 0.25)

    st.markdown("---")
    st.markdown("##### 🔤 Supported Signs (A–Z, except J & Z)")
    cells = "".join(f'<div class="ref-cell">{l}</div>'
                    for l in "ABCDEFGHIKLMNOPQRSTUVWXY")
    st.markdown(f'<div class="ref-grid">{cells}</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#2E2E55;font-size:.68rem;margin-top:6px">'
                'J &amp; Z require motion — not detectable geometrically</p>',
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="color:#4A4A6A;font-size:.75rem;line-height:1.6">'
                '💡 <b>Hold</b> a sign steady until the orange bar fills.<br>'
                'Letter is added to your word automatically.<br>'
                '🔊 Use <b>Speak</b> to hear your built text.</p>',
                unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<h1 style="color:#E2E8F0;font-size:1.8rem;font-weight:700;margin-bottom:2px">'
            '🤟 Real-Time ASL Sign Detector</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#4A4A6A;margin-bottom:18px">'
            'Geometric engine · Zero training · Full A–Z coverage · Word suggestions · Text-to-speech</p>',
            unsafe_allow_html=True)

col_cam, col_panel = st.columns([3, 2], gap="large")

# ── Camera column ─────────────────────────────────────────────────────────────
with col_cam:
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶  Start Camera", use_container_width=True):
            st.session_state.run_camera = True
            predictor.reset()
    with c2:
        if st.button("⏹  Stop Camera", use_container_width=True):
            st.session_state.run_camera = False

    frame_ph = st.empty()
    blank = np.zeros((400, 640, 3), dtype=np.uint8)
    cv2.putText(blank, "Press  Start Camera  to begin",
                (70, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (60, 60, 100), 2)
    frame_ph.image(blank, channels="BGR", use_container_width=True)

# ── Right panel ───────────────────────────────────────────────────────────────
with col_panel:
    # Detected letter
    st.markdown('<div class="sec">Detected Sign</div>', unsafe_allow_html=True)
    letter_ph = st.empty()
    letter_ph.markdown('<div class="letter-none">—</div>', unsafe_allow_html=True)

    # Confidence
    st.markdown('<div class="sec">Confidence</div>', unsafe_allow_html=True)
    conf_ph = st.empty()
    conf_ph.markdown('<div class="bar-track"><div class="bar-fill" style="width:0%"></div></div>'
                     '<p style="color:#3A3A5C;font-size:.72rem;margin:2px 0 0">0%</p>',
                     unsafe_allow_html=True)

    # Hold bar
    st.markdown('<div class="sec">Hold-to-Confirm</div>', unsafe_allow_html=True)
    hold_ph = st.empty()
    hold_ph.markdown('<div class="hold-track"><div class="hold-fill" style="width:0%"></div></div>',
                     unsafe_allow_html=True)

    # Finger states
    st.markdown('<div class="sec">Finger States</div>', unsafe_allow_html=True)
    finger_ph = st.empty()
    render_fingers({k: False for k in ('thumb','index','middle','ring','pinky')}, finger_ph)

    st.markdown("---")

    # Word builder
    st.markdown('<div class="sec">Word Builder</div>', unsafe_allow_html=True)
    word_ph = st.empty()
    render_word(st.session_state.current_word, word_ph)

    # Word control buttons
    wb1, wb2, wb3 = st.columns(3)
    with wb1:
        if st.button("⎵  Space", use_container_width=True):
            st.session_state.current_word += ' '
            render_word(st.session_state.current_word, word_ph)
    with wb2:
        if st.button("⌫  Back", use_container_width=True):
            st.session_state.current_word = st.session_state.current_word[:-1]
            render_word(st.session_state.current_word, word_ph)
    with wb3:
        if st.button("🗑  Clear", use_container_width=True):
            st.session_state.current_word = ''
            render_word(st.session_state.current_word, word_ph)

    # ── Word suggestions ──────────────────────────────────────────────────────
    st.markdown('<div class="sec">Word Suggestions</div>', unsafe_allow_html=True)
    render_suggestions(current_suggestions, st.session_state.current_word)

    # ── TTS controls ──────────────────────────────────────────────────────────
    st.markdown('<div class="sec">Text-to-Speech</div>', unsafe_allow_html=True)
    t1, t2 = st.columns(2)
    with t1:
        st.markdown('<div id="tts-zone">', unsafe_allow_html=True)
        if st.button("🔊  Speak", use_container_width=True, key="btn_speak"):
            text = st.session_state.current_word.strip()
            if text:
                tts.speak(text)
        st.markdown('</div>', unsafe_allow_html=True)
    with t2:
        st.markdown('<div id="tts-stop">', unsafe_allow_html=True)
        if st.button("⏹  Stop", use_container_width=True, key="btn_stop"):
            tts.stop()
        st.markdown('</div>', unsafe_allow_html=True)

    if not tts.available:
        st.caption("⚠️ TTS not available on this platform.")


# ── Camera loop ───────────────────────────────────────────────────────────────
if st.session_state.run_camera:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while st.session_state.run_camera:
        ok, frame = cap.read()
        if not ok:
            st.error("❌ Could not read from webcam.")
            break

        frame = cv2.flip(frame, 1)
        frame, landmarks = tracker.process(frame, draw=True)
        letter, conf, fs = predictor.predict(landmarks)

        hand_visible = landmarks is not None
        if conf < conf_threshold:
            letter = None

        if not hand_visible:
            h_f, w_f = frame.shape[:2]
            msg = "Show your hand to the camera"
            (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)
            cv2.putText(frame, msg, ((w_f - tw) // 2, h_f - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (70, 70, 120), 2, cv2.LINE_AA)

        # Letter display
        if letter:
            letter_ph.markdown(f'<div class="letter-display">{letter}</div>',
                                unsafe_allow_html=True)
        elif not hand_visible:
            letter_ph.markdown('<div class="letter-nohand">No hand detected</div>',
                               unsafe_allow_html=True)
        else:
            letter_ph.markdown('<div class="letter-none">—</div>', unsafe_allow_html=True)

        # Confidence bar
        pct = int(conf * 100)
        conf_ph.markdown(
            f'<div class="bar-track"><div class="bar-fill" style="width:{pct}%"></div></div>'
            f'<p style="color:#6B7280;font-size:.72rem;margin:2px 0 0">{pct}%</p>',
            unsafe_allow_html=True)

        # Finger states
        render_fingers(
            fs if fs else {k: False for k in ('thumb','index','middle','ring','pinky')},
            finger_ph)

        # Hold-to-confirm
        now = time.time()
        if letter and letter == st.session_state.last_letter:
            if st.session_state.hold_start is None:
                st.session_state.hold_start = now
            elapsed  = now - st.session_state.hold_start
            progress = min(elapsed / hold_dur, 1.0)
            st.session_state.hold_progress = progress

            if progress >= 1.0 and not st.session_state.just_added:
                st.session_state.current_word += letter
                st.session_state.hold_start    = None
                st.session_state.hold_progress = 0.0
                st.session_state.just_added    = True
                render_word(st.session_state.current_word, word_ph)
        else:
            st.session_state.last_letter   = letter
            st.session_state.hold_start    = None
            st.session_state.hold_progress = 0.0
            st.session_state.just_added    = False

        hp = int(st.session_state.hold_progress * 100)
        hold_ph.markdown(
            f'<div class="hold-track"><div class="hold-fill" style="width:{hp}%"></div></div>',
            unsafe_allow_html=True)

        frame_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

    cap.release()
