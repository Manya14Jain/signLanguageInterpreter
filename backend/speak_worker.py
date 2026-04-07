"""
speak_worker.py — minimal subprocess entry-point for pyttsx3 TTS.
Called as:  python speak_worker.py <text to speak>
Running pyttsx3 in its own process avoids threading conflicts with Streamlit.
"""
import sys
import pyttsx3

def main():
    if len(sys.argv) < 2:
        return
    text = " ".join(sys.argv[1:])
    engine = pyttsx3.init()
    engine.setProperty('rate', 165)    # comfortable speaking rate
    engine.setProperty('volume', 1.0)
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    main()
