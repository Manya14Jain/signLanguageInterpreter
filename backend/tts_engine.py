"""
Cross-platform Text-to-Speech engine.
Uses pyttsx3 via a subprocess worker to avoid Streamlit threading conflicts.
Works on Windows (SAPI5), macOS (NSSpeechSynthesizer), Linux (espeak).
"""

import os
import sys
import subprocess
from typing import Optional

# Absolute path to the worker so it works regardless of CWD
_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speak_worker.py")


class TTSEngine:
    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self.available = self._check()

    def _check(self) -> bool:
        """Verify pyttsx3 is importable on this machine."""
        try:
            import pyttsx3  # noqa: F401
            return True
        except ImportError:
            return False

    def speak(self, text: str):
        """Start speaking text non-blocking. Cancels any ongoing speech first."""
        text = text.strip()
        if not text or not self.available:
            return
        self.stop()
        self._proc = subprocess.Popen(
            [sys.executable, _WORKER, text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop(self):
        """Cancel ongoing speech."""
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None

    def is_speaking(self) -> bool:
        return self._proc is not None and self._proc.poll() is None
