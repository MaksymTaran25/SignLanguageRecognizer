"""Non-blocking text-to-speech for the demo.

macOS has the `say` binary built in, so we shell out via subprocess.Popen
(no extra dependency, no main-loop stalling). On Linux/Windows we try
pyttsx3 if it's installed; otherwise speech is silently disabled.
"""

import shutil
import subprocess
import sys
import time


class Speaker:
    def __init__(self, voice=None, rate=200, cooldown=1.5):
        """
        voice: macOS voice name (e.g. 'Samantha', 'Daniel'); ignored on other backends.
        rate:  speaking rate in words per minute.
        cooldown: minimum seconds between two utterances of the SAME word.
        """
        self.cooldown = cooldown
        self.voice = voice
        self.rate = rate
        self._proc = None
        self._last_text = None
        self._last_time = 0.0

        self.backend = None
        if sys.platform == "darwin" and shutil.which("say"):
            self.backend = "say"
        else:
            try:
                import pyttsx3  # noqa: F401
                self.backend = "pyttsx3"
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", rate)
            except Exception:
                self.backend = None  # speech disabled

    @property
    def enabled(self):
        return self.backend is not None

    def is_speaking(self):
        if self.backend == "say" and self._proc is not None:
            return self._proc.poll() is None
        return False

    def speak(self, text):
        """Speak `text` if we're not already speaking and cooldown has passed.

        Returns True if speech was actually started.
        """
        if not self.enabled or not text:
            return False
        now = time.time()
        if text == self._last_text and (now - self._last_time) < self.cooldown:
            return False
        if self.is_speaking():
            return False

        try:
            if self.backend == "say":
                cmd = ["say", "-r", str(self.rate)]
                if self.voice:
                    cmd += ["-v", self.voice]
                cmd.append(text)
                self._proc = subprocess.Popen(cmd,
                                              stdout=subprocess.DEVNULL,
                                              stderr=subprocess.DEVNULL)
            elif self.backend == "pyttsx3":
                self._engine.say(text)
                self._engine.runAndWait()
            self._last_text = text
            self._last_time = now
            return True
        except Exception:
            return False

    def reset(self):
        """Forget the last-spoken word so the next prediction can speak again."""
        self._last_text = None

    def close(self):
        if self.backend == "say" and self._proc is not None:
            try:
                if self._proc.poll() is None:
                    self._proc.terminate()
            except Exception:
                pass
