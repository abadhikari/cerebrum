import logging
import tempfile
import wave

import numpy as np

logger = logging.getLogger(__name__)


class WhisperSpeechToText:
    """
    Speech-to-text backend using faster-whisper.

    Implements a blocking transcription workflow that records audio
    from the default input device until the user signals completion,
    then runs the Whisper model to produce text output.
    """

    def __init__(self, model_name: str, sample_rate: int):
        """
        Initialize the speech-to-text engine.

        Args:
            model_name (str): Name of the Whisper model to load
                (e.g., "small.en").
            sample_rate (int): Recording sample rate in Hz.
        """

        from faster_whisper import WhisperModel

        self.sample_rate = sample_rate
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")

    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Transcribe raw PCM audio using faster-whisper.

        Args:
            audio_data: Single-channel PCM waveform array.

        Returns:
            str: The recognized speech content (may be empty).
        """
        # Write to temp WAV.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            with wave.open(tmp.name, "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)  # int16 = 2 bytes
                f.setframerate(self.sample_rate)
                f.writeframes(audio_data.tobytes())
            wav_path = tmp.name

        # Run Whisper transcription
        segments, _ = self.model.transcribe(wav_path, language="en")
        return "".join(seg.text for seg in segments).strip()
