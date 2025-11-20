import logging
import tempfile
import wave

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class WhisperSpeechToText:
    """
    Speech-to-text backend using faster-whisper.

    Implements a blocking transcription workflow that records audio
    from the default input device until the user signals completion,
    then runs the Whisper model to produce text output.
    """

    def __init__(self, model_name: str, sample_rate: int = 16000):
        """
        Initialize the speech-to-text engine.

        Args:
            model_name (str): Name of the Whisper model to load
                (e.g., "small.en").
            sample_rate (int): Recording sample rate in Hz.
        """
        self.sample_rate = sample_rate
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")

    def transcribe(self) -> str:
        """
        Record audio until interrupted and return the transcribed text.

        This method captures audio from the microphone until the user
        presses ENTER, processes the buffered audio through faster-whisper,
        and returns the resulting transcript.

        Returns:
            str: The recognized speech content.
        """
        print("Recording... Press ENTER to stop.")

        # Capture audio into memory (NumPy array)
        recording = []

        def callback(indata, frames, time_info, status):
            recording.append(indata.copy())

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            callback=callback,
        )
        stream.start()

        # User hits ENTER to stop
        input()

        stream.stop()
        stream.close()

        # Combine chunks
        audio_data = np.concatenate(recording, axis=0)

        # Write to temp WAV
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
