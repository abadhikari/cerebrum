import logging
import tempfile
import wave

import numpy as np
import sounddevice as sd

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

        from faster_whisper import WhisperModel

        self.sample_rate = sample_rate
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")

    def transcribe(self) -> str:
        """
        Record audio until interrupted and return the transcribed text.

        This method captures audio from the microphone until the user
        either presses ENTER or interrupts with Ctrl+C, processes the
        buffered audio through faster-whisper, and returns the transcript.

        Returns:
            str: The recognized speech content (empty string if nothing usable
                 was captured).
        """
        audio_data = self._record_audio() 
        if audio_data is None:
            logger.warning("No audio captured; returning empty transcript.")
            return ""
        return self._transcribe_audio(audio_data)
    
    def _record_audio(self) -> np.ndarray | None:
        """
        Record audio until the user stops.

        Blocks until ENTER (or Ctrl+C) and returns the concatenated
        audio buffer.

        Returns:
            np.ndarray | None: PCM audio data, or None if nothing
            was recorded.
        """
        print("Recording... Press ENTER to stop (Ctrl+C to abort).")

        # Capture audio into memory
        recording: list[np.ndarray] = []

        def callback(indata, frames, time_info, status):
            if status:
                logger.warning("InputStream status: %s", status)
            recording.append(indata.copy())

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            callback=callback,
        )

        try:
            stream.start()
            # User hits ENTER or Ctrl-C to stop
            try:
                input()
            except KeyboardInterrupt:
                print("\nRecording interrupted by user.")
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                logger.exception("Error stopping or closing stream")
        
        if not recording:
            return None

        # Combine chunks
        return np.concatenate(recording, axis=0)

    def _transcribe_audio(self, audio_data: np.ndarray) -> str:
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
