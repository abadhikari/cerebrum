import numpy as np
import logging

import sounddevice as sd

logger = logging.getLogger(__name__)


class CliAudioRecorder:
    """
    CLI-focused audio capture utility.

    This class is responsible for microphone recording and returning
    raw PCM audio buffers. It intentionally does not handle transcription
    or any STT logic — that separation keeps audio acquisition independent
    from downstream processing.
    """

    def __init__(self, sample_rate: int):
        """
        Initialize the audio recorder.

        Args:
            sample_rate:
                Target sampling rate (Hz) used when capturing microphone input.
        """
        self._sample_rate = sample_rate

    def record(self) -> np.ndarray | None:
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
            samplerate=self._sample_rate,
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
