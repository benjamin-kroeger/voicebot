import pyaudio


class AudioStreamWrapper:
    """
    Wrapper class for audio streams to ensure proper closure in all cases, so that a retry does not have to
    deal with remnant of previous tries
    """

    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)

    def __enter__(self):
        return self.stream

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
