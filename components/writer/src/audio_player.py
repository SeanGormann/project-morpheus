# src/audio_player.py
import os
import threading
from PyQt5.QtCore import QObject, pyqtSignal, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

class AudioPlayer(QObject):
    """
    Plays the gamma MP3 via QMediaPlayer and stops it after a given number
    of minutes using a Python Timer + Qt signal so everything runs on the
    main Qt thread.
    """
    _play_signal = pyqtSignal()
    _stop_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.player = QMediaPlayer()
        # load your 5-min gamma file
        path = os.path.join(os.path.dirname(__file__), '..', 'acoustic-data', 'gamma-40hz-5min-mid.mp3')
        url = QUrl.fromLocalFile(os.path.abspath(path))
        self.player.setMedia(QMediaContent(url))
        # connect the signals
        self._play_signal.connect(self._do_play)
        self._stop_signal.connect(self.player.stop)

    def _do_play(self):
        # always restart from the top
        self.player.stop()
        self.player.setPosition(0)
        self.player.play()

    def play_for(self, minutes: float):
        """
        Emit the play signal on the Qt thread, then schedule the stop signal
        after `minutes` minutes via a daemon Python Timer.
        """
        # kick off playback on the Qt event loop
        self._play_signal.emit()
        # schedule a stop back on the Qt thread
        timer = threading.Timer(minutes * 60, self._stop_signal.emit)
        timer.daemon = True
        timer.start()

    def stop(self):
        self.player.stop()
