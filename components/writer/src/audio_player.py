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
        path = os.path.join(os.path.dirname(__file__), '..', 'acoustic-data', 'gamma-40hz-30s-test.mp3')  #'gamma-40hz-30s-test.mp3'   'gamma-40hz-5min-mid.mp3'
        url = QUrl.fromLocalFile(os.path.abspath(path))
        self.player.setMedia(QMediaContent(url))
        # connect the signals
        self._play_signal.connect(self._do_play)
        self._stop_signal.connect(self.player.stop)
        # Connect to media status changes to handle looping
        self.player.mediaStatusChanged.connect(self._handle_media_status)
        self._looping = False
        self._target_duration = 0
        self._elapsed_time = 0
        # Skip the first 10 seconds of warm-up
        self._start_position = 10000  # milliseconds

    def _handle_media_status(self, status):
        if status == QMediaPlayer.EndOfMedia and self._looping:
            # If we're still within our target duration, restart the audio
            if self._elapsed_time < self._target_duration:
                self._do_play()
                self._elapsed_time += 5 * 60  # Add 5 minutes (length of audio file)
            else:
                self._looping = False
                self.stop()

    def _do_play(self):
        # always restart from the warm-up position
        self.player.stop()
        self.player.setPosition(self._start_position)
        self.player.play()

    def play_for(self, minutes: float):
        """
        Play the audio for the specified duration, looping if necessary.
        Starts after the initial warm-up period.
        """
        self._target_duration = minutes * 60  # Convert to seconds
        self._elapsed_time = 0
        self._looping = True
        
        # Start playback
        self._play_signal.emit()
        
        # Schedule the final stop
        timer = threading.Timer(minutes * 60, self._stop_signal.emit)
        timer.daemon = True
        timer.start()

    def stop(self):
        self.player.stop()
