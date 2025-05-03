import sys
import os
from datetime import datetime, time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLineEdit, QLabel, QComboBox, QSpinBox
)
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import Qt
from audio_player import AudioPlayer
from scheduler import Scheduler

NIGHT_START = time(22, 0)  # 10:00 PM
NIGHT_END   = time(9, 0)   #  9:00 AM

class TimelineWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.jobs = {}  # {"HH:MM": x_position}

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()
        y = rect.height() // 2
        painter.setPen(QPen(Qt.black, 2))
        painter.drawLine(20, y, rect.width() - 20, y)

        total_minutes = ((24 - NIGHT_START.hour) + NIGHT_END.hour) * 60
        for h in range(NIGHT_START.hour, 24):
            self._draw_tick(painter, h, rect, y, total_minutes)
        for h in range(0, NIGHT_END.hour + 1):
            self._draw_tick(painter, h, rect, y, total_minutes)

        painter.setPen(QPen(Qt.blue))
        painter.setBrush(QColor('blue'))
        for x in self.jobs.values():
            painter.drawEllipse(int(x - 5), y - 5, 10, 10)

    def _draw_tick(self, painter, hour, rect, y, total_minutes):
        minutes = ((hour - NIGHT_START.hour) % 24) * 60
        x = int(20 + minutes * (rect.width() - 40) / total_minutes)
        painter.drawLine(x, y - 10, x, y + 10)
        painter.drawText(x - 10, y + 25, f"{hour:02d}")

    def add_marker(self, t_str):
        h, m = map(int, t_str.split(':'))
        minutes = ((h - NIGHT_START.hour) % 24) * 60 + m
        total_minutes = ((24 - NIGHT_START.hour) + NIGHT_END.hour) * 60
        x = 20 + minutes * (self.width() - 40) / total_minutes
        self.jobs[t_str] = x
        self.update()

    def remove_marker(self, t_str):
        if t_str in self.jobs:
            del self.jobs[t_str]
            self.update()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Morpheus Console')
        self.player    = AudioPlayer()
        self.scheduler = Scheduler()
        self.init_ui()

    def init_ui(self):
        container = QWidget()
        layout    = QVBoxLayout()
        self.timeline = TimelineWidget()
        self.timeline.setMinimumHeight(150)
        layout.addWidget(self.timeline)

        ctrl = QHBoxLayout()
        # Time picker
        ctrl.addWidget(QLabel('Time (HH:MM):'))
        self.time_input = QLineEdit('22:00')
        self.time_input.setFixedWidth(60)
        ctrl.addWidget(self.time_input)

        # Duration picker
        ctrl.addWidget(QLabel('Duration (min):'))
        self.dur_input = QSpinBox()
        self.dur_input.setRange(1, 60)
        self.dur_input.setValue(5)
        ctrl.addWidget(self.dur_input)

        # Add button
        self.add_btn = QPushButton('Add')
        self.add_btn.clicked.connect(self.add_time)
        ctrl.addWidget(self.add_btn)

        # Remove dropdown & button
        self.remove_combo = QComboBox()
        self.remove_combo.setFixedWidth(80)
        ctrl.addWidget(self.remove_combo)
        self.remove_btn = QPushButton('Remove')
        self.remove_btn.clicked.connect(self.remove_time)
        ctrl.addWidget(self.remove_btn)

        # Play Now button
        self.play_btn = QPushButton('Play Now')
        self.play_btn.clicked.connect(lambda: self.player.play_for(self.dur_input.value()))
        ctrl.addWidget(self.play_btn)

        layout.addLayout(ctrl)
        container.setLayout(layout)
        self.setCentralWidget(container)

    def add_time(self):
        t_str = self.time_input.text()
        try:
            datetime.strptime(t_str, '%H:%M')
        except ValueError:
            return

        # avoid duplicates
        existing = [job.id for job in self.scheduler._sched.get_jobs()]
        if t_str in existing:
            return

        h, m     = map(int, t_str.split(':'))
        duration = self.dur_input.value()

        # Marker on timeline
        self.timeline.add_marker(t_str)

        # Schedule play_for(duration)
        self.scheduler._sched.add_job(
            self.player.play_for,
            'cron',
            hour=h,
            minute=m,
            args=[duration],
            id=t_str,
            replace_existing=True
        )
        self.remove_combo.addItem(t_str)

    def remove_time(self):
        t_str = self.remove_combo.currentText()
        self.timeline.remove_marker(t_str)
        try:
            self.scheduler._sched.remove_job(t_str)
        except:
            pass
        idx = self.remove_combo.currentIndex()
        self.remove_combo.removeItem(idx)

    def closeEvent(self, event):
        self.scheduler.shutdown()
        event.accept()

if __name__ == '__main__':
    app    = QApplication(sys.argv)
    window = MainWindow()
    window.resize(800, 400)
    window.show()
    sys.exit(app.exec_())
