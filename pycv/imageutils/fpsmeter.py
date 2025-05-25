import threading
import datetime
import time

class FPSMeter:
    def __init__(self, averaging_window = 2.0):
        self.timestamps = []
        self.lock = threading.Lock()
        self.averaging_window = averaging_window

    def update(self):
        with self.lock:
            now = datetime.datetime.now()
            timestamp = now.timestamp()
            self.timestamps.append(timestamp)

    def remove_older_entries(self):
        with self.lock:
            now = datetime.datetime.now()
            timestamp = now.timestamp()
            while len(self.timestamps) > 0 and (timestamp - self.timestamps[0]) > self.averaging_window:
                self.timestamps.pop(0)

    def get(self):
        self.remove_older_entries()
        with self.lock:
            if len(self.timestamps) == 0:
                return 0.0
            timestamp = datetime.datetime.now().timestamp()
            return len(self.timestamps) / (timestamp - self.timestamps[0])
