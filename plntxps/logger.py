from abc import ABC, abstractmethod
from pathlib import Path
import re

class Logger(ABC):
    @abstractmethod
    def log(self, message):
        pass

class NullLogger(Logger):
    def log(self, message):
        pass

def check_for_file(checked_filepath):
    file_path = Path(checked_filepath)
    if not file_path.is_file():
        return checked_filepath
    for n in range(0, 1000):
        next_guess = re.sub("\.", f"{n}\.", checked_filepath)
        if not Path(next_guess).is_file():
            return next_guess

class FileLogger(Logger):
    def __init__(self, filepath):
        self.filepath = filepath
    def log(self, message):
        with open(self.filepath, 'a') as f:
            # todo: timestamps would be helpful
            f.write(f"{message}\n")
            f.close()

class PrintLogger(Logger):
    def log(self, message):
        print(message)

