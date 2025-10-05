from abc import ABC, abstractmethod

class Detector:
    def detect(self, image):
        pass

    @abstractmethod
    def _detect(self, image):
        raise NotImplementedError("Subclasses must implement this method.")