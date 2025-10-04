from abc import ABC, abstractmethod

from typing import List, Iterable, Optional

class TranslatorBase(ABC):
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 timeout: int = 60
                 ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
    
    def translate_batch(self):
        pass

    @abstractmethod
    def _translate_batch(self):
        raise NotImplementedError