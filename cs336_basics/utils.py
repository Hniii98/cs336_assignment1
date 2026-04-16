
class AutoKeyDictWrapper:
    def __init__(self):
        self.dict: dict[int, bytes] = {}
        self._key: int = 0

    def add_one(self, value: bytes):
        self.dict[self._key] = value
        self._key += 1

    def add_list(self, values: list[bytes]):
        increment = len(values)
        
        for i, b in enumerate(values, start=self._key):
            self.dict[i] = b
        
        self._key += increment

        
        