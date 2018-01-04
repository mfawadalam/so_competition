import json

class JsonDataReader:
    def __init__(self, filename):
        self.filename = filename
        self._data = json.load(open(filename))

    @property
    def data(self):
        return self._data


