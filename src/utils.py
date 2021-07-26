from collections.abc import Mapping


class LazyDict(Mapping):
    def __init__(self, keys, func):
        self._func = func
        self._keys = set(keys)

    def __getitem__(self, key):
        if key not in self._keys:
            raise KeyError(key)
        return self._func(key)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)