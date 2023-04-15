import numpy as np
import json
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


# Compute onset strength envelope
def qtile(a, q=45, *args, **kwargs):
    return np.percentile(a, q, *args, **kwargs)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)