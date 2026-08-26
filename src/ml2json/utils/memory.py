# -*- coding: utf-8 -*-

import numpy as np
from joblib import Memory


def serialize_memory(memory):
    serialized_memory = {
        'meta': 'memory',
        'depth': memory.depth,
        '_verbose': memory._verbose,
        'mmap_mode': memory.mmap_mode,
        'timestamp': memory.timestamp,
        'backend': memory.backend,
        'compress': memory.compress,
        'backend_options': memory.backend_options,
        'location': memory.location,
    }
    # bytes_limit was removed from joblib's Memory in newer versions
    if hasattr(memory, 'bytes_limit'):
        serialized_memory['bytes_limit'] = memory.bytes_limit
    return serialized_memory


def deserialize_memory(memory_dict):
    kwargs = dict(location=memory_dict['location'],
                 backend=memory_dict['backend'],
                 mmap_mode=memory_dict['mmap_mode'],
                 compress=memory_dict['compress'],
                 verbose=memory_dict['_verbose'],
                 backend_options=memory_dict['backend_options'])
    # bytes_limit was removed from joblib's Memory in newer versions
    if 'bytes_limit' in memory_dict:
        try:
            memory = Memory(bytes_limit=memory_dict['bytes_limit'], **kwargs)
        except TypeError:
            memory = Memory(**kwargs)
    else:
        memory = Memory(**kwargs)

    memory.depth = memory_dict['depth']
    memory.timestamp = memory_dict['timestamp']

    return memory
