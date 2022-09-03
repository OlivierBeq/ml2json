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
        'bytes_limit': memory.bytes_limit,
        'backend': memory.backend,
        'compress': memory.compress,
        'backend_options': memory.backend_options,
        'location': memory.location,
    }
    return serialized_memory


def deserialize_memory(memory_dict):
    memory = Memory(location=memory_dict['location'],
                    backend=memory_dict['backend'],
                    mmap_mode=memory_dict['mmap_mode'],
                    compress=memory_dict['compress'],
                    verbose=memory_dict['_verbose'],
                    bytes_limit=memory_dict['bytes_limit'],
                    backend_options=memory_dict['backend_options'])

    memory.depth = memory_dict['depth']
    memory.timestamp = memory_dict['timestamp']

    return memory
