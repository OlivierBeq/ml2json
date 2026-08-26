# -*- coding: utf-8 -*-

import numpy as np


def serialize_random_state(random_state):
    params = random_state.get_state(legacy=False)
    params['state']['key'] = params['state']['key'].tolist()

    return {'meta': 'random_state', 'random_state': params}


def deserialize_random_state(model_dict):
    params = model_dict['random_state']
    params['state']['key'] = np.array(params['state']['key'], dtype=np.uint32)

    random_state = np.random.RandomState()
    random_state.set_state(params)

    return random_state
