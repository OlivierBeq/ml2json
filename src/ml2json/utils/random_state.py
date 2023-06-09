# -*- coding: utf-8 -*-

import numpy as np


def serialize_random_state(random_state):
    params = random_state.get_state(legacy=False)
    params['state']['key'] = params['state']['key'].tolist()

    return params


def deserialize_random_state(model_dict):
    model_dict['state']['key'] = np.array(model_dict['state']['key'], dtype=np.uint32)

    random_state = np.random.RandomState()
    random_state.set_state(model_dict)

    return random_state
