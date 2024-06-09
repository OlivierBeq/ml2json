# -*- coding: utf-8 -*-

import numpy as np
from sklearn.utils import Bunch


def serialize_bunch(bunch: Bunch):
    from ..ml2json import serialize_model
    serialized_bunch = {
        'meta': 'bunch',
        'items': {param: serialize_model(value)
                  for param, value in bunch.items()}
    }
    return serialized_bunch


def deserialize_bunch(bunch_dict):
    from ..ml2json import deserialize_model
    bunch = Bunch(**{param: deserialize_model(value)
                     for param, value in bunch_dict['items'].items()})

    return bunch
