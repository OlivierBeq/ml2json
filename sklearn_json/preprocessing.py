from sklearn.preprocessing import MultiLabelBinarizer


def serialize_multilabel_binarizer(model):
    serialized_model = {
        'meta': 'multilabel-binarizer',
        'classes': model.classes_,
        'sparse_output': model.sparse_output,
    }

    return serialized_model


def deserialize_multilabel_binarizer(model_dict):
    model = MultiLabelBinarizer()

    model.classes_ = model_dict['classes']
    model.sparse_output = model_dict['sparse_output']

    return model
