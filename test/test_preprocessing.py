from sklearn.preprocessing import MultiLabelBinarizer
from numpy import testing
import unittest
import sklearn_json as skljson


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.data = [
            {'action', 'drama', 'fantasy'},
            {'comedy', 'horror'},
            {'comedy', 'romance'},
            {'horror'},
            {'mystery', 'thriller'},
            {'sci-fi', 'thriller'},
        ]

    def check_model(self, model):
        expected_results = model.fit_transform(self.data)

        serialized_dict_model = skljson.to_dict(model)
        deserialized_dict_model = skljson.from_dict(serialized_dict_model)

        skljson.to_json(model, 'model.json')
        deserialized_json_model = skljson.from_json('model.json')

        for deserialized_model in [deserialized_dict_model, deserialized_json_model]:
            actual_results = deserialized_model.fit_transform(self.data)

            if model.sparse_output:
                testing.assert_array_equal(expected_results.indptr, actual_results.indptr)
                testing.assert_array_equal(expected_results.indices, actual_results.indices)
                testing.assert_array_equal(expected_results.data, actual_results.data)
            else:
                testing.assert_array_equal(expected_results, actual_results)

    def test_multilabel_binarizer(self):
        self.check_model(MultiLabelBinarizer())
        self.check_model(MultiLabelBinarizer(sparse_output=True))
