import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from collaborative_intelligence_simple import LocalModel, GlobalModel, CollaborativeLearning

class TestCollaborativeIntelligenceSimple(unittest.TestCase):
    def setUp(self):
        self.local_model = LocalModel()
        self.global_model = GlobalModel()
        self.collaborative_learning = CollaborativeLearning(num_clients=3)

    def test_local_model_train(self):
        data = np.array([[1, 2], [3, 4]])
        labels = np.array([0, 1])
        self.local_model.train(data, labels)
        self.assertIsNotNone(self.local_model.model)

    def test_local_model_predict(self):
        self.local_model.model = MagicMock()
        self.local_model.model.predict.return_value = np.array([0, 1])
        data = np.array([[1, 2], [3, 4]])
        predictions = self.local_model.predict(data)
        self.assertEqual(predictions.tolist(), [0, 1])

    def test_global_model_aggregate(self):
        local_models = [LocalModel() for _ in range(3)]
        for model in local_models:
            model.model = MagicMock()
            model.model.coef_ = np.array([1, 2])
            model.model.intercept_ = np.array([0])
        self.global_model.aggregate(local_models)
        self.assertTrue(np.array_equal(self.global_model.model.coef_, np.array([1, 2])))
        self.assertTrue(np.array_equal(self.global_model.model.intercept_, np.array([0])))

    def test_collaborative_learning_train(self):
        data = [np.array([[1, 2], [3, 4]]) for _ in range(3)]
        labels = [np.array([0, 1]) for _ in range(3)]
        with patch.object(LocalModel, 'train'), patch.object(GlobalModel, 'aggregate'):
            self.collaborative_learning.train(data, labels)
            self.assertEqual(len(self.collaborative_learning.local_models), 3)

    def test_collaborative_learning_predict(self):
        self.collaborative_learning.global_model.model = MagicMock()
        self.collaborative_learning.global_model.model.predict.return_value = np.array([0, 1])
        data = np.array([[1, 2], [3, 4]])
        predictions = self.collaborative_learning.predict(data)
        self.assertEqual(predictions.tolist(), [0, 1])

    def test_empty_data(self):
        data = [np.array([]) for _ in range(3)]
        labels = [np.array([]) for _ in range(3)]
        with self.assertRaises(ValueError):
            self.collaborative_learning.train(data, labels)

    def test_mismatched_data_labels(self):
        data = [np.array([[1, 2], [3, 4]]) for _ in range(3)]
        labels = [np.array([0]) for _ in range(3)]
        with self.assertRaises(ValueError):
            self.collaborative_learning.train(data, labels)

    def test_different_feature_dimensions(self):
        data = [np.array([[1, 2], [3, 4]]), np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 2], [3, 4]])]
        labels = [np.array([0, 1]) for _ in range(3)]
        with self.assertRaises(ValueError):
            self.collaborative_learning.train(data, labels)

if __name__ == '__main__':
    unittest.main()