import unittest
import numpy as np
from privacy import secure_aggregate, DifferentialPrivacy, federated_average, SecureMultiPartyComputation, private_set_intersection

class TestPrivacy(unittest.TestCase):
    def test_secure_aggregate(self):
        vectors = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
        result = secure_aggregate(vectors)
        self.assertEqual(result.shape, (3,))
        self.assertTrue(np.all(result >= 0) and np.all(result <= 10))

    def test_secure_aggregate_empty(self):
        vectors = []
        with self.assertRaises(ValueError):
            secure_aggregate(vectors)

    def test_secure_aggregate_single_vector(self):
        vectors = [np.array([1, 2, 3])]
        result = secure_aggregate(vectors)
        self.assertEqual(result.shape, (3,))
        self.assertTrue(np.all(result >= 0) and np.all(result <= 10))

    def test_differential_privacy(self):
        dp = DifferentialPrivacy(epsilon=0.1, delta=0.01)
        data = np.array([1, 2, 3, 4, 5])
        noisy_data = dp.add_noise(data)
        self.assertEqual(data.shape, noisy_data.shape)
        self.assertFalse(np.array_equal(data, noisy_data))

    def test_differential_privacy_empty(self):
        dp = DifferentialPrivacy(epsilon=0.1, delta=0.01)
        data = np.array([])
        noisy_data = dp.add_noise(data)
        self.assertEqual(data.shape, noisy_data.shape)

    def test_differential_privacy_large_epsilon(self):
        dp = DifferentialPrivacy(epsilon=10, delta=0.01)
        data = np.array([1, 2, 3, 4, 5])
        noisy_data = dp.add_noise(data)
        self.assertAlmostEqual(np.mean(np.abs(data - noisy_data)), 0, places=1)

    def test_federated_average(self):
        dp = DifferentialPrivacy(epsilon=0.1, delta=0.01)
        models = [
            {'layer1': np.array([1, 2, 3]), 'layer2': np.array([4, 5, 6])},
            {'layer1': np.array([7, 8, 9]), 'layer2': np.array([10, 11, 12])}
        ]
        result = federated_average(models, dp)
        self.assertEqual(set(result.keys()), {'layer1', 'layer2'})
        self.assertEqual(result['layer1'].shape, (3,))
        self.assertEqual(result['layer2'].shape, (3,))

    def test_federated_average_empty(self):
        dp = DifferentialPrivacy(epsilon=0.1, delta=0.01)
        models = []
        with self.assertRaises(ValueError):
            federated_average(models, dp)

    def test_federated_average_single_model(self):
        dp = DifferentialPrivacy(epsilon=0.1, delta=0.01)
        models = [{'layer1': np.array([1, 2, 3]), 'layer2': np.array([4, 5, 6])}]
        result = federated_average(models, dp)
        self.assertEqual(set(result.keys()), {'layer1', 'layer2'})
        self.assertEqual(result['layer1'].shape, (3,))
        self.assertEqual(result['layer2'].shape, (3,))

    def test_secure_multiparty_computation(self):
        values = [10, 20, 30, 40, 50]
        num_parties = 3
        result = SecureMultiPartyComputation.secure_sum(values, num_parties)
        self.assertAlmostEqual(result, sum(values), delta=0.01)

    def test_secure_multiparty_computation_empty(self):
        values = []
        num_parties = 3
        result = SecureMultiPartyComputation.secure_sum(values, num_parties)
        self.assertEqual(result, 0)

    def test_secure_multiparty_computation_single_party(self):
        values = [10, 20, 30]
        num_parties = 1
        result = SecureMultiPartyComputation.secure_sum(values, num_parties)
        self.assertAlmostEqual(result, sum(values), delta=0.01)

    def test_private_set_intersection(self):
        dp = DifferentialPrivacy(epsilon=0.1, delta=0.01)
        set1 = {1, 2, 3, 4, 5}
        set2 = {3, 4, 5, 6, 7}
        result = private_set_intersection(set1, set2, dp)
        self.assertTrue(result.issubset(set1.union(set2)))
        self.assertTrue(len(result) >= 0 and len(result) <= 7)

    def test_private_set_intersection_empty(self):
        dp = DifferentialPrivacy(epsilon=0.1, delta=0.01)
        set1 = set()
        set2 = {1, 2, 3}
        result = private_set_intersection(set1, set2, dp)
        self.assertEqual(len(result), 0)

    def test_private_set_intersection_identical(self):
        dp = DifferentialPrivacy(epsilon=0.1, delta=0.01)
        set1 = {1, 2, 3}
        set2 = {1, 2, 3}
        result = private_set_intersection(set1, set2, dp)
        self.assertTrue(result.issubset(set1))
        self.assertTrue(len(result) > 0)

if __name__ == '__main__':
    unittest.main()