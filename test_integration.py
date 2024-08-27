import unittest
import numpy as np
from collaborative_intelligence_simple import CollaborativeLearning as SimpleCollaborativeLearning
from enhanced_collaborative_intelligence import EnhancedCollaborativeLearning
from privacy import DifferentialPrivacy

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.num_clients = 3
        self.simple_cl = SimpleCollaborativeLearning(num_clients=self.num_clients)
        self.enhanced_cl = EnhancedCollaborativeLearning(num_clients=self.num_clients)
        self.dp = DifferentialPrivacy(epsilon=0.1, delta=0.01)

    def generate_dummy_data(self):
        data = [np.random.rand(100, 10) for _ in range(self.num_clients)]
        labels = [np.random.randint(0, 2, 100) for _ in range(self.num_clients)]
        return data, labels

    def test_simple_collaborative_learning_workflow(self):
        data, labels = self.generate_dummy_data()
        
        # Train the model
        self.simple_cl.train(data, labels)
        
        # Make predictions
        test_data = np.random.rand(10, 10)
        predictions = self.simple_cl.predict(test_data)
        
        self.assertEqual(len(predictions), 10)
        self.assertTrue(all(pred in [0, 1] for pred in predictions))

    def test_enhanced_collaborative_learning_workflow(self):
        data, labels = self.generate_dummy_data()
        
        # Train the model using federated learning
        self.enhanced_cl.federated_learning(data, labels)
        
        # Make predictions
        test_data = np.random.rand(10, 10)
        predictions = self.enhanced_cl.predict(test_data)
        
        self.assertEqual(len(predictions), 10)
        self.assertTrue(all(pred in [0, 1] for pred in predictions))

    def test_privacy_preserving_workflow(self):
        data, labels = self.generate_dummy_data()
        
        # Apply differential privacy to the data
        private_data = [self.dp.add_noise(client_data) for client_data in data]
        
        # Train the enhanced model with private data
        self.enhanced_cl.train(private_data, labels)
        
        # Make predictions
        test_data = np.random.rand(10, 10)
        predictions = self.enhanced_cl.predict(test_data)
        
        self.assertEqual(len(predictions), 10)
        self.assertTrue(all(pred in [0, 1] for pred in predictions))

    def test_end_to_end_workflow(self):
        # Generate data
        data, labels = self.generate_dummy_data()
        
        # Train simple collaborative learning model
        self.simple_cl.train(data, labels)
        
        # Train enhanced collaborative learning model with federated learning
        self.enhanced_cl.federated_learning(data, labels)
        
        # Apply differential privacy
        private_data = [self.dp.add_noise(client_data) for client_data in data]
        
        # Train enhanced model with private data
        private_enhanced_cl = EnhancedCollaborativeLearning(num_clients=self.num_clients)
        private_enhanced_cl.train(private_data, labels)
        
        # Generate test data
        test_data = np.random.rand(20, 10)
        
        # Make predictions with all models
        simple_predictions = self.simple_cl.predict(test_data)
        enhanced_predictions = self.enhanced_cl.predict(test_data)
        private_enhanced_predictions = private_enhanced_cl.predict(test_data)
        
        # Check predictions
        self.assertEqual(len(simple_predictions), 20)
        self.assertEqual(len(enhanced_predictions), 20)
        self.assertEqual(len(private_enhanced_predictions), 20)
        
        # Compare prediction distributions
        simple_positive = np.sum(simple_predictions)
        enhanced_positive = np.sum(enhanced_predictions)
        private_enhanced_positive = np.sum(private_enhanced_predictions)
        
        # Check if the number of positive predictions is within a reasonable range
        self.assertLess(abs(simple_positive - enhanced_positive), 5)
        self.assertLess(abs(simple_positive - private_enhanced_positive), 5)

if __name__ == '__main__':
    unittest.main()