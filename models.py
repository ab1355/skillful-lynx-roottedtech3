from abc import ABC, abstractmethod
import numpy as np
from typing import Dict

class SpecializedModel(ABC):
    @abstractmethod
    def train(self, data: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def set_parameters(self, parameters: Dict[str, np.ndarray]) -> None:
        pass

class SimpleNNModel(SpecializedModel):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)

    def train(self, data: np.ndarray) -> None:
        # Simplified training process
        self.w1 += np.random.randn(*self.w1.shape) * 0.01
        self.w2 += np.random.randn(*self.w2.shape) * 0.01

    def predict(self, data: np.ndarray) -> np.ndarray:
        h = np.maximum(0, data.dot(self.w1) + self.b1)  # ReLU activation
        return h.dot(self.w2) + self.b2

    def get_parameters(self) -> Dict[str, np.ndarray]:
        return {'w1': self.w1, 'b1': self.b1, 'w2': self.w2, 'b2': self.b2}

    def set_parameters(self, parameters: Dict[str, np.ndarray]) -> None:
        self.w1 = parameters['w1']
        self.b1 = parameters['b1']
        self.w2 = parameters['w2']
        self.b2 = parameters['b2']