import asyncio
import numpy as np
from typing import List, Dict, Any
from task import Task
from models import SpecializedModel, create_model

class Agent:
    def __init__(self, name: str, model_type: str, specialization: str):
        self.name = name
        self.model = create_model(model_type)
        self.specialization = specialization
        self.task_queue = asyncio.Queue()
        self.knowledge = np.random.rand(100)
        self.performance_history = []
        self.learning_rate = 0.1
        self.exploration_rate = 0.1
        self.task_preferences = {'classification': 0, 'regression': 0, 'clustering': 0}
        self.creation_time = 0
        self.utilization_score = 0

    def process_task(self, task: Task) -> float:
        task_vector = np.random.rand(100)  # Simplified task representation
        similarity = np.dot(self.knowledge, task_vector) / (np.linalg.norm(self.knowledge) * np.linalg.norm(task_vector))
        performance = similarity * (1 - np.exp(-self.model.complexity / task.complexity))
        
        self.performance_history.append(performance)
        self.update_task_preferences(task.domain, performance)
        
        return performance

    def update_knowledge(self, task: Task, performance: float):
        task_vector = np.random.rand(100)  # Simplified task representation
        self.knowledge += self.learning_rate * performance * task_vector
        self.knowledge /= np.linalg.norm(self.knowledge)

    def update_task_preferences(self, task_domain: str, performance: float):
        self.task_preferences[task_domain] = (1 - self.learning_rate) * self.task_preferences[task_domain] + self.learning_rate * performance

    def choose_task(self, available_tasks: List[Task]) -> Task:
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(available_tasks)
        else:
            return max(available_tasks, key=lambda t: self.task_preferences[t.domain])

    def adapt_specialization(self):
        if np.random.rand() < self.exploration_rate:
            self.specialization = np.random.choice(['classification', 'regression', 'clustering'])
        else:
            self.specialization = max(self.task_preferences, key=self.task_preferences.get)

    def get_state(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'specialization': self.specialization,
            'knowledge': self.knowledge.tolist(),
            'performance_history': self.performance_history,
            'task_preferences': self.task_preferences,
            'creation_time': self.creation_time,
            'utilization_score': self.utilization_score
        }

    def load_state(self, state: Dict[str, Any]):
        self.name = state['name']
        self.specialization = state['specialization']
        self.knowledge = np.array(state['knowledge'])
        self.performance_history = state['performance_history']
        self.task_preferences = state['task_preferences']
        self.creation_time = state['creation_time']
        self.utilization_score = state['utilization_score']