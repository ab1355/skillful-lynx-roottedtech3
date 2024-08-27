from typing import Dict, Any
import numpy as np

class Task:
    def __init__(self, name: str, complexity: float, domain: str, task_type: str = 'general'):
        self.name = name
        self.complexity = complexity
        self.domain = domain
        self.task_type = task_type
        self.requirements = self._generate_requirements()

    def _generate_requirements(self) -> Dict[str, float]:
        if self.task_type == 'general':
            return {'knowledge': self.complexity}
        elif self.task_type == 'specialized':
            return {
                'knowledge': self.complexity * 1.2,
                f'{self.domain}_skill': self.complexity * 1.5
            }
        elif self.task_type == 'collaborative':
            return {
                'knowledge': self.complexity,
                'collaboration': self.complexity * 0.8
            }
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    def evaluate_performance(self, agent_knowledge: np.ndarray, agent_skills: Dict[str, float]) -> float:
        performance = 0.0
        for req, value in self.requirements.items():
            if req == 'knowledge':
                performance += np.dot(agent_knowledge, np.random.rand(100)) / value
            elif req == 'collaboration':
                performance += agent_skills.get('collaboration', 0) / value
            else:
                performance += agent_skills.get(req, 0) / value
        return performance / len(self.requirements)

    def get_state(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'complexity': self.complexity,
            'domain': self.domain,
            'task_type': self.task_type,
            'requirements': self.requirements
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> 'Task':
        task = cls(state['name'], state['complexity'], state['domain'], state['task_type'])
        task.requirements = state['requirements']
        return task