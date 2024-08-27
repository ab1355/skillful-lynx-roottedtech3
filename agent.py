import random
import asyncio
from collections import defaultdict
from typing import Dict, Any, Tuple
import numpy as np
from task import Task
from models import create_model

class Agent:
    def __init__(self, agent_id: str, model_type: str, specialization: str):
        self.agent_id = agent_id
        self.model = create_model(model_type)
        self.specialization = specialization
        self.knowledge_base = {}
        self.task_queue = asyncio.Queue()
        self.reputation = 1.0
        self.task_history = []
        self.last_knowledge_share = 0
        self.last_information_request = 0
        self.performance_by_task_type = defaultdict(lambda: {'success': 0, 'total': 0})
        self.creation_time = 0
        self.specialization_change_cooldown = 0
        self.total_tasks_processed = 0
        self.specialization_strength = 1.0
        self.knowledge_specialization = defaultdict(float)
        self.expertise_level = {
            "classification": 1,
            "regression": 1,
            "clustering": 1
        }
        self.mentors = []
        self.mentees = []
        self.mentoring_impact = defaultdict(list)
        self.ramp_up_boost = 1.0
        self.utilization_score = 1.0
        self.long_term_performance = []
        self.domain_specific_expertise = {domain: 1.0 for domain in ["classification", "regression", "clustering"]}

    async def process_task(self, task: Task) -> Tuple[str, float]:
        processing_time = task.complexity * (1 / (self.reputation * self.ramp_up_boost)) * (1 / self.expertise_level[task.domain])
        await asyncio.sleep(processing_time)
        success_probability = min(1.0, (self.reputation * self.ramp_up_boost * self.specialization_strength * self.expertise_level[task.domain]) / task.complexity)
        success = random.random() < success_probability
        result = 1.0 if success else 0.0
        self.task_history.append((task.task_id, result))
        self.performance_by_task_type[task.domain]['success'] += result
        self.performance_by_task_type[task.domain]['total'] += 1
        self.total_tasks_processed += 1
        
        if success:
            self.generate_knowledge(task)
            self.update_specialization_strength(task.domain, 0.1)
            self.increase_domain_expertise(task.domain, 0.1)
        else:
            self.update_specialization_strength(task.domain, -0.05)
        
        return task.task_id, result

    def generate_knowledge(self, task: Task):
        knowledge_key = f"{task.domain}_technique_{random.randint(1, 100)}"
        confidence = random.uniform(0.6, 1.0) * self.expertise_level[task.domain]
        self.knowledge_base[knowledge_key] = {
            'content': f"Learned technique for {task.domain} tasks",
            'confidence': confidence,
            'domain': task.domain
        }
        self.knowledge_specialization[task.domain] += 0.1

    def update_specialization_strength(self, domain: str, change: float):
        if domain == self.specialization:
            self.specialization_strength = max(0.5, min(2.0, self.specialization_strength + change))

    def get_domain_expertise(self, domain: str) -> float:
        return self.domain_specific_expertise[domain]

    def increase_domain_expertise(self, domain: str, amount: float):
        self.domain_specific_expertise[domain] = min(10.0, self.domain_specific_expertise[domain] + amount)

    def update_utilization_score(self, total_tasks: int):
        if not self.performance_by_task_type:
            self.utilization_score = 1.0  # Default to 1.0 if no tasks have been processed
        else:
            expected_tasks = total_tasks / len(self.performance_by_task_type)
            actual_tasks = self.total_tasks_processed
            self.utilization_score = min(2.0, max(0.5, actual_tasks / max(1, expected_tasks)))

    def decay_ramp_up_boost(self):
        self.ramp_up_boost = max(1.0, self.ramp_up_boost * 0.95)

    def update_long_term_performance(self, performance: float):
        self.long_term_performance.append(performance)
        if len(self.long_term_performance) > 50:
            self.long_term_performance.pop(0)

    def get_parameters(self) -> Dict[str, np.ndarray]:
        return self.model.get_parameters()

    def set_parameters(self, parameters: Dict[str, np.ndarray]):
        self.model.set_parameters(parameters)

    def should_share_knowledge(self, current_time: int) -> bool:
        return current_time - self.last_knowledge_share >= 5

    def should_request_information(self, current_time: int) -> bool:
        return current_time - self.last_information_request >= 7

    def decide_knowledge_to_share(self) -> Dict[str, Any]:
        specialized_knowledge = {k: v for k, v in self.knowledge_base.items() 
                                 if v.get('confidence', 0) > 0.6 and v['domain'] == self.specialization}
        return specialized_knowledge

    def get_performance_difference(self) -> float:
        if len(self.performance_by_task_type) < 2:
            return 0
        performances = [perf['success'] / max(1, perf['total']) for perf in self.performance_by_task_type.values()]
        return max(performances) - min(performances)

    def update_knowledge(self, new_knowledge: Dict[str, Any]):
        for key, value in new_knowledge.items():
            if key not in self.knowledge_base or value['confidence'] > self.knowledge_base[key]['confidence']:
                self.knowledge_base[key] = value
                self.knowledge_specialization[value['domain']] += 0.05

    # ... (other methods remain the same)