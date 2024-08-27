import numpy as np
import asyncio
from typing import List, Dict
from task import Task
import random
import logging

class Agent:
    def __init__(self, agent_id: str, primary_domain: str, initial_specialization: str):
        self.id = agent_id
        self.primary_domain = primary_domain
        self.specialization = initial_specialization
        self.sub_specialization = random.choice(["basic", "intermediate", "advanced"])
        self.task_queue = asyncio.Queue()
        self.performance_history = []
        self.creation_time = 0
        self.learning_rate = 0.1
        self.knowledge = self._initialize_knowledge()
        self.learning_strategy = self._choose_learning_strategy()
        self.experience = 0

    def _initialize_knowledge(self):
        # Knowledge representation: 3D tensor (domain, sub-domain, aspect)
        domains = ["classification", "regression", "clustering"]
        sub_domains = ["basic", "intermediate", "advanced"]
        aspects = ["theory", "implementation", "optimization"]
        return np.random.rand(len(domains), len(sub_domains), len(aspects)) * 0.1  # Start with low knowledge

    def _choose_learning_strategy(self):
        strategies = ["gradient_descent", "reinforcement", "evolutionary"]
        return random.choice(strategies)

    def process_task(self, task: Task) -> float:
        domain_index = ["classification", "regression", "clustering"].index(task.domain)
        sub_domain_index = ["basic", "intermediate", "advanced"].index(task.sub_domain)
        relevant_knowledge = self.knowledge[domain_index, sub_domain_index, :]
        performance = np.mean(relevant_knowledge) * (1 - np.exp(-task.complexity))
        self.experience += 1
        result = min(1.0, max(0.0, performance + np.random.normal(0, 0.05)))
        logging.info(f"Agent {self.id} processed task {task.id} with performance {result:.4f}")
        return result

    def update_knowledge(self, task: Task, performance: float):
        domain_index = ["classification", "regression", "clustering"].index(task.domain)
        sub_domain_index = ["basic", "intermediate", "advanced"].index(task.sub_domain)
        
        learning_factor = 1 / (1 + np.exp(-self.experience / 100))  # Sigmoid function for gradual learning
        
        if self.learning_strategy == "gradient_descent":
            gradient = performance - np.mean(self.knowledge[domain_index, sub_domain_index, :])
            self.knowledge[domain_index, sub_domain_index, :] += self.learning_rate * gradient * learning_factor
        elif self.learning_strategy == "reinforcement":
            reward = performance - 0.5  # Assuming 0.5 as baseline performance
            self.knowledge[domain_index, sub_domain_index, :] += self.learning_rate * reward * learning_factor
        elif self.learning_strategy == "evolutionary":
            mutation = np.random.normal(0, 0.1, size=self.knowledge[domain_index, sub_domain_index, :].shape)
            self.knowledge[domain_index, sub_domain_index, :] += self.learning_rate * mutation * learning_factor

        self.knowledge = np.clip(self.knowledge, 0, 1)
        logging.info(f"Agent {self.id} updated knowledge for domain {task.domain}, sub-domain {task.sub_domain}")

    def adapt_specialization(self):
        domain_performance = np.mean(self.knowledge, axis=(1, 2))
        new_specialization = ["classification", "regression", "clustering"][np.argmax(domain_performance)]
        
        sub_domain_performance = np.mean(self.knowledge[["classification", "regression", "clustering"].index(new_specialization)], axis=1)
        new_sub_specialization = ["basic", "intermediate", "advanced"][np.argmax(sub_domain_performance)]

        if new_specialization != self.specialization or new_sub_specialization != self.sub_specialization:
            logging.info(f"Agent {self.id} changed specialization from {self.specialization}/{self.sub_specialization} to {new_specialization}/{new_sub_specialization}")
            self.specialization = new_specialization
            self.sub_specialization = new_sub_specialization

    def adapt_learning_strategy(self):
        if len(self.performance_history) < 10:
            return

        recent_performance = self.performance_history[-10:]
        if np.mean(recent_performance[:5]) > np.mean(recent_performance[5:]):
            strategies = ["gradient_descent", "reinforcement", "evolutionary"]
            strategies.remove(self.learning_strategy)
            new_strategy = random.choice(strategies)
            logging.info(f"Agent {self.id} changed learning strategy from {self.learning_strategy} to {new_strategy}")
            self.learning_strategy = new_strategy