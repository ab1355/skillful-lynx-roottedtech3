import numpy as np
import random
import asyncio
from collections import defaultdict

class Agent:
    def __init__(self, agent_id, primary_domain, initial_specialization):
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
        self.reputation = 0.5
        self.curiosity = random.uniform(0.1, 0.9)
        self.uncertainty = defaultdict(lambda: 1.0)
        self.hypotheses = []
        self.skills = self._initialize_skills()
        self.collaboration_preference = random.uniform(0.3, 0.7)

    def _initialize_knowledge(self):
        domains = ["classification", "regression", "clustering", "natural_language_processing", "computer_vision"]
        sub_domains = ["basic", "intermediate", "advanced"]
        aspects = ["theory", "implementation", "optimization"]
        return np.random.rand(len(domains), len(sub_domains), len(aspects)) * 0.1

    def _initialize_skills(self):
        skills = {
            "prediction": random.uniform(0.1, 0.5),
            "optimization": random.uniform(0.1, 0.5),
            "feature_engineering": random.uniform(0.1, 0.5),
            "model_selection": random.uniform(0.1, 0.5),
            "hyperparameter_tuning": random.uniform(0.1, 0.5),
            "data_preprocessing": random.uniform(0.1, 0.5),
            "ensemble_methods": random.uniform(0.1, 0.5),
            "transfer_learning": random.uniform(0.1, 0.5)
        }
        return skills

    def _choose_learning_strategy(self):
        strategies = ["gradient_descent", "reinforcement", "evolutionary", "bayesian_optimization"]
        return random.choice(strategies)

    def process_task(self, task, environmental_factor):
        domain_index = ["classification", "regression", "clustering", "natural_language_processing", "computer_vision"].index(task.domain)
        sub_domain_index = ["basic", "intermediate", "advanced"].index(task.sub_domain)
        relevant_knowledge = self.knowledge[domain_index, sub_domain_index, :]
        skill_factor = self.skills[task.task_type]
        
        performance = (np.mean(relevant_knowledge) * skill_factor * (1 - np.exp(-task.complexity)))
        self.experience += 1
        result = min(1.0, max(0.0, performance + np.random.normal(0, 0.05) - environmental_factor))
        self.uncertainty[f"{task.domain}_{task.sub_domain}_{task.task_type}"] *= 0.9
        return result

    def update_knowledge(self, task, performance):
        domain_index = ["classification", "regression", "clustering", "natural_language_processing", "computer_vision"].index(task.domain)
        sub_domain_index = ["basic", "intermediate", "advanced"].index(task.sub_domain)
        
        learning_factor = 1 / (1 + np.exp(-self.experience / 100))
        
        if self.learning_strategy == "gradient_descent":
            gradient = performance - np.mean(self.knowledge[domain_index, sub_domain_index, :])
            self.knowledge[domain_index, sub_domain_index, :] += self.learning_rate * gradient * learning_factor
        elif self.learning_strategy == "reinforcement":
            reward = performance - 0.5
            self.knowledge[domain_index, sub_domain_index, :] += self.learning_rate * reward * learning_factor
        elif self.learning_strategy == "evolutionary":
            mutation = np.random.normal(0, 0.1, size=self.knowledge[domain_index, sub_domain_index, :].shape)
            self.knowledge[domain_index, sub_domain_index, :] += self.learning_rate * mutation * learning_factor
        elif self.learning_strategy == "bayesian_optimization":
            self.knowledge[domain_index, sub_domain_index, :] = (self.knowledge[domain_index, sub_domain_index, :] + performance) / 2

        self.knowledge = np.clip(self.knowledge, 0, 1)
        self._apply_forgetting()
        self._transfer_learning(task.domain, task.sub_domain)
        self._update_skills(task.task_type, performance)

    def _update_skills(self, task_type, performance):
        self.skills[task_type] = self.skills[task_type] * 0.9 + performance * 0.1

    def adapt_specialization(self):
        domain_performance = np.mean(self.knowledge, axis=(1, 2))
        new_specialization = ["classification", "regression", "clustering", "natural_language_processing", "computer_vision"][np.argmax(domain_performance)]
        
        sub_domain_performance = np.mean(self.knowledge[["classification", "regression", "clustering", "natural_language_processing", "computer_vision"].index(new_specialization)], axis=1)
        new_sub_specialization = ["basic", "intermediate", "advanced"][np.argmax(sub_domain_performance)]

        if new_specialization != self.specialization or new_sub_specialization != self.sub_specialization:
            self.specialization = new_specialization
            self.sub_specialization = new_sub_specialization

    def adapt_learning_strategy(self):
        if len(self.performance_history) < 10:
            return

        recent_performance = self.performance_history[-10:]
        if np.mean(recent_performance[:5]) > np.mean(recent_performance[5:]):
            strategies = ["gradient_descent", "reinforcement", "evolutionary", "bayesian_optimization"]
            strategies.remove(self.learning_strategy)
            self.learning_strategy = random.choice(strategies)

    def _apply_forgetting(self):
        forgetting_rate = 0.001
        self.knowledge *= (1 - forgetting_rate)

    def _transfer_learning(self, domain, sub_domain):
        domain_index = ["classification", "regression", "clustering", "natural_language_processing", "computer_vision"].index(domain)
        sub_domain_index = ["basic", "intermediate", "advanced"].index(sub_domain)
        transfer_rate = 0.1
        for d in range(5):
            for s in range(3):
                if d != domain_index or s != sub_domain_index:
                    self.knowledge[d, s, :] += transfer_rate * self.knowledge[domain_index, sub_domain_index, :]
        self.knowledge = np.clip(self.knowledge, 0, 1)

    def propose_hypothesis(self):
        domain = random.choice(["classification", "regression", "clustering", "natural_language_processing", "computer_vision"])
        sub_domain = random.choice(["basic", "intermediate", "advanced"])
        aspect = random.choice(["theory", "implementation", "optimization"])
        hypothesis = f"Improving {aspect} in {domain}/{sub_domain} will lead to better performance"
        self.hypotheses.append(hypothesis)
        return hypothesis

    def test_hypothesis(self, hypothesis, result):
        if result:
            domain = hypothesis.split()[3].split('/')[0]
            sub_domain = hypothesis.split()[3].split('/')[1]
            aspect = hypothesis.split()[1]
            domain_index = ["classification", "regression", "clustering", "natural_language_processing", "computer_vision"].index(domain)
            sub_domain_index = ["basic", "intermediate", "advanced"].index(sub_domain)
            aspect_index = ["theory", "implementation", "optimization"].index(aspect)
            self.knowledge[domain_index, sub_domain_index, aspect_index] += 0.1
            self.knowledge = np.clip(self.knowledge, 0, 1)
        self.hypotheses.remove(hypothesis)

    def update_reputation(self, performance):
        self.reputation = 0.9 * self.reputation + 0.1 * performance

    def explore(self):
        least_known_domain = np.argmin(np.mean(self.knowledge, axis=(1, 2)))
        least_known_sub_domain = np.argmin(np.mean(self.knowledge[least_known_domain], axis=1))
        return ["classification", "regression", "clustering", "natural_language_processing", "computer_vision"][least_known_domain], ["basic", "intermediate", "advanced"][least_known_sub_domain]

    def decompose_task(self, task):
        subtasks = []
        for aspect in ["theory", "implementation", "optimization"]:
            subtask = type(task)(f"{task.id}_{aspect}", task.complexity * 0.5, task.domain, task.sub_domain, task.task_type)
            subtasks.append(subtask)
        return subtasks

    def mentor(self, mentee, task):
        domain_index = ["classification", "regression", "clustering", "natural_language_processing", "computer_vision"].index(task.domain)
        sub_domain_index = ["basic", "intermediate", "advanced"].index(task.sub_domain)
        knowledge_diff = self.knowledge[domain_index, sub_domain_index, :] - mentee.knowledge[domain_index, sub_domain_index, :]
        mentee.knowledge[domain_index, sub_domain_index, :] += 0.1 * knowledge_diff
        mentee.knowledge = np.clip(mentee.knowledge, 0, 1)
        
        skill_diff = self.skills[task.task_type] - mentee.skills[task.task_type]
        mentee.skills[task.task_type] += 0.1 * skill_diff

    def express_uncertainty(self, task):
        return self.uncertainty[f"{task.domain}_{task.sub_domain}_{task.task_type}"]

    def collaborate(self, collaborator, task):
        combined_knowledge = (self.knowledge + collaborator.knowledge) / 2
        combined_skill = (self.skills[task.task_type] + collaborator.skills[task.task_type]) / 2
        
        domain_index = ["classification", "regression", "clustering", "natural_language_processing", "computer_vision"].index(task.domain)
        sub_domain_index = ["basic", "intermediate", "advanced"].index(task.sub_domain)
        
        performance = np.mean(combined_knowledge[domain_index, sub_domain_index, :]) * combined_skill * (1 - np.exp(-task.complexity))
        return min(1.0, max(0.0, performance + np.random.normal(0, 0.05)))

    def __str__(self):
        return f"Agent {self.id} - Specialization: {self.specialization}/{self.sub_specialization}, Reputation: {self.reputation:.2f}"