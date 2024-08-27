import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import random
import asyncio
import logging
import json
from collections import defaultdict
from agent import Agent
from task import Task
from privacy import secure_aggregate
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import entropy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    def __init__(self, config_file: Optional[str] = None):
        self.num_initial_agents: int = 10
        self.num_steps: int = 50
        self.task_complexity_range: Tuple[float, float] = (0.1, 0.5)
        self.tasks_per_step: int = 10
        self.add_agent_threshold: float = 0.6
        self.remove_agent_threshold: float = 0.2
        self.task_complexity_adjustment_rate: float = 0.01
        self.epsilon: float = 0.5
        self.mentoring_threshold: float = 0.4
        self.mentoring_boost: float = 0.1
        self.knowledge_transfer_rate: float = 0.2
        self.federated_learning_weight: float = 0.3
        self.environmental_factor_range: Tuple[float, float] = (0, 0.2)
        self.collaboration_threshold: float = 0.6

        if config_file:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                self.__dict__.update(config_data)

class MultiAgentSystem:
    def __init__(self, agents: List[Agent], config: Config):
        self.agents = agents
        self.config = config
        self.task_history: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        self.current_time = 0
        self.log: List[str] = []
        self.performance_history: List[float] = []
        self.workload_history: List[Dict[str, int]] = []
        self.specialization_changes: List[Tuple[str, str, str]] = []
        self.long_term_performance: List[float] = []
        self.domain_performance: Dict[str, List[float]] = defaultdict(list)
        self.mentoring_reports: List[Tuple[int, str, str]] = []
        self.task_complexity_adjustment: float = 1.0
        self.domain_specific_complexity: Dict[str, float] = {domain: 1.0 for domain in ["classification", "regression", "clustering", "natural_language_processing", "computer_vision"]}
        self.agent_interactions = nx.Graph()
        self.knowledge_graph = nx.Graph()
        self.collective_hypotheses: List[Tuple[str, str]] = []
        for agent in self.agents:
            agent.creation_time = self.current_time
            self.agent_interactions.add_node(agent.id)

    def _generate_tasks(self) -> List[Task]:
        tasks = []
        for i in range(self.config.tasks_per_step):
            task = Task.generate_random_task(f"Task_{self.current_time}_{i}", self.config.task_complexity_range)
            tasks.append(task)
        return tasks

    def _choose_agent_for_task(self, task: Task, agent_workloads: Dict[Agent, int], domain_workloads: Dict[str, int]) -> Agent:
        suitable_agents = [agent for agent in self.agents if agent.specialization == task.domain and agent.sub_specialization == task.sub_domain]
        if not suitable_agents:
            suitable_agents = [agent for agent in self.agents if agent.specialization == task.domain]
        if not suitable_agents:
            suitable_agents = self.agents

        agent_scores = {}
        for agent in suitable_agents:
            specialization_score = 1 if agent.specialization == task.domain else 0.5
            sub_specialization_score = 1 if agent.sub_specialization == task.sub_domain else 0.75
            performance_score = agent.performance_history[-1] if agent.performance_history else 0
            workload_score = 1 / (agent_workloads[agent] + 1)
            reputation_score = agent.reputation
            uncertainty_score = 1 - agent.express_uncertainty(task)
            curiosity_score = agent.curiosity if task.domain != agent.specialization else 0
            skill_score = agent.skills[task.task_type]
            agent_scores[agent] = (
                specialization_score * sub_specialization_score * performance_score * 
                workload_score * reputation_score * uncertainty_score * skill_score + curiosity_score
            )

        return max(agent_scores, key=agent_scores.get)

    def evaluate_system_performance(self) -> float:
        if not self.agents:
            return 0.0
        
        total_weighted_performance = 0
        total_complexity = 0
        for agent in self.agents:
            if agent.performance_history:
                recent_performances = agent.performance_history[-10:]
                recent_tasks = list(agent.task_queue._queue)[-10:]
                for performance, task in zip(recent_performances, recent_tasks):
                    total_weighted_performance += performance * task.complexity
                    total_complexity += task.complexity

        return total_weighted_performance / total_complexity if total_complexity > 0 else 0

    def federated_learning_round(self):
        aggregated_knowledge = secure_aggregate([agent.knowledge for agent in self.agents])
        agent_weights = [agent.reputation for agent in self.agents]
        total_weight = sum(agent_weights)
        if total_weight == 0:
            agent_weights = [1 for _ in self.agents]
            total_weight = len(self.agents)
        
        normalized_weights = [w / total_weight for w in agent_weights]
        
        for agent, weight in zip(self.agents, normalized_weights):
            agent.knowledge = (1 - self.config.federated_learning_weight) * agent.knowledge + \
                              self.config.federated_learning_weight * (weight * aggregated_knowledge + (1 - weight) * agent.knowledge)

    def detect_biases(self) -> Dict[str, str]:
        domain_knowledge = defaultdict(list)
        for agent in self.agents:
            for domain_idx, domain in enumerate(["classification", "regression", "clustering", "natural_language_processing", "computer_vision"]):
                domain_knowledge[domain].append(np.mean(agent.knowledge[domain_idx]))
        
        biases = {}
        for domain, knowledge in domain_knowledge.items():
            mean_knowledge = np.mean(knowledge)
            std_knowledge = np.std(knowledge)
            if std_knowledge > 0.2:  # Arbitrary threshold
                biases[domain] = f"High variance in {domain} knowledge (std: {std_knowledge:.2f})"
            if mean_knowledge < 0.3:  # Arbitrary threshold
                biases[domain] = f"Low overall knowledge in {domain} (mean: {mean_knowledge:.2f})"
        
        return biases

    def calculate_system_entropy(self) -> Dict[str, float]:
        domain_knowledge = defaultdict(list)
        for agent in self.agents:
            for domain_idx, domain in enumerate(["classification", "regression", "clustering", "natural_language_processing", "computer_vision"]):
                domain_knowledge[domain].append(np.mean(agent.knowledge[domain_idx]))
        
        entropies = {}
        for domain, knowledge in domain_knowledge.items():
            hist, _ = np.histogram(knowledge, bins=10, range=(0, 1), density=True)
            entropies[domain] = entropy(hist + 1e-10)  # Add small constant to avoid log(0)
        
        return entropies

    def analyze_collaboration_patterns(self):
        collaboration_counts = defaultdict(int)
        for edge in self.agent_interactions.edges(data=True):
            collaboration_counts[edge[0]] += 1
            collaboration_counts[edge[1]] += 1
        
        most_collaborative = max(collaboration_counts, key=collaboration_counts.get)
        least_collaborative = min(collaboration_counts, key=collaboration_counts.get)
        
        logging.info(f"Most collaborative agent: {most_collaborative} with {collaboration_counts[most_collaborative]} collaborations")
        logging.info(f"Least collaborative agent: {least_collaborative} with {collaboration_counts[least_collaborative]} collaborations")
        
        plt.figure(figsize=(10, 6))
        plt.bar(collaboration_counts.keys(), collaboration_counts.values())
        plt.title("Agent Collaboration Frequency")
        plt.xlabel("Agent ID")
        plt.ylabel("Number of Collaborations")
        plt.savefig("collaboration_frequency.png")
        plt.close()

    def analyze_learning_strategies(self):
        strategy_counts = defaultdict(int)
        for agent in self.agents:
            strategy_counts[agent.learning_strategy] += 1
        
        plt.figure(figsize=(10, 6))
        plt.bar(strategy_counts.keys(), strategy_counts.values())
        plt.title("Learning Strategy Distribution")
        plt.xlabel("Learning Strategy")
        plt.ylabel("Number of Agents")
        plt.savefig("learning_strategy_distribution.png")
        plt.close()
        
        logging.info("Learning Strategy Distribution:")
        for strategy, count in strategy_counts.items():
            logging.info(f"  {strategy}: {count}")

    def analyze_task_performance(self):
        task_type_performance = defaultdict(list)
        for domain, performances in self.task_history.items():
            for _, performance in performances:
                task_type_performance[domain].append(performance)
        
        plt.figure(figsize=(12, 6))
        plt.boxplot([performances for performances in task_type_performance.values()], labels=task_type_performance.keys())
        plt.title("Task Performance by Domain")
        plt.xlabel("Domain")
        plt.ylabel("Performance")
        plt.savefig("task_performance_by_domain.png")
        plt.close()
        
        logging.info("Task Performance by Domain:")
        for domain, performances in task_type_performance.items():
            logging.info(f"  {domain}: Mean = {np.mean(performances):.4f}, Std = {np.std(performances):.4f}")

    # ... (other methods remain the same)