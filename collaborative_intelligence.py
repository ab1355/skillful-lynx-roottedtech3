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

    async def run_simulation(self, num_steps: int) -> float:
        for step in range(num_steps):
            self.current_time += 1
            
            if step % 10 == 0:
                logging.info(f"Simulation step: {step}/{num_steps}")
                self._log_agent_stats()

            new_tasks = self._generate_tasks()
            await self.allocate_tasks(new_tasks)
            
            await self.process_all_tasks()
            self.federated_learning_round()
            await self.collaborative_exchange()
            self._adjust_agent_specializations()
            self._improve_mentoring()
            self._adjust_learning_rates()
            self._remove_underperforming_agents()
            self._add_new_agents()
            self._handle_hypotheses()
            self._update_knowledge_graph()

            for agent in self.agents:
                agent.adapt_learning_strategy()

            performance = self.evaluate_system_performance()
            self.performance_history.append(performance)

            self._adjust_task_complexity()

        self._visualize_agent_network()
        self._visualize_knowledge_distribution()
        self._visualize_performance_over_time()
        self._visualize_knowledge_graph()
        return self.evaluate_system_performance()

    async def allocate_tasks(self, tasks: List[Task]):
        agent_workloads = {agent: agent.task_queue.qsize() for agent in self.agents}
        domain_workloads = defaultdict(int)
        for task in tasks:
            domain_workloads[task.domain] += 1

        for task in tasks:
            chosen_agent = self._choose_agent_for_task(task, agent_workloads, domain_workloads)
            await chosen_agent.task_queue.put(task)
            agent_workloads[chosen_agent] += 1
            domain_workloads[task.domain] -= 1
        logging.info(f"Time {self.current_time}: Allocated {len(tasks)} tasks")

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

    async def process_all_tasks(self):
        for agent in self.agents:
            while not agent.task_queue.empty():
                task = await agent.task_queue.get()
                environmental_factor = random.uniform(*self.config.environmental_factor_range)
                
                if random.random() < self.config.collaboration_threshold:
                    collaborator = random.choice([a for a in self.agents if a != agent])
                    result = agent.collaborate(collaborator, task)
                    self.agent_interactions.add_edge(agent.id, collaborator.id, weight=result)
                else:
                    result = agent.process_task(task, environmental_factor)
                
                agent.update_knowledge(task, result)
                self.task_history[task.domain].append((self.current_time, result))
                agent.performance_history.append(result)
                agent.update_reputation(result)

    def _generate_tasks(self) -> List[Task]:
        tasks = []
        for i in range(self.config.tasks_per_step):
            task = Task.generate_random_task(f"Task_{self.current_time}_{i}", self.config.task_complexity_range)
            if random.random() < 0.2:  # 20% chance of task decomposition
                tasks.extend(random.choice(self.agents).decompose_task(task))
            else:
                tasks.append(task)
        return tasks

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

    async def collaborative_exchange(self):
        for i, agent1 in enumerate(self.agents):
            for agent2 in self.agents[i+1:]:
                performance_diff = agent1.reputation - agent2.reputation
                specialization_factor = 1 if agent1.specialization == agent2.specialization else 0.5
                sub_specialization_factor = 1 if agent1.sub_specialization == agent2.sub_specialization else 0.75
                exchange_rate = 0.05 * specialization_factor * sub_specialization_factor * abs(performance_diff)
                
                knowledge_diff = agent1.knowledge - agent2.knowledge
                agent1.knowledge -= exchange_rate * knowledge_diff
                agent2.knowledge += exchange_rate * knowledge_diff

                self.agent_interactions.add_edge(agent1.id, agent2.id, weight=exchange_rate)

    def _adjust_agent_specializations(self):
        for agent in self.agents:
            agent.adapt_specialization()

    def _improve_mentoring(self):
        sorted_agents = sorted(self.agents, key=lambda a: a.reputation, reverse=True)
        mentors = sorted_agents[:len(sorted_agents)//3]
        mentees = sorted_agents[len(sorted_agents)//3:]
        
        for mentor, mentee in zip(mentors, mentees):
            if mentee.performance_history and mentee.performance_history[-1] < self.config.mentoring_threshold:
                if mentee.task_queue.empty():
                    # If the mentee's task queue is empty, generate a new task for mentoring
                    task = Task.generate_random_task(f"Mentoring_Task_{self.current_time}", self.config.task_complexity_range)
                else:
                    task = random.choice(list(mentee.task_queue._queue))
                mentor.mentor(mentee, task)
                self.mentoring_reports.append((self.current_time, mentor.id, mentee.id))
                self.agent_interactions.add_edge(mentor.id, mentee.id, weight=self.config.mentoring_boost)

    def _adjust_learning_rates(self):
        for agent in self.agents:
            if len(agent.performance_history) > 1:
                if agent.performance_history[-1] > agent.performance_history[-2]:
                    agent.learning_rate *= 1.1
                else:
                    agent.learning_rate *= 0.9
                agent.learning_rate = max(0.01, min(0.5, agent.learning_rate))

    def _remove_underperforming_agents(self):
        initial_count = len(self.agents)
        self.agents = [agent for agent in self.agents if agent.reputation > self.config.remove_agent_threshold]
        removed_count = initial_count - len(self.agents)
        if removed_count > 0:
            logging.info(f"Removed {removed_count} underperforming agents")

    def _add_new_agents(self):
        initial_count = len(self.agents)
        while len(self.agents) < self.config.num_initial_agents:
            new_agent = Agent(f"Agent_{self.current_time}_{len(self.agents)}", 
                              random.choice(["classification", "regression", "clustering", "natural_language_processing", "computer_vision"]),
                              random.choice(["basic", "intermediate", "advanced"]))
            new_agent.creation_time = self.current_time
            
            if self.agents:
                donor_agent = random.choice(self.agents)
                new_agent.knowledge = donor_agent.knowledge * self.config.knowledge_transfer_rate
            
            self.agents.append(new_agent)
            self.agent_interactions.add_node(new_agent.id)
        
        added_count = len(self.agents) - initial_count
        if added_count > 0:
            logging.info(f"Added {added_count} new agents")

    def _handle_hypotheses(self):
        for agent in self.agents:
            if random.random() < 0.1:  # 10% chance of proposing a hypothesis
                hypothesis = agent.propose_hypothesis()
                self.collective_hypotheses.append((agent.id, hypothesis))

        for agent_id, hypothesis in self.collective_hypotheses[:]:
            if random.random() < 0.2:  # 20% chance of testing a hypothesis
                result = random.random() < 0.5  # 50% chance of success
                agent = next((a for a in self.agents if a.id == agent_id), None)
                if agent:
                    agent.test_hypothesis(hypothesis, result)
                self.collective_hypotheses.remove((agent_id, hypothesis))

    def _update_knowledge_graph(self):
        self.knowledge_graph.clear()
        for agent in self.agents:
            for d1, domain1 in enumerate(["classification", "regression", "clustering", "natural_language_processing", "computer_vision"]):
                for s1, subdomain1 in enumerate(["basic", "intermediate", "advanced"]):
                    for a1, aspect1 in enumerate(["theory", "implementation", "optimization"]):
                        node1 = f"{domain1}_{subdomain1}_{aspect1}"
                        self.knowledge_graph.add_node(node1, weight=agent.knowledge[d1, s1, a1])
                        for d2, domain2 in enumerate(["classification", "regression", "clustering", "natural_language_processing", "computer_vision"]):
                            for s2, subdomain2 in enumerate(["basic", "intermediate", "advanced"]):
                                for a2, aspect2 in enumerate(["theory", "implementation", "optimization"]):
                                    if d1 != d2 or s1 != s2 or a1 != a2:
                                        node2 = f"{domain2}_{subdomain2}_{aspect2}"
                                        weight = np.corrcoef(agent.knowledge[d1, s1, :], agent.knowledge[d2, s2, :])[0, 1]
                                        self.knowledge_graph.add_edge(node1, node2, weight=weight)

    def _log_agent_stats(self):
        avg_performance = sum(agent.performance_history[-1] if agent.performance_history else 0 for agent in self.agents) / len(self.agents)
        avg_knowledge = np.mean([np.mean(agent.knowledge) for agent in self.agents])
        specializations = [agent.specialization for agent in self.agents]
        sub_specializations = [agent.sub_specialization for agent in self.agents]
        spec_counts = {spec: specializations.count(spec) for spec in set(specializations)}
        sub_spec_counts = {sub_spec: sub_specializations.count(sub_spec) for sub_spec in set(sub_specializations)}
        
        logging.info(f"Time {self.current_time}:")
        logging.info(f"  Average Performance: {avg_performance:.4f}")
        logging.info(f"  Average Knowledge: {avg_knowledge:.4f}")
        logging.info(f"  Specializations: {spec_counts}")
        logging.info(f"  Sub-specializations: {sub_spec_counts}")
        logging.info(f"  Number of Agents: {len(self.agents)}")
        logging.info(f"  Current task complexity range: {self.config.task_complexity_range}")

    def _visualize_agent_network(self):
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.agent_interactions)
        nx.draw(self.agent_interactions, pos, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=8, font_weight='bold')
        edge_weights = nx.get_edge_attributes(self.agent_interactions, 'weight')
        nx.draw_networkx_edge_labels(self.agent_interactions, pos, edge_labels=edge_weights)
        plt.title("Agent Interaction Network")
        plt.savefig("agent_network.png")
        plt.close()

    def _visualize_knowledge_distribution(self):
        knowledge_data = []
        for agent in self.agents:
            for domain_idx, domain in enumerate(["classification", "regression", "clustering", "natural_language_processing", "computer_vision"]):
                for subdomain_idx, subdomain in enumerate(["basic", "intermediate", "advanced"]):
                    knowledge_data.append({
                        "Agent": agent.id,
                        "Domain": domain,
                        "Subdomain": subdomain,
                        "Knowledge": np.mean(agent.knowledge[domain_idx, subdomain_idx, :])
                    })
        
        df = pd.DataFrame(knowledge_data)
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.pivot(index="Agent", columns=["Domain", "Subdomain"], values="Knowledge"),
                    cmap="YlOrRd", annot=True, fmt=".2f")
        plt.title("Knowledge Distribution Across Agents and Domains")
        plt.savefig("knowledge_distribution.png")
        plt.close()

    def _visualize_performance_over_time(self):
        plt.figure(figsize=(12, 6))
        for agent in self.agents:
            plt.plot(range(len(agent.performance_history)), agent.performance_history, label=agent.id)
        plt.title("Agent Performance Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Performance")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("agent_performance_over_time.png")
        plt.close()

    def _visualize_knowledge_graph(self):
        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(self.knowledge_graph)
        nx.draw(self.knowledge_graph, pos, with_labels=True, node_color='lightgreen', 
                node_size=300, font_size=6, font_weight='bold')
        edge_weights = nx.get_edge_attributes(self.knowledge_graph, 'weight')
        
        # Handle potential errors in drawing edge labels
        try:
            nx.draw_networkx_edge_labels(self.knowledge_graph, pos, edge_labels=edge_weights)
        except Exception as e:
            logging.warning(f"Error drawing edge labels: {e}")
            logging.warning("Skipping edge labels in knowledge graph visualization.")
        
        plt.title("Knowledge Graph")
        plt.savefig("knowledge_graph.png")
        plt.close()

    def _adjust_task_complexity(self):
        avg_performance = np.mean(self.performance_history[-10:])  # Consider the last 10 performances
        if avg_performance > 0.7:  # If agents are performing well, increase complexity
            self.config.task_complexity_range = (
                min(self.config.task_complexity_range[0] * 1.05, 1.0),
                min(self.config.task_complexity_range[1] * 1.05, 1.0)
            )
        elif avg_performance < 0.3:  # If agents are struggling, decrease complexity
            self.config.task_complexity_range = (
                max(self.config.task_complexity_range[0] * 0.95, 0.1),
                max(self.config.task_complexity_range[1] * 0.95, 0.1)
            )
        logging.info(f"Adjusted task complexity range: {self.config.task_complexity_range}")

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

    async def address_biases(self, biases: Dict[str, str]):
        for domain, bias_description in biases.items():
            logging.info(f"Addressing bias: {bias_description}")
            for agent in self.agents:
                if "High variance" in bias_description:
                    # Encourage knowledge sharing
                    agent.learning_rate *= 1.1
                elif "Low overall knowledge" in bias_description:
                    # Generate more tasks in this domain
                    self.config.tasks_per_step += 1
                    task = Task.generate_random_task(f"Bias_Task_{self.current_time}", self.config.task_complexity_range)
                    await agent.task_queue.put(task)

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

    def visualize_skill_distribution(self):
        skill_data = []
        for agent in self.agents:
            for skill, value in agent.skills.items():
                skill_data.append({
                    "Agent": agent.id,
                    "Skill": skill,
                    "Value": value
                })
        
        df = pd.DataFrame(skill_data)
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.pivot(index="Agent", columns="Skill", values="Value"),
                    cmap="YlOrRd", annot=True, fmt=".2f")
        plt.title("Skill Distribution Across Agents")
        plt.savefig("skill_distribution.png")
        plt.close()

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

# Add other methods here as needed