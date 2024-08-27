import numpy as np
from typing import List, Dict, Any, Tuple
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    def __init__(self, config_file=None):
        self.num_initial_agents = 10
        self.num_steps = 50
        self.task_complexity_range = (0.1, 0.5)
        self.tasks_per_step = 10
        self.add_agent_threshold = 0.6
        self.remove_agent_threshold = 0.2
        self.task_complexity_adjustment_rate = 0.01
        self.epsilon = 0.5
        self.mentoring_threshold = 0.4
        self.mentoring_boost = 0.1
        self.knowledge_transfer_rate = 0.2
        self.federated_learning_weight = 0.3

        if config_file:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                self.__dict__.update(config_data)

class MultiAgentSystem:
    def __init__(self, agents: List[Agent], config: Config):
        self.agents = agents
        self.config = config
        self.task_history = defaultdict(list)
        self.current_time = 0
        self.log = []
        self.performance_history = []
        self.workload_history = []
        self.specialization_changes = []
        self.long_term_performance = []
        self.domain_performance = defaultdict(list)
        self.mentoring_reports = []
        self.task_complexity_adjustment = 1.0
        self.domain_specific_complexity = {domain: 1.0 for domain in ["classification", "regression", "clustering"]}
        self.agent_interactions = nx.Graph()
        for agent in self.agents:
            agent.creation_time = self.current_time
            self.agent_interactions.add_node(agent.id)

    async def allocate_tasks(self, tasks: List[Task]):
        agent_workloads = {agent: len(agent.task_queue._queue) for agent in self.agents}
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
            agent_scores[agent] = specialization_score * sub_specialization_score * performance_score * workload_score

        return max(agent_scores, key=agent_scores.get)

    async def process_all_tasks(self):
        for agent in self.agents:
            while not agent.task_queue.empty():
                task = await agent.task_queue.get()
                result = agent.process_task(task)
                agent.update_knowledge(task, result)
                self.task_history[task.domain].append((self.current_time, result))
                agent.performance_history.append(result)
                logging.info(f"Agent {agent.id} processed task {task.id} with performance {result:.4f}")

    async def run_simulation(self, num_steps: int):
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

            for agent in self.agents:
                agent.adapt_learning_strategy()

            performance = self.evaluate_system_performance()
            self.performance_history.append(performance)
            logging.info(f"System performance at step {step}: {performance:.4f}")

            # Gradually increase task complexity
            self._adjust_task_complexity()

        self._visualize_agent_network()
        self._visualize_knowledge_distribution()
        self._visualize_performance_over_time()
        return self.evaluate_system_performance()

    def _generate_tasks(self) -> List[Task]:
        tasks = []
        for i in range(self.config.tasks_per_step):
            domain = random.choice(["classification", "regression", "clustering"])
            sub_domain = random.choice(["basic", "intermediate", "advanced"])
            complexity = random.uniform(*self.config.task_complexity_range) * self.domain_specific_complexity[domain]
            tasks.append(Task(f"Task_{self.current_time}_{i}", complexity, domain, sub_domain))
        return tasks

    def evaluate_system_performance(self):
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

        performance = total_weighted_performance / total_complexity if total_complexity > 0 else 0
        logging.info(f"System performance: {performance:.4f}")
        return performance

    def federated_learning_round(self):
        aggregated_knowledge = secure_aggregate([agent.knowledge for agent in self.agents])
        agent_weights = [agent.performance_history[-1] if agent.performance_history else 0 for agent in self.agents]
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
                performance_diff = agent1.performance_history[-1] - agent2.performance_history[-1] if agent1.performance_history and agent2.performance_history else 0
                specialization_factor = 1 if agent1.specialization == agent2.specialization else 0.5
                sub_specialization_factor = 1 if agent1.sub_specialization == agent2.sub_specialization else 0.75
                exchange_rate = 0.05 * specialization_factor * sub_specialization_factor * abs(performance_diff)
                
                knowledge_diff = agent1.knowledge - agent2.knowledge
                agent1.knowledge -= exchange_rate * knowledge_diff
                agent2.knowledge += exchange_rate * knowledge_diff

                self.agent_interactions.add_edge(agent1.id, agent2.id, weight=exchange_rate)

    def _adjust_agent_specializations(self):
        for agent in self.agents:
            old_spec = agent.specialization
            old_sub_spec = agent.sub_specialization
            agent.adapt_specialization()
            if old_spec != agent.specialization or old_sub_spec != agent.sub_specialization:
                logging.info(f"Agent {agent.id} changed specialization from {old_spec}/{old_sub_spec} to {agent.specialization}/{agent.sub_specialization}")

    def _improve_mentoring(self):
        sorted_agents = sorted(self.agents, key=lambda a: a.performance_history[-1] if a.performance_history else 0, reverse=True)
        mentors = sorted_agents[:len(sorted_agents)//3]
        mentees = sorted_agents[len(sorted_agents)//3:]
        
        for mentor, mentee in zip(mentors, mentees):
            if mentee.performance_history and mentee.performance_history[-1] < self.config.mentoring_threshold:
                mentee.knowledge += self.config.mentoring_boost * (mentor.knowledge - mentee.knowledge)
                self.mentoring_reports.append((self.current_time, mentor.id, mentee.id))
                self.agent_interactions.add_edge(mentor.id, mentee.id, weight=self.config.mentoring_boost)
                logging.info(f"Agent {mentor.id} mentored Agent {mentee.id}")

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
        self.agents = [agent for agent in self.agents if not agent.performance_history or agent.performance_history[-1] > self.config.remove_agent_threshold]
        removed_count = initial_count - len(self.agents)
        if removed_count > 0:
            logging.info(f"Removed {removed_count} underperforming agents")

    def _add_new_agents(self):
        initial_count = len(self.agents)
        while len(self.agents) < self.config.num_initial_agents:
            new_agent = Agent(f"Agent_{self.current_time}_{len(self.agents)}", 
                              random.choice(["classification", "regression", "clustering"]),
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
            for domain_idx, domain in enumerate(["classification", "regression", "clustering"]):
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

# Add other methods here as needed