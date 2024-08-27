import numpy as np
from typing import List, Dict, Any, Tuple
import random
import asyncio
import logging
import json
import pickle
from collections import defaultdict
from agent import Agent
from models import SpecializedModel, create_model
from task import Task
from privacy import secure_aggregate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    def __init__(self, config_file=None):
        self.num_initial_agents = 9
        self.num_steps = 100
        self.task_complexity_range = (0.3, 1.5)
        self.tasks_per_step = 5
        self.add_agent_threshold = 0.7
        self.remove_agent_threshold = 0.4
        self.task_complexity_adjustment_rate = 0.01
        self.epsilon = 0.5

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
        for agent in self.agents:
            agent.creation_time = self.current_time

    async def allocate_tasks(self, tasks: List[Task]):
        try:
            agent_workloads = {agent: len(agent.task_queue._queue) for agent in self.agents}
            domain_workloads = defaultdict(int)
            for task in tasks:
                domain_workloads[task.domain] += 1

            sorted_agents = sorted(self.agents, key=lambda a: (a.utilization_score, -len(a.task_queue._queue)))
            for task in tasks:
                suitable_agents = [agent for agent in sorted_agents if agent.specialization == task.domain]
                if not suitable_agents:
                    suitable_agents = sorted_agents
                chosen_agent = self._choose_agent_for_task(suitable_agents, task, agent_workloads, domain_workloads)
                await chosen_agent.task_queue.put(task)
                agent_workloads[chosen_agent] += 1
                domain_workloads[task.domain] -= 1
            logging.info(f"Time {self.current_time}: Allocated {len(tasks)} tasks")
        except Exception as e:
            logging.error(f"Error in allocate_tasks: {str(e)}")

    # ... (other methods with similar try-except blocks and logging)

    async def run_simulation(self, num_steps: int):
        try:
            for step in range(num_steps):
                self.current_time += 1
                
                if step % 10 == 0:
                    logging.info(f"Simulation step: {step}/{num_steps}")

                new_tasks = [Task(f"Task_{self.current_time}_{i}", random.uniform(*self.config.task_complexity_range), random.choice(["classification", "regression", "clustering"])) for i in range(self.config.tasks_per_step)]
                await self.allocate_tasks(new_tasks)
                
                await self.process_all_tasks()
                self.federated_learning_round()
                await self.collaborative_exchange()
                self._adjust_agent_specializations()
                self._improve_mentoring()

                if self.current_time % 10 == 0:
                    performance = self.evaluate_system_performance()
                    self._update_performance_thresholds()
                    self._long_term_performance_analysis()
                    should_remove, agent_to_remove = self._should_remove_agent()
                    if should_remove:
                        self.remove_agent(agent_to_remove)
                    elif self._should_add_agent():
                        new_agent = Agent(f"Agent_{len(self.agents)+1}", random.choice(['classification', 'regression', 'clustering']), random.choice(['classification', 'regression', 'clustering']))
                        self.add_agent(new_agent)
                    self._adjust_task_complexity_rates()

                if self.current_time % 50 == 0:
                    self._system_wide_optimization()
                    self.save_checkpoint(f"checkpoint_{self.current_time}.pkl")

            return self.evaluate_system_performance()
        except Exception as e:
            logging.error(f"Error in run_simulation: {str(e)}")
            return None

    def save_checkpoint(self, filename: str):
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
            logging.info(f"Checkpoint saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving checkpoint: {str(e)}")

    @staticmethod
    def load_checkpoint(filename: str):
        try:
            with open(filename, 'rb') as f:
                mas = pickle.load(f)
            logging.info(f"Checkpoint loaded from {filename}")
            return mas
        except Exception as e:
            logging.error(f"Error loading checkpoint: {str(e)}")
            return None

    # ... (rest of the methods remain the same)