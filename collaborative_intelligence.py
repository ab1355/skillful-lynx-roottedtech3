import numpy as np
from typing import List, Dict, Any, Tuple
import random
import asyncio
from collections import defaultdict
from models import SpecializedModel
from task import Task

class Agent:
    def __init__(self, agent_id: str, model: SpecializedModel, specialization: str):
        self.agent_id = agent_id
        self.model = model
        self.specialization = specialization
        self.knowledge_base = {}
        self.task_queue = asyncio.Queue()
        self.reputation = 1.0

    async def process_task(self, task: Task) -> Tuple[str, float]:
        # Simulate task processing
        processing_time = task.complexity * (1 / self.reputation)
        await asyncio.sleep(processing_time)
        success_probability = min(1.0, self.reputation / task.complexity)
        success = random.random() < success_probability
        return task.task_id, 1.0 if success else 0.0

    def update_knowledge(self, new_knowledge: Dict[str, Any]):
        for key, value in new_knowledge.items():
            if key in self.knowledge_base and self.knowledge_base[key] != value:
                # Conflict resolution: keep the knowledge with higher confidence
                old_confidence = self.knowledge_base[key].get('confidence', 0.5)
                new_confidence = value.get('confidence', 0.5)
                if new_confidence > old_confidence:
                    self.knowledge_base[key] = value
            else:
                self.knowledge_base[key] = value

    def decide_knowledge_to_share(self) -> Dict[str, Any]:
        # Share knowledge with confidence above a threshold
        return {k: v for k, v in self.knowledge_base.items() if v.get('confidence', 0) > 0.7}

    async def request_information(self, topic: str, mas: 'MultiAgentSystem'):
        return await mas.handle_information_request(self, topic)

    def train_on_local_data(self, data: np.ndarray):
        self.model.train(data)

    def get_parameters(self) -> Dict[str, np.ndarray]:
        return self.model.get_parameters()

class MultiAgentSystem:
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.task_history = defaultdict(list)

    async def allocate_tasks(self, tasks: List[Task]):
        for task in tasks:
            suitable_agents = [agent for agent in self.agents if agent.specialization == task.domain]
            if suitable_agents:
                chosen_agent = max(suitable_agents, key=lambda a: a.reputation / (len(a.task_queue._queue) + 1))
            else:
                # If no suitable agent, assign to the agent with the highest reputation
                chosen_agent = max(self.agents, key=lambda a: a.reputation / (len(a.task_queue._queue) + 1))
            await chosen_agent.task_queue.put(task)

    async def process_all_tasks(self):
        tasks = [agent.process_task(task) for agent in self.agents for task in agent.task_queue._queue]
        results = await asyncio.gather(*tasks)
        for agent, (task_id, success_rate) in zip(self.agents, results):
            self.task_history[agent.agent_id].append((task_id, success_rate))
            agent.reputation = sum(s for _, s in self.task_history[agent.agent_id][-10:]) / 10  # Moving average of last 10 tasks

    def federated_learning_round(self):
        # Simple averaging of model parameters
        averaged_params = {}
        for param_name in self.agents[0].get_parameters().keys():
            averaged_params[param_name] = np.mean([agent.get_parameters()[param_name] for agent in self.agents], axis=0)

        for agent in self.agents:
            agent.model.set_parameters(averaged_params)

    async def collaborative_exchange(self):
        for agent in self.agents:
            knowledge_to_share = agent.decide_knowledge_to_share()
            for other_agent in self.agents:
                if other_agent != agent:
                    other_agent.update_knowledge(knowledge_to_share)

    async def handle_information_request(self, requesting_agent: Agent, topic: str) -> Any:
        for agent in self.agents:
            if agent != requesting_agent and topic in agent.knowledge_base:
                return agent.knowledge_base[topic]
        return None