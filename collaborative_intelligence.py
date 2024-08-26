import numpy as np
from typing import List, Dict, Any, Tuple
import random
import asyncio
from collections import defaultdict
from models import SpecializedModel, create_model
from task import Task
from privacy import secure_aggregate

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

    async def process_task(self, task: Task) -> Tuple[str, float]:
        processing_time = task.complexity * (1 / self.reputation)
        await asyncio.sleep(processing_time)
        success_probability = min(1.0, self.reputation / task.complexity)
        success = random.random() < success_probability
        result = 1.0 if success else 0.0
        self.task_history.append((task.task_id, result))
        return task.task_id, result

    def update_knowledge(self, new_knowledge: Dict[str, Any]):
        for key, value in new_knowledge.items():
            if key in self.knowledge_base and self.knowledge_base[key] != value:
                old_confidence = self.knowledge_base[key].get('confidence', 0.5)
                new_confidence = value.get('confidence', 0.5)
                if new_confidence > old_confidence:
                    self.knowledge_base[key] = value
            else:
                self.knowledge_base[key] = value

    def decide_knowledge_to_share(self) -> Dict[str, Any]:
        return {k: v for k, v in self.knowledge_base.items() if v.get('confidence', 0) > 0.7}

    async def request_information(self, topic: str, mas: 'MultiAgentSystem'):
        self.last_information_request = mas.current_time
        return await mas.handle_information_request(self, topic)

    def train_on_local_data(self, X: np.ndarray, y: np.ndarray):
        self.model.train(X, y)

    def get_parameters(self) -> Dict[str, np.ndarray]:
        return self.model.get_parameters()

    def set_parameters(self, parameters: Dict[str, np.ndarray]):
        self.model.set_parameters(parameters)

    def should_share_knowledge(self, current_time: int) -> bool:
        return current_time - self.last_knowledge_share >= 10

    def should_request_information(self, current_time: int) -> bool:
        return current_time - self.last_information_request >= 15

class MultiAgentSystem:
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.task_history = defaultdict(list)
        self.current_time = 0

    async def allocate_tasks(self, tasks: List[Task]):
        for task in tasks:
            suitable_agents = [agent for agent in self.agents if agent.specialization == task.domain]
            if suitable_agents:
                chosen_agent = self._choose_agent_for_task(suitable_agents, task)
            else:
                chosen_agent = self._choose_agent_for_task(self.agents, task)
            await chosen_agent.task_queue.put(task)

    def _choose_agent_for_task(self, agents: List[Agent], task: Task) -> Agent:
        agent_scores = []
        for agent in agents:
            performance_score = self._calculate_agent_performance(agent)
            workload_score = 1 / (len(agent.task_queue._queue) + 1)
            specialization_score = 2 if agent.specialization == task.domain else 1
            total_score = performance_score * workload_score * specialization_score
            agent_scores.append((agent, total_score))
        return max(agent_scores, key=lambda x: x[1])[0]

    def _calculate_agent_performance(self, agent: Agent) -> float:
        if not agent.task_history:
            return agent.reputation
        recent_tasks = agent.task_history[-10:]
        return sum(result for _, result in recent_tasks) / len(recent_tasks)

    async def process_all_tasks(self):
        tasks = [agent.process_task(task) for agent in self.agents for task in agent.task_queue._queue]
        results = await asyncio.gather(*tasks)
        for agent, (task_id, success_rate) in zip(self.agents, results):
            self.task_history[agent.agent_id].append((task_id, success_rate))
            self._update_agent_reputation(agent)

    def _update_agent_reputation(self, agent: Agent):
        recent_performance = self._calculate_agent_performance(agent)
        task_diversity = len(set(task_id for task_id, _ in agent.task_history[-20:]))
        knowledge_contribution = len(agent.knowledge_base) / 10  # Assuming a max of 10 knowledge items is good
        agent.reputation = 0.5 * recent_performance + 0.3 * (task_diversity / 20) + 0.2 * knowledge_contribution

    def federated_learning_round(self):
        parameters_list = [agent.get_parameters() for agent in self.agents]
        aggregated_parameters = secure_aggregate(parameters_list)
        for agent in self.agents:
            agent.set_parameters(aggregated_parameters)

    async def collaborative_exchange(self):
        for agent in self.agents:
            if agent.should_share_knowledge(self.current_time):
                knowledge_to_share = agent.decide_knowledge_to_share()
                for other_agent in self.agents:
                    if other_agent != agent:
                        other_agent.update_knowledge(knowledge_to_share)
                agent.last_knowledge_share = self.current_time

            if agent.should_request_information(self.current_time):
                topic = random.choice(list(agent.knowledge_base.keys()))
                await agent.request_information(topic, self)

    async def handle_information_request(self, requesting_agent: Agent, topic: str) -> Any:
        for agent in self.agents:
            if agent != requesting_agent and topic in agent.knowledge_base:
                return agent.knowledge_base[topic]
        return None

    def add_agent(self, agent: Agent):
        self.agents.append(agent)

    def remove_agent(self, agent: Agent):
        if agent in self.agents:
            self.agents.remove(agent)

    def evaluate_system_performance(self):
        overall_performance = sum(agent.reputation for agent in self.agents) / len(self.agents)
        return overall_performance

    async def run_simulation(self, num_steps: int):
        for _ in range(num_steps):
            self.current_time += 1
            await self.process_all_tasks()
            self.federated_learning_round()
            await self.collaborative_exchange()

            if self.current_time % 50 == 0:
                performance = self.evaluate_system_performance()
                if performance < 0.5 and len(self.agents) > 3:
                    worst_agent = min(self.agents, key=lambda a: a.reputation)
                    self.remove_agent(worst_agent)
                elif performance > 0.8 and len(self.agents) < 10:
                    new_agent = Agent(f"Agent_{len(self.agents)+1}", random.choice(['classification', 'regression', 'clustering']), random.choice(['classification', 'regression', 'clustering']))
                    self.add_agent(new_agent)

        return self.evaluate_system_performance()