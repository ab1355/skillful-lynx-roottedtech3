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
        self.performance_by_task_type = defaultdict(lambda: {'success': 0, 'total': 0})
        self.creation_time = 0
        self.specialization_change_cooldown = 0
        self.total_tasks_processed = 0

    async def process_task(self, task: Task) -> Tuple[str, float]:
        processing_time = task.complexity * (1 / self.reputation)
        await asyncio.sleep(processing_time)
        success_probability = min(1.0, self.reputation / task.complexity)
        success = random.random() < success_probability
        result = 1.0 if success else 0.0
        self.task_history.append((task.task_id, result))
        self.performance_by_task_type[task.domain]['success'] += result
        self.performance_by_task_type[task.domain]['total'] += 1
        self.total_tasks_processed += 1
        
        if success:
            self.generate_knowledge(task)
        
        return task.task_id, result

    def generate_knowledge(self, task: Task):
        knowledge_key = f"{task.domain}_technique_{random.randint(1, 100)}"
        confidence = random.uniform(0.6, 1.0)
        self.knowledge_base[knowledge_key] = {
            'content': f"Learned technique for {task.domain} tasks",
            'confidence': confidence,
            'domain': task.domain
        }

    def update_knowledge(self, new_knowledge: Dict[str, Any]):
        for key, value in new_knowledge.items():
            if key not in self.knowledge_base or value.get('confidence', 0) > self.knowledge_base[key].get('confidence', 0):
                self.knowledge_base[key] = value

    def decide_knowledge_to_share(self) -> Dict[str, Any]:
        return {k: v for k, v in self.knowledge_base.items() if v.get('confidence', 0) > 0.6}

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
        return current_time - self.last_knowledge_share >= 5

    def should_request_information(self, current_time: int) -> bool:
        return current_time - self.last_information_request >= 7

    def get_best_specialization(self) -> str:
        if not self.performance_by_task_type:
            return self.specialization
        return max(self.performance_by_task_type, key=lambda x: self.performance_by_task_type[x]['success'] / max(1, self.performance_by_task_type[x]['total']))

    def get_performance_difference(self) -> float:
        if len(self.performance_by_task_type) < 2:
            return 0
        performances = [perf['success'] / max(1, perf['total']) for perf in self.performance_by_task_type.values()]
        return max(performances) - min(performances)

class MultiAgentSystem:
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.task_history = defaultdict(list)
        self.current_time = 0
        self.log = []
        self.performance_history = []
        self.workload_history = []
        self.specialization_changes = []
        for agent in self.agents:
            agent.creation_time = self.current_time

    async def allocate_tasks(self, tasks: List[Task]):
        agent_workloads = {agent: len(agent.task_queue._queue) for agent in self.agents}
        for task in tasks:
            suitable_agents = [agent for agent in self.agents if agent.specialization == task.domain]
            if not suitable_agents:
                suitable_agents = self.agents
            chosen_agent = self._choose_agent_for_task(suitable_agents, task, agent_workloads)
            await chosen_agent.task_queue.put(task)
            agent_workloads[chosen_agent] += 1
        self.log.append(f"Time {self.current_time}: Allocated {len(tasks)} tasks")

    def _choose_agent_for_task(self, agents: List[Agent], task: Task, agent_workloads: Dict[Agent, int]) -> Agent:
        agent_scores = []
        avg_workload = sum(agent_workloads.values()) / len(agent_workloads)
        for agent in agents:
            performance_score = self._calculate_agent_performance(agent)
            workload_score = 1 / (1 + abs(agent_workloads[agent] - avg_workload))
            specialization_score = 2 if agent.specialization == task.domain else 1
            warm_up_score = min(1, (self.current_time - agent.creation_time) / 3)
            integration_score = 2 if agent.total_tasks_processed < 50 else 1  # Boost for newer agents
            total_score = performance_score * workload_score * specialization_score * warm_up_score * integration_score
            agent_scores.append((agent, total_score))
        chosen_agent = max(agent_scores, key=lambda x: x[1])[0]
        self.log.append(f"Time {self.current_time}: Chose {chosen_agent.agent_id} for task {task.task_id}")
        return chosen_agent

    def _calculate_agent_performance(self, agent: Agent) -> float:
        if not agent.task_history:
            return agent.reputation
        recent_tasks = agent.task_history[-20:]
        return sum(result for _, result in recent_tasks) / len(recent_tasks)

    async def process_all_tasks(self):
        tasks = [agent.process_task(task) for agent in self.agents for task in list(agent.task_queue._queue)]
        results = await asyncio.gather(*tasks)
        for agent in self.agents:
            agent.task_queue._queue.clear()
        for agent, (task_id, success_rate) in zip(self.agents, results):
            self.task_history[agent.agent_id].append((task_id, success_rate))
            self._update_agent_reputation(agent)
        self.log.append(f"Time {self.current_time}: Processed {len(results)} tasks")

    def _update_agent_reputation(self, agent: Agent):
        recent_performance = self._calculate_agent_performance(agent)
        task_diversity = len(set(task_id for task_id, _ in agent.task_history[-20:]))
        knowledge_contribution = len(agent.knowledge_base) / 20
        agent.reputation = 0.5 * recent_performance + 0.3 * (task_diversity / 20) + 0.2 * knowledge_contribution
        self.log.append(f"Time {self.current_time}: Updated {agent.agent_id} reputation to {agent.reputation:.2f}")

    def federated_learning_round(self):
        parameters_list = [agent.get_parameters() for agent in self.agents]
        aggregated_parameters = secure_aggregate(parameters_list, epsilon=0.5)
        for agent in self.agents:
            agent.set_parameters(aggregated_parameters)
        self.log.append(f"Time {self.current_time}: Completed federated learning round")

    async def collaborative_exchange(self):
        for agent in self.agents:
            if agent.should_share_knowledge(self.current_time):
                knowledge_to_share = agent.decide_knowledge_to_share()
                if knowledge_to_share:
                    self._distribute_knowledge(agent, knowledge_to_share)
                    agent.last_knowledge_share = self.current_time
                    self.log.append(f"Time {self.current_time}: {agent.agent_id} shared knowledge: {list(knowledge_to_share.keys())}")
                else:
                    self.log.append(f"Time {self.current_time}: {agent.agent_id} had no knowledge to share")

            if agent.should_request_information(self.current_time):
                topic = self._choose_information_request_topic(agent)
                info = await agent.request_information(topic, self)
                if info:
                    agent.update_knowledge({topic: info})
                    self.log.append(f"Time {self.current_time}: {agent.agent_id} received information on {topic}")
                else:
                    self.log.append(f"Time {self.current_time}: {agent.agent_id} requested information on {topic}, but none was available")

    def _distribute_knowledge(self, sharing_agent: Agent, knowledge: Dict[str, Any]):
        for receiving_agent in self.agents:
            if receiving_agent != sharing_agent:
                knowledge_to_share = {}
                for key, value in knowledge.items():
                    if key not in receiving_agent.knowledge_base or value['confidence'] > receiving_agent.knowledge_base[key]['confidence']:
                        if len(receiving_agent.knowledge_base) < 30 or random.random() < 0.5:
                            knowledge_to_share[key] = value
                receiving_agent.update_knowledge(knowledge_to_share)

    def _choose_information_request_topic(self, agent: Agent) -> str:
        if not agent.knowledge_base:
            return "general"
        agent_domains = set(item['domain'] for item in agent.knowledge_base.values())
        all_domains = set(agent.specialization for agent in self.agents)
        missing_domains = all_domains - agent_domains
        if missing_domains:
            return random.choice(list(missing_domains))
        return random.choice(list(agent.knowledge_base.keys()))

    async def handle_information_request(self, requesting_agent: Agent, topic: str) -> Any:
        potential_providers = [agent for agent in self.agents if agent != requesting_agent and topic in agent.knowledge_base]
        if potential_providers:
            provider = max(potential_providers, key=lambda a: a.knowledge_base[topic]['confidence'])
            return provider.knowledge_base[topic]
        return None

    def add_agent(self, agent: Agent):
        agent.creation_time = self.current_time
        self.agents.append(agent)
        self.log.append(f"Time {self.current_time}: Added new agent {agent.agent_id}")

    def remove_agent(self, agent: Agent):
        if agent in self.agents:
            self.agents.remove(agent)
            self.log.append(f"Time {self.current_time}: Removed agent {agent.agent_id}")

    def evaluate_system_performance(self):
        if not self.agents:
            return 0
        agent_performances = [self._calculate_agent_performance(agent) for agent in self.agents]
        overall_performance = sum(agent_performances) / len(agent_performances)
        task_coverage = len(set(agent.specialization for agent in self.agents)) / 3
        knowledge_diversity = self._calculate_knowledge_diversity()
        workload_balance = self._calculate_workload_balance()
        system_performance = 0.4 * overall_performance + 0.2 * task_coverage + 0.2 * knowledge_diversity + 0.2 * workload_balance
        self.performance_history.append(system_performance)
        self.workload_history.append(self._get_workload_distribution())
        self.log.append(f"Time {self.current_time}: System performance: {system_performance:.2f}")
        return system_performance

    def _calculate_knowledge_diversity(self):
        all_knowledge = set()
        for agent in self.agents:
            all_knowledge.update(agent.knowledge_base.keys())
        return len(all_knowledge) / (30 * len(self.agents))

    def _calculate_workload_balance(self):
        if not self.agents:
            return 0
        workloads = [agent.total_tasks_processed for agent in self.agents]
        if not workloads or sum(workloads) == 0:
            return 1
        avg_workload = sum(workloads) / len(workloads)
        max_deviation = max(abs(w - avg_workload) for w in workloads)
        return 1 - (max_deviation / avg_workload)

    def _get_workload_distribution(self):
        total_tasks = sum(agent.total_tasks_processed for agent in self.agents)
        return {agent.agent_id: agent.total_tasks_processed / total_tasks for agent in self.agents}

    def _adjust_agent_specializations(self):
        for agent in self.agents:
            if agent.specialization_change_cooldown > 0:
                agent.specialization_change_cooldown -= 1
                continue

            best_specialization = agent.get_best_specialization()
            if best_specialization != agent.specialization:
                performance_diff = agent.get_performance_difference()
                change_probability = min(1.0, performance_diff * 3)  # Increased change probability
                if random.random() < change_probability:
                    self.log.append(f"Time {self.current_time}: {agent.agent_id} changed specialization from {agent.specialization} to {best_specialization}")
                    self.specialization_changes.append((self.current_time, agent.agent_id, agent.specialization, best_specialization))
                    agent.specialization = best_specialization
                    agent.specialization_change_cooldown = 5  # Reduced cooldown period

    async def run_simulation(self, num_steps: int):
        for _ in range(num_steps):
            self.current_time += 1
            
            new_tasks = [Task(f"Task_{self.current_time}_{i}", random.uniform(0.3, 0.9), random.choice(["classification", "regression", "clustering"])) for i in range(5)]
            await self.allocate_tasks(new_tasks)
            
            await self.process_all_tasks()
            self.federated_learning_round()
            await self.collaborative_exchange()
            self._adjust_agent_specializations()

            if self.current_time % 10 == 0:
                performance = self.evaluate_system_performance()
                if performance < 0.4 and len(self.agents) > 3:
                    worst_agent = min(self.agents, key=lambda a: a.reputation)
                    self.remove_agent(worst_agent)
                elif performance > 0.7 and len(self.agents) < 10:
                    new_agent = Agent(f"Agent_{len(self.agents)+1}", random.choice(['classification', 'regression', 'clustering']), random.choice(['classification', 'regression', 'clustering']))
                    self.add_agent(new_agent)

        return self.evaluate_system_performance()

    def get_log(self):
        return self.log

    def get_performance_history(self):
        return self.performance_history

    def get_workload_history(self):
        return self.workload_history

    def get_specialization_changes(self):
        return self.specialization_changes