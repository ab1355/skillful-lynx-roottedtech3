import numpy as np
from typing import List, Dict, Any, Tuple
import random
import asyncio
from collections import defaultdict
from agent import Agent
from models import SpecializedModel, create_model
from task import Task
from privacy import secure_aggregate

class MultiAgentSystem:
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.task_history = defaultdict(list)
        self.current_time = 0
        self.log = []
        self.performance_history = []
        self.workload_history = []
        self.specialization_changes = []
        self.long_term_performance = []
        self.add_agent_threshold = 0.7
        self.remove_agent_threshold = 0.4
        self.domain_performance = defaultdict(list)
        self.mentoring_reports = []
        self.task_complexity_adjustment = 1.0
        self.task_complexity_adjustment_rate = 0.01
        self.domain_specific_complexity = {domain: 1.0 for domain in ["classification", "regression", "clustering"]}
        for agent in self.agents:
            agent.creation_time = self.current_time

    async def allocate_tasks(self, tasks: List[Task]):
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
        self.log.append(f"Time {self.current_time}: Allocated {len(tasks)} tasks")

    def _choose_agent_for_task(self, agents: List[Agent], task: Task, agent_workloads: Dict[Agent, int], domain_workloads: Dict[str, int]) -> Agent:
        agent_scores = []
        avg_workload = max(1, sum(agent_workloads.values()) / len(agent_workloads))
        total_domain_workload = sum(domain_workloads.values())
        for agent in agents:
            performance_score = self._calculate_agent_performance(agent)
            workload_score = 1 / (1 + abs(agent_workloads[agent] - avg_workload))
            specialization_score = 2 if agent.specialization == task.domain else 1
            warm_up_score = min(1, (self.current_time - agent.creation_time) / 3)
            integration_score = 5 if agent.total_tasks_processed < 50 else 1
            catch_up_score = 1 + max(0, (avg_workload - agent.total_tasks_processed) / avg_workload)
            domain_balance_score = domain_workloads[task.domain] / total_domain_workload if total_domain_workload > 0 else 1
            expertise_score = agent.get_domain_expertise(task.domain)
            ramp_up_score = agent.ramp_up_boost
            utilization_score = 3 - agent.utilization_score  # Even more aggressive favoring of underutilized agents
            long_term_trend = sum(agent.long_term_performance[-10:]) / 10 if agent.long_term_performance else 1
            total_score = performance_score * workload_score * specialization_score * warm_up_score * integration_score * catch_up_score * domain_balance_score * expertise_score * ramp_up_score * utilization_score * long_term_trend
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
        tasks = [agent.process_task(self._adjust_task_complexity(task)) for agent in self.agents for task in list(agent.task_queue._queue)]
        results = await asyncio.gather(*tasks)
        for agent in self.agents:
            agent.task_queue._queue.clear()
        for agent, (task_id, success_rate) in zip(self.agents, results):
            self.task_history[agent.agent_id].append((task_id, success_rate))
            self._update_agent_reputation(agent)
        self.log.append(f"Time {self.current_time}: Processed {len(results)} tasks")

        # Update utilization scores
        total_tasks = sum(len(agent.task_history) for agent in self.agents)
        for agent in self.agents:
            agent.update_utilization_score(total_tasks)

    def _adjust_task_complexity(self, task: Task) -> Task:
        task.complexity *= self.task_complexity_adjustment * self.domain_specific_complexity[task.domain]
        return task

    def _update_agent_reputation(self, agent: Agent):
        recent_performance = self._calculate_agent_performance(agent)
        task_diversity = len(set(task_id for task_id, _ in agent.task_history[-20:]))
        knowledge_contribution = len(agent.knowledge_base) / 20
        agent.reputation = 0.5 * recent_performance + 0.3 * (task_diversity / 20) + 0.2 * knowledge_contribution
        agent.decay_ramp_up_boost()
        agent.update_long_term_performance(recent_performance)
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
                info = await self.handle_information_request(agent, topic)
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

    def _adjust_agent_specializations(self):
        global_domain_performance = self._calculate_global_domain_performance()
        for agent in self.agents:
            if agent.specialization_change_cooldown > 0:
                agent.specialization_change_cooldown -= 1
                continue

            best_specialization = self._get_best_specialization_for_agent(agent, global_domain_performance)
            if best_specialization != agent.specialization:
                performance_diff = agent.get_performance_difference()
                utilization_factor = 1 - agent.utilization_score
                long_term_trend = (agent.long_term_performance[-1] - agent.long_term_performance[0]) / len(agent.long_term_performance) if agent.long_term_performance else 0
                change_probability = min(1.0, performance_diff * 40 + utilization_factor * 0.3 + long_term_trend * 0.2)
                if random.random() < change_probability:
                    self._remove_mentorship(agent)
                    self.log.append(f"Time {self.current_time}: {agent.agent_id} changed specialization from {agent.specialization} to {best_specialization}")
                    self.specialization_changes.append((self.current_time, agent.agent_id, agent.specialization, best_specialization))
                    agent.specialization = best_specialization
                    agent.specialization_strength = 1.0  # Reset specialization strength
                    agent.specialization_change_cooldown = 1  # Reduced cooldown period
                    self._assign_mentors(agent)

    def _improve_mentoring(self):
        for agent in self.agents:
            for mentee in agent.mentees:
                for domain in ["classification", "regression", "clustering"]:
                    expertise_diff = max(0, agent.get_domain_expertise(domain) - mentee.get_domain_expertise(domain))
                    mentoring_impact = expertise_diff * 0.1  # 10% of the difference
                    if domain == "clustering":
                        mentoring_impact *= 1.5  # 50% boost for clustering domain
                    mentee.increase_domain_expertise(domain, mentoring_impact)
                    agent.mentoring_impact[domain].append(mentoring_impact)

    def _should_remove_agent(self) -> Tuple[bool, Agent]:
        if len(self.agents) <= 3:
            return False, None
        underperforming_agents = [agent for agent in self.agents if sum(agent.long_term_performance[-10:]) / 10 < self.remove_agent_threshold]
        if not underperforming_agents:
            return False, None
        domain_counts = defaultdict(int)
        for agent in self.agents:
            domain_counts[agent.specialization] += 1
        removable_agents = [agent for agent in underperforming_agents if domain_counts[agent.specialization] > 1]
        if not removable_agents:
            return False, None
        return True, min(removable_agents, key=lambda a: sum(a.long_term_performance[-10:]) / 10)

    def _should_add_agent(self) -> bool:
        if len(self.agents) >= 10:
            return False
        recent_performance = self.performance_history[-10:]
        performance_trend = sum(recent_performance) / len(recent_performance)
        domain_needs = any(sum(perf[-5:]) / 5 < 0.6 for perf in self.domain_performance.values())
        return performance_trend > self.add_agent_threshold or domain_needs

    def _adjust_task_complexity_rates(self):
        for domain in ["classification", "regression", "clustering"]:
            domain_performance = sum(self.domain_performance[domain][-10:]) / 10
            if domain_performance > 0.9:
                self.domain_specific_complexity[domain] *= (1 + self.task_complexity_adjustment_rate)
            elif domain_performance < 0.7:
                self.domain_specific_complexity[domain] *= (1 - self.task_complexity_adjustment_rate)
            self.domain_specific_complexity[domain] = max(0.5, min(2.0, self.domain_specific_complexity[domain]))

    def _calculate_global_domain_performance(self):
        global_performance = defaultdict(lambda: {'success': 0, 'total': 0})
        for agent in self.agents:
            for domain, performance in agent.performance_by_task_type.items():
                global_performance[domain]['success'] += performance['success']
                global_performance[domain]['total'] += performance['total']
        return {domain: perf['success'] / max(1, perf['total']) for domain, perf in global_performance.items()}

    def _get_best_specialization_for_agent(self, agent: Agent, global_domain_performance: Dict[str, float]):
        agent_performance = {domain: perf['success'] / max(1, perf['total']) for domain, perf in agent.performance_by_task_type.items()}
        combined_performance = {}
        for domain in agent_performance:
            combined_performance[domain] = 0.6 * agent_performance[domain] + 0.3 * global_domain_performance[domain] + 0.1 * (1 - agent.utilization_score)
        return max(combined_performance, key=combined_performance.get)

    async def run_simulation(self, num_steps: int):
        for _ in range(num_steps):
            self.current_time += 1
            
            new_tasks = [Task(f"Task_{self.current_time}_{i}", random.uniform(0.3, 1.5), random.choice(["classification", "regression", "clustering"])) for i in range(5)]
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

        return self.evaluate_system_performance()

    # ... (Add other necessary methods)

    def get_log(self):
        return self.log

    def get_performance_history(self):
        return self.performance_history

    def get_workload_history(self):
        return self.workload_history

    def get_specialization_changes(self):
        return self.specialization_changes

    def get_long_term_performance(self):
        return self.long_term_performance

    def get_domain_performance(self):
        return self.domain_performance

    def get_mentoring_reports(self):
        return self.mentoring_reports