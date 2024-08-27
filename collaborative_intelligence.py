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
        self.specialization_strength = 1.0
        self.knowledge_specialization = defaultdict(float)
        self.expertise_level = {
            "classification": 1,
            "regression": 1,
            "clustering": 1
        }
        self.mentor = None
        self.mentee = None
        self.mentoring_impact = defaultdict(list)  # Track impact of mentoring
        self.ramp_up_boost = 1.0  # Temporary boost for new agents
        self.utilization_score = 1.0  # New attribute to track agent utilization

    async def process_task(self, task: Task) -> Tuple[str, float]:
        processing_time = task.complexity * (1 / (self.reputation * self.ramp_up_boost)) * (1 / self.expertise_level[task.domain])
        await asyncio.sleep(processing_time)
        success_probability = min(1.0, (self.reputation * self.ramp_up_boost * self.specialization_strength * self.expertise_level[task.domain]) / task.complexity)
        success = random.random() < success_probability
        result = 1.0 if success else 0.0
        self.task_history.append((task.task_id, result))
        self.performance_by_task_type[task.domain]['success'] += result
        self.performance_by_task_type[task.domain]['total'] += 1
        self.total_tasks_processed += 1
        
        if success:
            self.generate_knowledge(task)
            self.update_specialization_strength(task.domain, 0.1)
            self.increase_expertise(task.domain)
        else:
            self.update_specialization_strength(task.domain, -0.05)
        
        return task.task_id, result

    def generate_knowledge(self, task: Task):
        knowledge_key = f"{task.domain}_technique_{random.randint(1, 100)}"
        confidence = random.uniform(0.6, 1.0) * self.expertise_level[task.domain]
        self.knowledge_base[knowledge_key] = {
            'content': f"Learned technique for {task.domain} tasks",
            'confidence': confidence,
            'domain': task.domain
        }
        self.knowledge_specialization[task.domain] += 0.1

    def update_knowledge(self, new_knowledge: Dict[str, Any]):
        for key, value in new_knowledge.items():
            if key not in self.knowledge_base or value.get('confidence', 0) > self.knowledge_base[key].get('confidence', 0):
                self.knowledge_base[key] = value
                self.knowledge_specialization[value['domain']] += 0.05

    def decide_knowledge_to_share(self) -> Dict[str, Any]:
        specialized_knowledge = {k: v for k, v in self.knowledge_base.items() 
                                 if v.get('confidence', 0) > 0.6 and v['domain'] == self.specialization}
        return specialized_knowledge

    def update_specialization_strength(self, domain: str, change: float):
        if domain == self.specialization:
            self.specialization_strength = max(0.5, min(2.0, self.specialization_strength + change))

    def get_best_specialization(self) -> str:
        if not self.performance_by_task_type:
            return self.specialization
        return max(self.performance_by_task_type, key=lambda x: self.performance_by_task_type[x]['success'] / max(1, self.performance_by_task_type[x]['total']))

    def get_performance_difference(self) -> float:
        if len(self.performance_by_task_type) < 2:
            return 0
        performances = [perf['success'] / max(1, perf['total']) for perf in self.performance_by_task_type.values()]
        return max(performances) - min(performances)

    def get_parameters(self) -> Dict[str, np.ndarray]:
        return self.model.get_parameters()

    def set_parameters(self, parameters: Dict[str, np.ndarray]):
        self.model.set_parameters(parameters)

    def should_share_knowledge(self, current_time: int) -> bool:
        return current_time - self.last_knowledge_share >= 5

    def should_request_information(self, current_time: int) -> bool:
        return current_time - self.last_information_request >= 7

    def increase_expertise(self, domain: str):
        old_expertise = self.expertise_level[domain]
        self.expertise_level[domain] = min(10, self.expertise_level[domain] + 0.1)
        if self.mentee:
            mentee_old_expertise = self.mentee.expertise_level[domain]
            mentee_expertise_gain = max(0, min(self.mentee.expertise_level[domain] + 0.05, self.expertise_level[domain]) - mentee_old_expertise)
            self.mentee.expertise_level[domain] += mentee_expertise_gain
            self.mentoring_impact[domain].append(mentee_expertise_gain)
        return self.expertise_level[domain] - old_expertise

    def decay_ramp_up_boost(self):
        self.ramp_up_boost = max(1.0, self.ramp_up_boost * 0.95)

    def update_utilization_score(self, total_tasks: int):
        expected_tasks = total_tasks / len(self.performance_by_task_type)
        actual_tasks = self.total_tasks_processed
        self.utilization_score = min(2.0, max(0.5, actual_tasks / expected_tasks))

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
        for agent in self.agents:
            agent.creation_time = self.current_time

    async def allocate_tasks(self, tasks: List[Task]):
        agent_workloads = {agent: len(agent.task_queue._queue) for agent in self.agents}
        domain_workloads = defaultdict(int)
        for task in tasks:
            domain_workloads[task.domain] += 1

        for task in tasks:
            suitable_agents = [agent for agent in self.agents if agent.specialization == task.domain]
            if not suitable_agents:
                suitable_agents = self.agents
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
            integration_score = 5 if agent.total_tasks_processed < 50 else 1  # Increased boost for newer agents
            catch_up_score = 1 + max(0, (avg_workload - agent.total_tasks_processed) / avg_workload)
            domain_balance_score = domain_workloads[task.domain] / total_domain_workload if total_domain_workload > 0 else 1
            expertise_score = agent.expertise_level[task.domain]
            ramp_up_score = agent.ramp_up_boost
            utilization_score = 1 / agent.utilization_score  # Favor underutilized agents
            total_score = performance_score * workload_score * specialization_score * warm_up_score * integration_score * catch_up_score * domain_balance_score * expertise_score * ramp_up_score * utilization_score
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

        # Update utilization scores
        total_tasks = sum(len(agent.task_history) for agent in self.agents)
        for agent in self.agents:
            agent.update_utilization_score(total_tasks)

    def _update_agent_reputation(self, agent: Agent):
        recent_performance = self._calculate_agent_performance(agent)
        task_diversity = len(set(task_id for task_id, _ in agent.task_history[-20:]))
        knowledge_contribution = len(agent.knowledge_base) / 20
        agent.reputation = 0.5 * recent_performance + 0.3 * (task_diversity / 20) + 0.2 * knowledge_contribution
        agent.decay_ramp_up_boost()
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

    def add_agent(self, agent: Agent):
        agent.creation_time = self.current_time
        agent.ramp_up_boost = 2.0  # Start with a 2x boost
        self.agents.append(agent)
        self._assign_mentor(agent)
        self.log.append(f"Time {self.current_time}: Added new agent {agent.agent_id}")

    def remove_agent(self, agent: Agent):
        if agent in self.agents:
            self._remove_mentorship(agent)
            self.agents.remove(agent)
            self.log.append(f"Time {self.current_time}: Removed agent {agent.agent_id}")

    def _assign_mentor(self, new_agent: Agent):
        potential_mentors = [agent for agent in self.agents if agent.specialization == new_agent.specialization and agent != new_agent]
        if potential_mentors:
            mentor = max(potential_mentors, key=lambda a: a.reputation)
            new_agent.mentor = mentor
            mentor.mentee = new_agent
            self.log.append(f"Time {self.current_time}: Assigned {mentor.agent_id} as mentor to {new_agent.agent_id}")

    def _remove_mentorship(self, agent: Agent):
        if agent.mentor:
            agent.mentor.mentee = None
            agent.mentor = None
        if agent.mentee:
            agent.mentee.mentor = None
            self._assign_mentor(agent.mentee)
            agent.mentee = None

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
        self._update_domain_performance()
        self._generate_mentoring_report()
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
        return {agent.agent_id: agent.total_tasks_processed / max(1, total_tasks) for agent in self.agents}

    def _update_domain_performance(self):
        for domain in ["classification", "regression", "clustering"]:
            domain_agents = [agent for agent in self.agents if agent.specialization == domain]
            if domain_agents:
                avg_performance = sum(agent.performance_by_task_type[domain]['success'] / max(1, agent.performance_by_task_type[domain]['total']) for agent in domain_agents) / len(domain_agents)
                self.domain_performance[domain].append(avg_performance)

    def _generate_mentoring_report(self):
        report = {}
        for agent in self.agents:
            if agent.mentee:
                mentee_improvements = {domain: sum(impacts) / len(impacts) if impacts else 0 for domain, impacts in agent.mentoring_impact.items()}
                report[agent.agent_id] = {
                    "mentee": agent.mentee.agent_id,
                    "improvements": mentee_improvements
                }
        self.mentoring_reports.append(report)

    def _adjust_agent_specializations(self):
        global_domain_performance = self._calculate_global_domain_performance()
        for agent in self.agents:
            if agent.specialization_change_cooldown > 0:
                agent.specialization_change_cooldown -= 1
                continue

            best_specialization = self._get_best_specialization_for_agent(agent, global_domain_performance)
            if best_specialization != agent.specialization:
                performance_diff = agent.get_performance_difference()
                change_probability = min(1.0, performance_diff * 30)  # Increased sensitivity even further
                if random.random() < change_probability:
                    self._remove_mentorship(agent)
                    self.log.append(f"Time {self.current_time}: {agent.agent_id} changed specialization from {agent.specialization} to {best_specialization}")
                    self.specialization_changes.append((self.current_time, agent.agent_id, agent.specialization, best_specialization))
                    agent.specialization = best_specialization
                    agent.specialization_strength = 1.0  # Reset specialization strength
                    agent.specialization_change_cooldown = 2  # Kept the reduced cooldown period
                    self._assign_mentor(agent)

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
            combined_performance[domain] = 0.7 * agent_performance[domain] + 0.3 * global_domain_performance[domain]
        return max(combined_performance, key=combined_performance.get)

    def _update_performance_thresholds(self):
        if len(self.performance_history) > 10:
            recent_performance = self.performance_history[-10:]
            trend = sum(recent_performance) / len(recent_performance)
            self.add_agent_threshold = max(0.6, min(0.8, trend + 0.1))
            self.remove_agent_threshold = max(0.3, min(0.5, trend - 0.1))

    def _long_term_performance_analysis(self):
        if len(self.performance_history) > 50:
            long_term_avg = sum(self.performance_history[-50:]) / 50
            self.long_term_performance.append(long_term_avg)
            if len(self.long_term_performance) > 5:
                trend = (self.long_term_performance[-1] - self.long_term_performance[-5]) / 5
                if trend < -0.01:
                    self.log.append(f"Time {self.current_time}: Long-term performance declining. Adjusting system parameters.")
                    self._adjust_system_parameters()
                elif trend > 0.01:
                    self.log.append(f"Time {self.current_time}: Long-term performance improving. Optimizing system parameters.")
                    self._optimize_system_parameters()
            
            self._identify_and_address_bottlenecks()

    def _adjust_system_parameters(self):
        for agent in self.agents:
            agent.reputation *= 0.95  # Slightly reduce all agent reputations to encourage improvement
        self.add_agent_threshold *= 0.98  # Make it slightly easier to add new agents
        self.remove_agent_threshold *= 1.02  # Make it slightly harder to remove agents

    def _optimize_system_parameters(self):
        for agent in self.agents:
            agent.reputation *= 1.05  # Slightly increase all agent reputations to reward good performance
        self.add_agent_threshold *= 1.02  # Make it slightly harder to add new agents
        self.remove_agent_threshold *= 0.98  # Make it slightly easier to remove agents

    def _identify_and_address_bottlenecks(self):
        if len(self.domain_performance["classification"]) < 5:
            return

        for domain in ["classification", "regression", "clustering"]:
            recent_performance = self.domain_performance[domain][-5:]
            if sum(recent_performance) / len(recent_performance) < 0.6:  # If average performance in domain is below 0.6
                self.log.append(f"Time {self.current_time}: Identified bottleneck in {domain} domain. Taking action.")
                self._address_domain_bottleneck(domain)

    def _address_domain_bottleneck(self, domain: str):
        domain_agents = [agent for agent in self.agents if agent.specialization == domain]
        if len(domain_agents) < 3:
            # Add a new agent specialized in this domain
            new_agent = Agent(f"Agent_{len(self.agents)+1}", domain, domain)
            self.add_agent(new_agent)
        else:
            # Increase training and knowledge sharing for existing agents
            for agent in domain_agents:
                agent.expertise_level[domain] = min(10, agent.expertise_level[domain] * 1.1)
                if agent.mentor:
                    agent.expertise_level[domain] = min(10, (agent.expertise_level[domain] + agent.mentor.expertise_level[domain]) / 2)

    def _system_wide_optimization(self):
        domain_performances = defaultdict(list)
        for agent in self.agents:
            for domain, perf in agent.performance_by_task_type.items():
                if perf['total'] > 0:
                    domain_performances[domain].append((agent, perf['success'] / perf['total']))
        
        for domain, performances in domain_performances.items():
            best_agents = sorted(performances, key=lambda x: x[1], reverse=True)[:3]
            for agent, _ in best_agents:
                if agent.specialization != domain:
                    self._remove_mentorship(agent)
                    self.log.append(f"Time {self.current_time}: Reassigned {agent.agent_id} from {agent.specialization} to {domain}")
                    self.specialization_changes.append((self.current_time, agent.agent_id, agent.specialization, domain))
                    agent.specialization = domain
                    agent.specialization_strength = 1.5  # Give a boost to the new specialization
                    self._assign_mentor(agent)

    async def run_simulation(self, num_steps: int):
        for _ in range(num_steps):
            self.current_time += 1
            
            new_tasks = [Task(f"Task_{self.current_time}_{i}", random.uniform(0.3, 1.5), random.choice(["classification", "regression", "clustering"])) for i in range(5)]
            await self.allocate_tasks(new_tasks)
            
            await self.process_all_tasks()
            self.federated_learning_round()
            await self.collaborative_exchange()
            self._adjust_agent_specializations()

            if self.current_time % 10 == 0:
                performance = self.evaluate_system_performance()
                self._update_performance_thresholds()
                self._long_term_performance_analysis()
                if performance < self.remove_agent_threshold and len(self.agents) > 3:
                    worst_agent = min(self.agents, key=lambda a: a.reputation)
                    self.remove_agent(worst_agent)
                elif performance > self.add_agent_threshold and len(self.agents) < 10:
                    new_agent = Agent(f"Agent_{len(self.agents)+1}", random.choice(['classification', 'regression', 'clustering']), random.choice(['classification', 'regression', 'clustering']))
                    self.add_agent(new_agent)

            if self.current_time % 50 == 0:
                self._system_wide_optimization()

        return self.evaluate_system_performance()

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