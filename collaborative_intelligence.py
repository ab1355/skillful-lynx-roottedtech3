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
    # ... (previous methods remain the same)

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
        for domain in global_domain_performance:
            agent_perf = agent_performance.get(domain, 0)
            global_perf = global_domain_performance[domain]
            combined_performance[domain] = 0.6 * agent_perf + 0.3 * global_perf + 0.1 * (1 - agent.utilization_score)
        if not combined_performance:
            return agent.specialization  # Keep current specialization if no performance data
        return max(combined_performance, key=combined_performance.get)

    def _remove_mentorship(self, agent: Agent):
        for mentor in agent.mentors:
            mentor.mentees.remove(agent)
        agent.mentors.clear()
        for mentee in agent.mentees:
            mentee.mentors.remove(agent)
        agent.mentees.clear()

    def _assign_mentors(self, agent: Agent):
        potential_mentors = [a for a in self.agents if a != agent and a.specialization == agent.specialization]
        if potential_mentors:
            mentor = max(potential_mentors, key=lambda m: m.get_domain_expertise(agent.specialization))
            agent.mentors.append(mentor)
            mentor.mentees.append(agent)

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

    # ... (rest of the methods remain the same)

    async def run_simulation(self, num_steps: int):
        for step in range(num_steps):
            self.current_time += 1
            
            if step % 10 == 0:
                print(f"Simulation step: {step}/{num_steps}")

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

    # ... (rest of the methods remain the same)