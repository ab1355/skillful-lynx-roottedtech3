import random
from typing import List, Callable

class ExternalEvent:
    def __init__(self, name: str, effect: Callable, probability: float):
        self.name = name
        self.effect = effect
        self.probability = probability

    def trigger(self, system):
        if random.random() < self.probability:
            self.effect(system)
            return True
        return False

def task_surge(system):
    system.config.tasks_per_step *= 2
    system.log.append(f"Time {system.current_time}: Task surge occurred")

def agent_failure(system):
    if len(system.agents) > 1:
        failed_agent = random.choice(system.agents)
        system.remove_agent(failed_agent)
        system.log.append(f"Time {system.current_time}: Agent {failed_agent.name} failed and was removed")

def knowledge_boost(system):
    for agent in system.agents:
        agent.knowledge *= 1.2
    system.log.append(f"Time {system.current_time}: Knowledge boost applied to all agents")

def domain_shift(system):
    new_domain_distribution = {
        'classification': random.random(),
        'regression': random.random(),
        'clustering': random.random()
    }
    total = sum(new_domain_distribution.values())
    for domain in new_domain_distribution:
        new_domain_distribution[domain] /= total
    system.domain_distribution = new_domain_distribution
    system.log.append(f"Time {system.current_time}: Domain distribution shifted to {new_domain_distribution}")

default_events = [
    ExternalEvent("Task Surge", task_surge, 0.05),
    ExternalEvent("Agent Failure", agent_failure, 0.02),
    ExternalEvent("Knowledge Boost", knowledge_boost, 0.03),
    ExternalEvent("Domain Shift", domain_shift, 0.01)
]

class ExternalEventSystem:
    def __init__(self, events: List[ExternalEvent] = default_events):
        self.events = events

    def process_events(self, system):
        for event in self.events:
            event.trigger(system)