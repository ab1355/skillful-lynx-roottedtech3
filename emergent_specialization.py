from collections import defaultdict

class EmergentAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_history = defaultdict(list)

    def update_task_history(self, task, performance):
        self.task_history[task.domain].append(performance)

    def get_emergent_specialization(self):
        if not self.task_history:
            return None
        return max(self.task_history, key=lambda k: sum(self.task_history[k]) / len(self.task_history[k]))

# Add this to the MultiAgentSystem class
def update_emergent_specializations(self):
    for agent in self.agents:
        if isinstance(agent, EmergentAgent):
            new_specialization = agent.get_emergent_specialization()
            if new_specialization and new_specialization != agent.specialization:
                print(f"Agent {agent.id} specialized from {agent.specialization} to {new_specialization}")
                agent.specialization = new_specialization