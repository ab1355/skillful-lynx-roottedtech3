class CognitiveLoadAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cognitive_load = 0
        self.max_load = 100

    def can_accept_task(self, task):
        return self.cognitive_load + task.complexity <= self.max_load

    def assign_task(self, task):
        if self.can_accept_task(task):
            self.cognitive_load += task.complexity
            return True
        return False

    def complete_task(self, task):
        self.cognitive_load -= task.complexity

# Add this to the MultiAgentSystem class
def assign_task_with_load_balancing(self, task):
    suitable_agents = [agent for agent in self.agents if isinstance(agent, CognitiveLoadAgent) and agent.can_accept_task(task)]
    if suitable_agents:
        chosen_agent = min(suitable_agents, key=lambda a: a.cognitive_load)
        if chosen_agent.assign_task(task):
            print(f"Assigned task to Agent {chosen_agent.id}. New cognitive load: {chosen_agent.cognitive_load}")
            return True
    print("No suitable agent found for task")
    return False