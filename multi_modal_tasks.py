class MultiModalTask(Task):
    def __init__(self, name, modalities, complexity):
        super().__init__(name, "multi_modal", "combined", "analysis", complexity)
        self.modalities = modalities

class MultiModalAgent(Agent):
    def __init__(self, id, specializations):
        super().__init__(id, "multi_modal", "combined")
        self.specializations = specializations

# Add this to the MultiAgentSystem class
def handle_multi_modal_task(self, task):
    suitable_agents = [agent for agent in self.agents if isinstance(agent, MultiModalAgent) and 
                       all(modality in agent.specializations for modality in task.modalities)]
    if suitable_agents:
        chosen_agent = max(suitable_agents, key=lambda a: np.mean(a.performance_history))
        self.assign_task(task, chosen_agent)
    else:
        print(f"No suitable agent found for multi-modal task {task.name}")