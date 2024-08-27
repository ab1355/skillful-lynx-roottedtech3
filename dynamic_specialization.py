import numpy as np

def update_agent_specialization(agent, performance_threshold=0.8):
    if np.mean(agent.performance_history[-10:]) > performance_threshold:
        new_specialization = np.random.choice(list(agent.skills.keys()))
        agent.specialization = new_specialization
        print(f"Agent {agent.id} updated specialization to {new_specialization}")

# Add this to the MultiAgentSystem class
def dynamic_specialization_update(self):
    for agent in self.agents:
        update_agent_specialization(agent)