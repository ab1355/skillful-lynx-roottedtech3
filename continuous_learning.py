import numpy as np

def update_agent_model(agent, new_data, learning_rate=0.1):
    # Simulating model update with simple weighted average
    agent.knowledge = (1 - learning_rate) * agent.knowledge + learning_rate * new_data

# Add this to the MultiAgentSystem class
def continuous_learning_update(self):
    for agent in self.agents:
        if agent.performance_history:
            new_data = np.random.rand(len(agent.knowledge))  # Simulated new data
            update_agent_model(agent, new_data)
            print(f"Updated model for Agent {agent.id}")