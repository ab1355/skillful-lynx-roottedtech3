import numpy as np
from scipy import stats

def detect_anomalies(data, threshold=3):
    z_scores = np.abs(stats.zscore(data))
    return z_scores > threshold

# Add this to the MultiAgentSystem class
def system_anomaly_detection(self):
    agent_performances = [np.mean(agent.performance_history[-10:]) for agent in self.agents]
    anomalies = detect_anomalies(agent_performances)
    
    for i, is_anomaly in enumerate(anomalies):
        if is_anomaly:
            print(f"Anomaly detected in Agent {self.agents[i].id}")
            self.heal_agent(self.agents[i])

def heal_agent(self, agent):
    print(f"Initiating self-healing for Agent {agent.id}")
    # Reset agent's knowledge to the average of well-performing agents
    well_performing_agents = [a for a in self.agents if np.mean(a.performance_history[-10:]) > 0.7]
    if well_performing_agents:
        average_knowledge = np.mean([a.knowledge for a in well_performing_agents], axis=0)
        agent.knowledge = average_knowledge
        print(f"Agent {agent.id} has been reset with average knowledge from well-performing agents")