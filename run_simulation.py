import asyncio
import random
from collaborative_intelligence import MultiAgentSystem, Config
from agent import Agent
import matplotlib.pyplot as plt

async def run_simulation(num_steps: int):
    config = Config()
    config.num_initial_agents = 15  # Increased number of agents
    config.num_steps = num_steps
    config.tasks_per_step = 15  # Increased number of tasks per step
    config.task_complexity_range = (0.5, 1.2)  # Adjusted task complexity range
    config.remove_agent_threshold = 0.3
    config.mentoring_threshold = 0.4
    config.mentoring_boost = 0.1
    
    agents = [Agent(f"Agent_{i}", random.choice(["classification", "regression", "clustering"]), random.choice(["classification", "regression", "clustering"])) for i in range(config.num_initial_agents)]
    for agent in agents:
        agent.learning_rate = 0.2  # Initial learning rate
    
    mas = MultiAgentSystem(agents, config)
    
    performance_history = []
    agent_count_history = []
    
    for step in range(num_steps):
        print(f"Step {step + 1}/{num_steps}")
        performance = await mas.run_simulation(1)
        performance_history.append(performance)
        agent_count_history.append(len(mas.agents))
    
    return performance_history, agent_count_history

if __name__ == "__main__":
    num_steps = 200  # Increased number of steps for a longer simulation
    performance_history, agent_count_history = asyncio.run(run_simulation(num_steps))
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(range(1, num_steps + 1), performance_history)
    plt.title("System Performance Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Performance")
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(range(1, num_steps + 1), agent_count_history)
    plt.title("Number of Agents Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Number of Agents")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("simulation_results.png")
    print("Simulation completed. Results plot saved as simulation_results.png")
    
    print("Final system performance:", performance_history[-1])
    print("Average performance:", sum(performance_history) / len(performance_history))
    print("Best performance:", max(performance_history))
    print("Final number of agents:", agent_count_history[-1])
    print("Average number of agents:", sum(agent_count_history) / len(agent_count_history))