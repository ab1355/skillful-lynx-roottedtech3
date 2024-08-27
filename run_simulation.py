import asyncio
from collaborative_intelligence import MultiAgentSystem, Config
from agent import Agent
import matplotlib.pyplot as plt

async def run_simulation(num_steps: int):
    config = Config()
    config.num_initial_agents = 10  # Increased number of agents
    config.num_steps = num_steps
    config.tasks_per_step = 10  # Increased number of tasks per step
    config.task_complexity_range = (0.5, 1.2)  # Adjusted task complexity range
    
    agents = [Agent(f"Agent_{i}", "classification", "classification") for i in range(config.num_initial_agents)]
    for agent in agents:
        agent.learning_rate = 0.2  # Increased learning rate
    
    mas = MultiAgentSystem(agents, config)
    
    performance_history = []
    
    for step in range(num_steps):
        print(f"Step {step + 1}/{num_steps}")
        performance = await mas.run_simulation(1)
        performance_history.append(performance)
    
    return performance_history

if __name__ == "__main__":
    num_steps = 100  # Increased number of steps
    performance_history = asyncio.run(run_simulation(num_steps))
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_steps + 1), performance_history)
    plt.title("System Performance Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Performance")
    plt.grid(True)
    plt.savefig("simulation_performance_improved.png")
    print("Simulation completed. Performance plot saved as simulation_performance_improved.png")
    
    print("Final system performance:", performance_history[-1])
    print("Average performance:", sum(performance_history) / len(performance_history))
    print("Best performance:", max(performance_history))