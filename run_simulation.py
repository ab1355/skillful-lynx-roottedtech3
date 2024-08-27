import asyncio
import random
from collaborative_intelligence import MultiAgentSystem, Config
from agent import Agent
import matplotlib.pyplot as plt

async def run_simulation(num_steps: int):
    config = Config()
    config.num_initial_agents = 10  # Reduced from 15
    config.num_steps = num_steps
    config.tasks_per_step = 10  # Reduced from 15
    config.task_complexity_range = (0.3, 0.8)
    config.remove_agent_threshold = 0.1
    config.mentoring_threshold = 0.4
    config.mentoring_boost = 0.1
    
    agents = [Agent(f"Agent_{i}", 
                    random.choice(["classification", "regression", "clustering"]),
                    random.choice(["basic", "intermediate", "advanced"])) 
              for i in range(config.num_initial_agents)]
    
    mas = MultiAgentSystem(agents, config)
    
    performance_history = []
    agent_count_history = []
    
    for step in range(num_steps):
        print(f"Step {step + 1}/{num_steps}")
        performance = await mas.run_simulation(1)
        performance_history.append(performance)
        agent_count_history.append(len(mas.agents))
    
    return performance_history, agent_count_history, mas

def plot_results(performance_history, agent_count_history, num_steps):
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
    print("Simulation results plot saved as simulation_results.png")

if __name__ == "__main__":
    num_steps = 50  # Reduced from 200
    performance_history, agent_count_history, mas = asyncio.run(run_simulation(num_steps))
    
    plot_results(performance_history, agent_count_history, num_steps)
    
    print("Final system performance:", performance_history[-1])
    print("Average performance:", sum(performance_history) / len(performance_history))
    print("Best performance:", max(performance_history))
    print("Final number of agents:", agent_count_history[-1])
    print("Average number of agents:", sum(agent_count_history) / len(agent_count_history))
    
    # Additional statistics
    specializations = [agent.specialization for agent in mas.agents]
    sub_specializations = [agent.sub_specialization for agent in mas.agents]
    learning_strategies = [agent.learning_strategy for agent in mas.agents]
    
    print("\nFinal agent specializations:", {spec: specializations.count(spec) for spec in set(specializations)})
    print("Final agent sub-specializations:", {sub_spec: sub_specializations.count(sub_spec) for sub_spec in set(sub_specializations)})
    print("Final agent learning strategies:", {strategy: learning_strategies.count(strategy) for strategy in set(learning_strategies)})