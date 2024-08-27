import asyncio
import matplotlib.pyplot as plt
from collaborative_intelligence import MultiAgentSystem, Config
from agent import Agent
import logging

async def run_simulation(config_file=None):
    config = Config(config_file)

    # Create initial agents
    agents = [
        Agent(f"Agent_{i}", "classification", "classification") for i in range(config.num_initial_agents // 3)
    ] + [
        Agent(f"Agent_{i+3}", "regression", "regression") for i in range(config.num_initial_agents // 3)
    ] + [
        Agent(f"Agent_{i+6}", "clustering", "clustering") for i in range(config.num_initial_agents // 3)
    ]

    # Create MultiAgentSystem
    mas = MultiAgentSystem(agents, config)

    # Run simulation
    logging.info(f"Running simulation for {config.num_steps} steps...")
    final_performance = await mas.run_simulation(config.num_steps)

    # Get results
    performance_history = mas.performance_history
    workload_history = mas.workload_history
    specialization_changes = mas.specialization_changes
    long_term_performance = mas.long_term_performance
    domain_performance = mas.domain_performance

    # Visualize results
    visualize_results(performance_history, workload_history, specialization_changes, long_term_performance, domain_performance)

    logging.info(f"Final system performance: {final_performance}")
    logging.info(f"Number of specialization changes: {len(specialization_changes)}")
    logging.info(f"Final number of agents: {len(mas.agents)}")

def visualize_results(performance_history, workload_history, specialization_changes, long_term_performance, domain_performance):
    plt.figure(figsize=(15, 10))

    # System Performance
    plt.subplot(2, 2, 1)
    plt.plot(performance_history)
    plt.title("System Performance Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Performance")

    # Domain Performance
    plt.subplot(2, 2, 2)
    for domain, perf in domain_performance.items():
        plt.plot(perf, label=domain)
    plt.title("Domain Performance Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Performance")
    plt.legend()

    # Workload History
    plt.subplot(2, 2, 3)
    plt.plot(workload_history)
    plt.title("System Workload Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Workload")

    # Long-term Performance
    plt.subplot(2, 2, 4)
    plt.plot(long_term_performance)
    plt.title("Long-term System Performance")
    plt.xlabel("Time Step")
    plt.ylabel("Performance Trend")

    plt.tight_layout()
    plt.savefig("system_performance_visualization.png")
    plt.close()

    # Specialization Changes
    plt.figure(figsize=(12, 6))
    for change in specialization_changes:
        plt.plot([change[0], change[0]], [0, 1], 'r-')
    plt.title("Agent Specialization Changes")
    plt.xlabel("Time Step")
    plt.yticks([])
    plt.savefig("specialization_changes.png")
    plt.close()

if __name__ == "__main__":
    asyncio.run(run_simulation())