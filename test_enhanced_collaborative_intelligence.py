import asyncio
from collaborative_intelligence import MultiAgentSystem
from agent import Agent
import matplotlib.pyplot as plt

async def run_simulation():
    # Create initial agents
    agents = [
        Agent(f"Agent_{i}", "classification", "classification") for i in range(3)
    ] + [
        Agent(f"Agent_{i+3}", "regression", "regression") for i in range(3)
    ] + [
        Agent(f"Agent_{i+6}", "clustering", "clustering") for i in range(3)
    ]

    # Create MultiAgentSystem
    mas = MultiAgentSystem(agents)

    # Run simulation
    final_performance = await mas.run_simulation(1000)

    # Get results
    performance_history = mas.get_performance_history()
    workload_history = mas.get_workload_history()
    specialization_changes = mas.get_specialization_changes()
    long_term_performance = mas.get_long_term_performance()
    domain_performance = mas.get_domain_performance()

    # Plot results
    plt.figure(figsize=(12, 8))
    plt.plot(performance_history)
    plt.title("System Performance Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Performance")
    plt.savefig("system_performance.png")
    plt.close()

    plt.figure(figsize=(12, 8))
    for domain, perf in domain_performance.items():
        plt.plot(perf, label=domain)
    plt.title("Domain Performance Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Performance")
    plt.legend()
    plt.savefig("domain_performance.png")
    plt.close()

    print(f"Final system performance: {final_performance}")
    print(f"Number of specialization changes: {len(specialization_changes)}")
    print(f"Final number of agents: {len(mas.agents)}")

if __name__ == "__main__":
    asyncio.run(run_simulation())