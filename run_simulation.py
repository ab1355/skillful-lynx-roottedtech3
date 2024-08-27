import asyncio
import random
from collaborative_intelligence import MultiAgentSystem, Config
from agent import Agent
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def run_simulation(num_steps: int):
    config = Config()
    config.num_initial_agents = 5  # Reduced from 10
    config.num_steps = num_steps
    config.tasks_per_step = 5  # Reduced from 15
    config.task_complexity_range = (0.2, 0.7)
    config.remove_agent_threshold = 0.2
    config.mentoring_threshold = 0.4
    config.mentoring_boost = 0.1
    config.environmental_factor_range = (0, 0.2)
    config.collaboration_threshold = 0.6
    
    agents = [Agent(f"Agent_{i}", 
                    random.choice(["classification", "regression", "clustering", "natural_language_processing", "computer_vision"]),
                    random.choice(["basic", "intermediate", "advanced"])) 
              for i in range(config.num_initial_agents)]
    
    mas = MultiAgentSystem(agents, config)
    
    performance_history = []
    agent_count_history = []
    entropy_history = []
    
    for step in range(num_steps):
        logging.info(f"Step {step + 1}/{num_steps}")
        performance = await mas.run_simulation(1)
        performance_history.append(performance)
        agent_count_history.append(len(mas.agents))
        
        # Detect and address biases
        biases = mas.detect_biases()
        if biases:
            logging.info(f"Detected biases: {biases}")
            await mas.address_biases(biases)
        
        # Calculate system entropy
        entropies = mas.calculate_system_entropy()
        entropy_history.append(sum(entropies.values()) / len(entropies))
        
        if step % 5 == 0:  # Reduced from 10
            logging.info(f"Current system entropy: {entropies}")
    
    return performance_history, agent_count_history, entropy_history, mas

def plot_results(performance_history, agent_count_history, entropy_history, num_steps):
    plt.figure(figsize=(15, 15))
    
    plt.subplot(3, 1, 1)
    plt.plot(range(1, num_steps + 1), performance_history)
    plt.title("System Performance Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Performance")
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(range(1, num_steps + 1), agent_count_history)
    plt.title("Number of Agents Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Number of Agents")
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(range(1, num_steps + 1), entropy_history)
    plt.title("System Entropy Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Entropy")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("simulation_results.png")
    logging.info("Simulation results plot saved as simulation_results.png")

if __name__ == "__main__":
    num_steps = 20  # Reduced from 50
    performance_history, agent_count_history, entropy_history, mas = asyncio.run(run_simulation(num_steps))
    
    plot_results(performance_history, agent_count_history, entropy_history, num_steps)
    
    logging.info(f"Final system performance: {performance_history[-1]:.4f}")
    logging.info(f"Average performance: {sum(performance_history) / len(performance_history):.4f}")
    logging.info(f"Best performance: {max(performance_history):.4f}")
    logging.info(f"Final number of agents: {agent_count_history[-1]}")
    logging.info(f"Average number of agents: {sum(agent_count_history) / len(agent_count_history):.2f}")
    logging.info(f"Final system entropy: {entropy_history[-1]:.4f}")
    
    # Additional statistics
    specializations = [agent.specialization for agent in mas.agents]
    sub_specializations = [agent.sub_specialization for agent in mas.agents]
    learning_strategies = [agent.learning_strategy for agent in mas.agents]
    
    logging.info("\nFinal agent specializations:")
    for spec in set(specializations):
        logging.info(f"  {spec}: {specializations.count(spec)}")
    
    logging.info("\nFinal agent sub-specializations:")
    for sub_spec in set(sub_specializations):
        logging.info(f"  {sub_spec}: {sub_specializations.count(sub_spec)}")
    
    logging.info("\nFinal agent learning strategies:")
    for strategy in set(learning_strategies):
        logging.info(f"  {strategy}: {learning_strategies.count(strategy)}")
    
    # Log final biases
    final_biases = mas.detect_biases()
    if final_biases:
        logging.info("\nFinal detected biases:")
        for domain, bias in final_biases.items():
            logging.info(f"  {domain}: {bias}")
    else:
        logging.info("\nNo biases detected in the final state.")
    
    # Log final entropies
    final_entropies = mas.calculate_system_entropy()
    logging.info("\nFinal system entropies:")
    for domain, entropy_value in final_entropies.items():
        logging.info(f"  {domain}: {entropy_value:.4f}")

    # Additional visualizations and analyses
    mas.visualize_skill_distribution()
    mas.analyze_collaboration_patterns()
    mas.analyze_learning_strategies()
    mas.analyze_task_performance()

    logging.info("\nSimulation completed. Check the generated plots and log file for detailed results.")