import asyncio
from collaborative_intelligence import Agent, MultiAgentSystem
from task import Task
import numpy as np

async def test_enhanced_collaborative_intelligence():
    print("Testing Enhanced Collaborative Intelligence Framework")

    # Create specialized agents
    agents = [
        Agent("Agent_1", "classification", "classification"),
        Agent("Agent_2", "regression", "regression"),
        Agent("Agent_3", "clustering", "clustering")
    ]
    mas = MultiAgentSystem(agents)

    # Simulate tasks
    tasks = [
        Task("Task_1", 0.5, "classification"),
        Task("Task_2", 0.7, "regression"),
        Task("Task_3", 0.6, "clustering"),
        Task("Task_4", 0.8, "classification"),
        Task("Task_5", 0.4, "regression")
    ]

    print("\n1. Task Allocation and Processing:")
    await mas.allocate_tasks(tasks)
    await mas.process_all_tasks()

    print("Tasks allocated and processed")
    for agent in agents:
        print(f"{agent.agent_id} reputation: {agent.reputation}")

    print("\n2. Federated Learning with Privacy:")
    # Simulate local training data
    for agent in agents:
        X = np.random.rand(100, 10)
        if agent.specialization == "classification":
            y = np.random.randint(0, 2, 100)  # Binary classification
        elif agent.specialization == "regression":
            y = np.random.rand(100)  # Continuous values for regression
        else:  # Clustering
            y = None  # Clustering doesn't need labels
        agent.train_on_local_data(X, y)

    mas.federated_learning_round()
    print("Federated learning round completed with privacy-preserving aggregation")

    print("\n3. Collaborative Knowledge Exchange:")
    # Add some knowledge to agents
    agents[0].update_knowledge({"important_data": {"value": 42, "confidence": 0.9}})
    agents[1].update_knowledge({"useful_info": {"value": "ABC", "confidence": 0.8}})
    agents[2].update_knowledge({"critical_knowledge": {"value": [1, 2, 3], "confidence": 0.75}})

    await mas.collaborative_exchange()
    print("Collaborative exchange completed")

    for agent in agents:
        print(f"{agent.agent_id} knowledge: {agent.knowledge_base}")

    print("\n4. System Simulation:")
    num_steps = 10  # Reduce the number of simulation steps
    for step in range(num_steps):
        print(f"Simulation step {step + 1}/{num_steps}")
        await mas.process_all_tasks()
        mas.federated_learning_round()
        await mas.collaborative_exchange()

    final_performance = mas.evaluate_system_performance()
    print(f"Final system performance: {final_performance}")
    print(f"Final number of agents: {len(mas.agents)}")

    print("\nEnhanced Collaborative Intelligence test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_enhanced_collaborative_intelligence())