import asyncio
from collaborative_intelligence import Agent, MultiAgentSystem
import numpy as np

async def test_collaborative_intelligence():
    print("Testing Collaborative Intelligence Framework")

    # Create agents
    agents = [Agent(f"Agent_{i}", np.random.randn(10)) for i in range(3)]
    mas = MultiAgentSystem(agents)

    # Simulate tasks
    tasks = [{"id": i, "data": np.random.randn(10)} for i in range(5)]

    print("\n1. Task Coordination:")
    await mas.coordinate_tasks(tasks)

    print("\n2. Federated Learning:")
    # Simulate local updates
    for agent in agents:
        agent.model_params += np.random.randn(10) * 0.1
    
    # Perform federated learning
    mas.federated_learning_round()
    print("Federated learning round completed")

    print("\n3. Collaborative Exchange:")
    # Add some knowledge to agents
    for i, agent in enumerate(agents):
        agent.update_knowledge({f"key_{i}": f"value_{i}"})
    
    # Perform collaborative exchange
    await mas.collaborative_exchange()

    print("\nCollaborative Intelligence test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_collaborative_intelligence())