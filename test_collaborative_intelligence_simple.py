from collaborative_intelligence import Agent, MultiAgentSystem
import numpy as np

def test_collaborative_intelligence_simple():
    print("Testing Collaborative Intelligence Framework (Simple Version)")

    # Create agents
    agents = [Agent(f"Agent_{i}", np.random.randn(10)) for i in range(3)]
    mas = MultiAgentSystem(agents)

    print("\n1. Agent Creation:")
    for agent in agents:
        print(f"Created {agent.agent_id} with initial parameters: {agent.model_params}")

    print("\n2. Federated Learning:")
    # Simulate local updates
    for agent in agents:
        agent.model_params += np.random.randn(10) * 0.1
        print(f"{agent.agent_id} updated parameters: {agent.model_params}")
    
    # Perform federated learning
    mas.federated_learning_round()
    print("Federated learning round completed")
    print(f"Updated parameters for all agents: {agents[0].model_params}")

    print("\n3. Knowledge Exchange:")
    # Add some knowledge to agents
    for i, agent in enumerate(agents):
        agent.update_knowledge({f"key_{i}": f"value_{i}"})
        print(f"{agent.agent_id} knowledge: {agent.knowledge_base}")
    
    # Simulate knowledge exchange
    for i, agent in enumerate(agents):
        next_agent = agents[(i + 1) % len(agents)]
        knowledge = list(agent.knowledge_base.items())[0]
        next_agent.update_knowledge({knowledge[0]: knowledge[1]})
        print(f"{agent.agent_id} shared knowledge '{knowledge[0]}' with {next_agent.agent_id}")

    print("\nFinal knowledge state:")
    for agent in agents:
        print(f"{agent.agent_id} knowledge: {agent.knowledge_base}")

    print("\nCollaborative Intelligence test completed successfully!")

if __name__ == "__main__":
    test_collaborative_intelligence_simple()