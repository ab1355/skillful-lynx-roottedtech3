import asyncio
from collaborative_intelligence import Agent, MultiAgentSystem
from models import SimpleNNModel
from task import Task

async def test_enhanced_collaborative_intelligence():
    print("Testing Enhanced Collaborative Intelligence Framework")

    # Create specialized agents
    agents = [
        Agent("Agent_1", SimpleNNModel(10, 5, 1), "classification"),
        Agent("Agent_2", SimpleNNModel(10, 5, 1), "regression"),
        Agent("Agent_3", SimpleNNModel(10, 5, 1), "clustering")
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

    print("\n2. Federated Learning:")
    mas.federated_learning_round()
    print("Federated learning round completed")

    print("\n3. Collaborative Knowledge Exchange:")
    # Add some knowledge to agents
    agents[0].update_knowledge({"important_data": {"value": 42, "confidence": 0.9}})
    agents[1].update_knowledge({"useful_info": {"value": "ABC", "confidence": 0.8}})
    agents[2].update_knowledge({"critical_knowledge": {"value": [1, 2, 3], "confidence": 0.75}})

    await mas.collaborative_exchange()
    print("Collaborative exchange completed")

    for agent in agents:
        print(f"{agent.agent_id} knowledge: {agent.knowledge_base}")

    print("\n4. Information Request:")
    result = await agents[2].request_information("important_data", mas)
    print(f"Agent_3 requested 'important_data': {result}")

    print("\nEnhanced Collaborative Intelligence test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_enhanced_collaborative_intelligence())