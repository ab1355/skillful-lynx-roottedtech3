import asyncio
from collaborative_intelligence import Agent, MultiAgentSystem
from task import Task
import numpy as np
import random

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
    task_types = ["classification", "regression", "clustering"]
    tasks = [Task(f"Task_{i}", random.uniform(0.3, 0.9), random.choice(task_types)) for i in range(100)]

    print("\n1. Task Allocation and Processing:")
    await mas.allocate_tasks(tasks)
    await mas.process_all_tasks()

    print("Initial task allocation and processing completed")
    for agent in agents:
        print(f"{agent.agent_id} reputation: {agent.reputation:.2f}")

    print("\n2. Extended Simulation:")
    num_steps = 100  # Increased simulation steps
    final_performance = await mas.run_simulation(num_steps)

    print(f"\nFinal system performance: {final_performance:.2f}")
    print(f"Final number of agents: {len(mas.agents)}")

    print("\n3. Agent Specializations:")
    for agent in mas.agents:
        print(f"{agent.agent_id} final specialization: {agent.specialization}")

    print("\n4. Knowledge Sharing Statistics:")
    for agent in mas.agents:
        print(f"{agent.agent_id} knowledge base size: {len(agent.knowledge_base)}")
        print(f"  Knowledge topics: {list(agent.knowledge_base.keys())}")

    print("\n5. Task Performance by Type:")
    for agent in mas.agents:
        print(f"{agent.agent_id} task performance:")
        for task_type, performance in agent.performance_by_task_type.items():
            success_rate = performance['success'] / max(1, performance['total'])
            print(f"  {task_type}: {success_rate:.2f} ({performance['total']} tasks)")

    print("\n6. Workload Balance:")
    total_tasks = sum(sum(perf['total'] for perf in agent.performance_by_task_type.values()) for agent in mas.agents)
    for agent in mas.agents:
        agent_tasks = sum(perf['total'] for perf in agent.performance_by_task_type.values())
        print(f"{agent.agent_id}: {agent_tasks} tasks ({agent_tasks/total_tasks*100:.2f}% of total)")

    print("\n7. System Log Highlights:")
    log = mas.get_log()
    print(f"Total log entries: {len(log)}")
    print("Last 10 log entries:")
    for entry in log[-10:]:
        print(entry)

    print("\nEnhanced Collaborative Intelligence test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_enhanced_collaborative_intelligence())