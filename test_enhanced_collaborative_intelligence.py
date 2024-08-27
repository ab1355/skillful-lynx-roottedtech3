import asyncio
from collaborative_intelligence import Agent, MultiAgentSystem
from task import Task
import random
import matplotlib.pyplot as plt

async def test_enhanced_collaborative_intelligence():
    print("Testing Enhanced Collaborative Intelligence Framework")

    # Create specialized agents
    agents = [
        Agent("Agent_1", "classification", "classification"),
        Agent("Agent_2", "regression", "regression"),
        Agent("Agent_3", "clustering", "clustering")
    ]
    mas = MultiAgentSystem(agents)

    # Simulate initial tasks
    task_types = ["classification", "regression", "clustering"]
    tasks = [Task(f"Task_{i}", random.uniform(0.3, 1.5), random.choice(task_types)) for i in range(50)]

    print("\n1. Initial Task Allocation and Processing:")
    await mas.allocate_tasks(tasks)
    await mas.process_all_tasks()

    print("Initial task allocation and processing completed")
    for agent in agents:
        print(f"{agent.agent_id} reputation: {agent.reputation:.2f}")

    print("\n2. Extended Simulation:")
    num_steps = 1000  # Increased simulation steps
    for step in range(num_steps):
        if step % 100 == 0:
            print(f"Step {step}/{num_steps}")
        await mas.run_simulation(1)
    final_performance = mas.evaluate_system_performance()

    print(f"\nFinal system performance: {final_performance:.2f}")
    print(f"Final number of agents: {len(mas.agents)}")

    print("\n3. Agent Specializations:")
    for agent in mas.agents:
        print(f"{agent.agent_id} final specialization: {agent.specialization}")
        print(f"  Specialization strength: {agent.specialization_strength:.2f}")
        print(f"  Expertise levels: {agent.expertise_level}")
        print(f"  Ramp-up boost: {agent.ramp_up_boost:.2f}")
        if agent.mentor:
            print(f"  Mentor: {agent.mentor.agent_id}")
        if agent.mentee:
            print(f"  Mentee: {agent.mentee.agent_id}")

    print("\n4. Knowledge Sharing Statistics:")
    for agent in mas.agents:
        print(f"{agent.agent_id} knowledge base size: {len(agent.knowledge_base)}")
        print(f"  Knowledge specialization: {dict(agent.knowledge_specialization)}")
        print(f"  Top 5 knowledge topics: {list(agent.knowledge_base.keys())[:5]}")

    print("\n5. Task Performance by Type:")
    for agent in mas.agents:
        print(f"{agent.agent_id} task performance:")
        for task_type, performance in agent.performance_by_task_type.items():
            success_rate = performance['success'] / max(1, performance['total'])
            print(f"  {task_type}: {success_rate:.2f} ({performance['total']} tasks)")

    print("\n6. Workload Balance:")
    total_tasks = sum(agent.total_tasks_processed for agent in mas.agents)
    for agent in mas.agents:
        print(f"{agent.agent_id}: {agent.total_tasks_processed} tasks ({agent.total_tasks_processed/total_tasks*100:.2f}% of total)")

    print("\n7. System Log Highlights:")
    log = mas.get_log()
    print(f"Total log entries: {len(log)}")
    print("Last 20 log entries:")
    for entry in log[-20:]:
        print(entry)

    print("\n8. Long-term Performance:")
    performance_history = mas.get_performance_history()
    plt.figure(figsize=(10, 6))
    plt.plot(performance_history)
    plt.title("System Performance Over Time")
    plt.xlabel("Evaluation Interval")
    plt.ylabel("System Performance")
    plt.savefig("system_performance.png")
    print("Long-term performance graph saved as 'system_performance.png'")

    print("\n9. Workload Distribution Over Time:")
    workload_history = mas.get_workload_history()
    plt.figure(figsize=(12, 6))
    for agent_id in workload_history[0].keys():
        agent_workload = [wl[agent_id] for wl in workload_history if agent_id in wl]
        plt.plot(agent_workload, label=agent_id)
    plt.title("Workload Distribution Over Time")
    plt.xlabel("Evaluation Interval")
    plt.ylabel("Proportion of Total Workload")
    plt.legend()
    plt.savefig("workload_distribution.png")
    print("Workload distribution graph saved as 'workload_distribution.png'")

    print("\n10. Specialization Changes:")
    specialization_changes = mas.get_specialization_changes()
    print(f"Total specialization changes: {len(specialization_changes)}")
    print("Last 10 specialization changes:")
    for change in specialization_changes[-10:]:
        print(f"Time {change[0]}: {change[1]} changed from {change[2]} to {change[3]}")

    print("\n11. Long-term Performance Analysis:")
    long_term_performance = mas.get_long_term_performance()
    plt.figure(figsize=(10, 6))
    plt.plot(long_term_performance)
    plt.title("Long-term System Performance")
    plt.xlabel("Analysis Interval")
    plt.ylabel("Average Performance")
    plt.savefig("long_term_performance.png")
    print("Long-term performance graph saved as 'long_term_performance.png'")

    print("\n12. Domain Performance:")
    domain_performance = mas.get_domain_performance()
    plt.figure(figsize=(12, 6))
    for domain, performance in domain_performance.items():
        plt.plot(performance, label=domain)
    plt.title("Domain Performance Over Time")
    plt.xlabel("Evaluation Interval")
    plt.ylabel("Average Performance")
    plt.legend()
    plt.savefig("domain_performance.png")
    print("Domain performance graph saved as 'domain_performance.png'")

    print("\n13. Mentoring Impact:")
    mentoring_reports = mas.get_mentoring_reports()
    for i, report in enumerate(mentoring_reports[-5:], 1):
        print(f"Report {i}:")
        for mentor, data in report.items():
            print(f"  Mentor {mentor} - Mentee {data['mentee']}:")
            for domain, improvement in data['improvements'].items():
                print(f"    {domain}: {improvement:.4f}")

    print("\n14. Final Performance Thresholds:")
    print(f"Add agent threshold: {mas.add_agent_threshold:.2f}")
    print(f"Remove agent threshold: {mas.remove_agent_threshold:.2f}")

    print("\nEnhanced Collaborative Intelligence test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_enhanced_collaborative_intelligence())