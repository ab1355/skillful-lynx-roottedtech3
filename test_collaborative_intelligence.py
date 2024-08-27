import unittest
import asyncio
from collaborative_intelligence import MultiAgentSystem, Config
from agent import Agent
from task import Task

class TestMultiAgentSystem(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.agents = [Agent(f"agent_{i}", "classification", "basic") for i in range(5)]
        self.system = MultiAgentSystem(self.agents, self.config)

    def test_initialization(self):
        self.assertEqual(len(self.system.agents), 5)
        self.assertIsInstance(self.system.config, Config)
        self.assertEqual(self.system.current_time, 0)

    def test_generate_tasks(self):
        tasks = self.system._generate_tasks()
        self.assertEqual(len(tasks), self.config.tasks_per_step)
        for task in tasks:
            self.assertIsInstance(task, Task)

    def test_choose_agent_for_task(self):
        task = Task("test_task", 0.5, "classification", "basic", "prediction")
        agent_workloads = {agent: 0 for agent in self.agents}
        domain_workloads = {"classification": 1}
        chosen_agent = self.system._choose_agent_for_task(task, agent_workloads, domain_workloads)
        self.assertIn(chosen_agent, self.agents)

    def test_evaluate_system_performance(self):
        # Add some performance history to agents
        for agent in self.agents:
            agent.performance_history = [0.5, 0.6, 0.7]
            agent.task_queue._queue = [Task("test_task", 0.5, "classification", "basic", "prediction") for _ in range(3)]
        
        performance = self.system.evaluate_system_performance()
        self.assertGreaterEqual(performance, 0)
        self.assertLessEqual(performance, 1)

    def test_federated_learning_round(self):
        initial_knowledge = [agent.knowledge.copy() for agent in self.agents]
        self.system.federated_learning_round()
        for agent, initial in zip(self.agents, initial_knowledge):
            self.assertFalse((agent.knowledge == initial).all())

    @unittest.skip("This test requires running the full simulation")
    def test_run_simulation(self):
        loop = asyncio.get_event_loop()
        final_performance = loop.run_until_complete(self.system.run_simulation(10))
        self.assertGreaterEqual(final_performance, 0)
        self.assertLessEqual(final_performance, 1)

    def test_detect_biases(self):
        # Intentionally create a bias
        for agent in self.agents:
            agent.knowledge[0, :, :] = 0.9  # High knowledge in classification
            agent.knowledge[1, :, :] = 0.1  # Low knowledge in regression
        
        biases = self.system.detect_biases()
        self.assertIn("regression", biases)
        self.assertIn("Low overall knowledge in regression", biases["regression"])

    def test_calculate_system_entropy(self):
        entropies = self.system.calculate_system_entropy()
        for domain, entropy_value in entropies.items():
            self.assertGreaterEqual(entropy_value, 0)

    @unittest.skip("This test generates visualizations")
    def test_visualizations(self):
        self.system._visualize_agent_network()
        self.system._visualize_knowledge_distribution()
        self.system._visualize_performance_over_time()
        self.system._visualize_knowledge_graph()
        self.system.visualize_skill_distribution()
        # Add assertions to check if files are created

    def test_analyze_collaboration_patterns(self):
        # Add some edges to the agent interaction graph
        self.system.agent_interactions.add_edge("agent_0", "agent_1", weight=0.5)
        self.system.agent_interactions.add_edge("agent_0", "agent_2", weight=0.7)
        self.system.agent_interactions.add_edge("agent_1", "agent_3", weight=0.6)
        
        self.system.analyze_collaboration_patterns()
        # Add assertions to check if the analysis produces expected results

    def test_analyze_learning_strategies(self):
        self.system.analyze_learning_strategies()
        # Add assertions to check if the analysis produces expected results

    def test_analyze_task_performance(self):
        # Add some task history
        self.system.task_history["classification"] = [(1, 0.7), (2, 0.8), (3, 0.9)]
        self.system.task_history["regression"] = [(1, 0.5), (2, 0.6), (3, 0.7)]
        
        self.system.analyze_task_performance()
        # Add assertions to check if the analysis produces expected results

if __name__ == '__main__':
    unittest.main()