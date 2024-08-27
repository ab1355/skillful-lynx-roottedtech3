import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import logging
from collaborative_intelligence import MultiAgentSystem, Config, Agent, Task

logging.basicConfig(level=logging.INFO)

class TestCollaborativeIntelligence(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.agents = [
            Agent(f"Agent_{i}", "classification", "binary") for i in range(3)
        ] + [
            Agent(f"Agent_{i}", "regression", "linear") for i in range(3, 5)
        ]
        self.system = MultiAgentSystem(self.agents, self.config)

    def test_generate_tasks(self):
        tasks = self.system._generate_tasks()
        self.assertEqual(len(tasks), self.config.tasks_per_step)
        for task in tasks:
            self.assertIsInstance(task, Task)
            self.assertTrue(self.config.task_complexity_range[0] <= task.complexity <= self.config.task_complexity_range[1])

    def test_choose_agent_for_task(self):
        task = Task("Test_Task", "classification", "binary", "classification", 0.5)
        task.task_type = "classification"  # Explicitly set task_type
        agent_workloads = {agent: 0 for agent in self.agents}
        domain_workloads = {"classification": 0}
        for agent in self.agents:
            agent.skills = {"classification": 0.5, "regression": 0.5}
            agent.performance_history = [0.5]
            agent.reputation = 1.0
        chosen_agent = self.system._choose_agent_for_task(task, agent_workloads, domain_workloads)
        self.assertIsInstance(chosen_agent, Agent)
        logging.info(f"Chosen agent: specialization={chosen_agent.specialization}, sub_specialization={chosen_agent.sub_specialization}")
        logging.info(f"Task: domain={task.domain}, sub_domain={task.sub_domain}, task_type={task.task_type}")
        self.assertEqual(chosen_agent.specialization, "binary")
        self.assertEqual(chosen_agent.sub_specialization, "basic")

    def test_evaluate_system_performance(self):
        for agent in self.agents:
            agent.performance_history = [0.5, 0.6, 0.7]
            agent.task_queue._queue = [MagicMock(complexity=0.5) for _ in range(3)]
        performance = self.system.evaluate_system_performance()
        self.assertTrue(0 <= performance <= 1)

    @patch('collaborative_intelligence.secure_aggregate')
    def test_federated_learning_round(self, mock_secure_aggregate):
        mock_secure_aggregate.return_value = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        for agent in self.agents:
            agent.knowledge = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
            agent.reputation = 1.0
        self.system.federated_learning_round()
        for agent in self.system.agents:
            self.assertFalse(np.array_equal(agent.knowledge, np.array([0.1, 0.2, 0.3, 0.4, 0.5])))

    def test_detect_biases(self):
        for i, agent in enumerate(self.system.agents):
            agent.knowledge = np.array([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i, 0.5 * i])
        biases = self.system.detect_biases()
        self.assertIsInstance(biases, dict)

    def test_calculate_system_entropy(self):
        for i, agent in enumerate(self.system.agents):
            agent.knowledge = np.array([0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i, 0.5 * i])
        entropies = self.system.calculate_system_entropy()
        self.assertIsInstance(entropies, dict)
        self.assertEqual(set(entropies.keys()), {"classification", "regression", "clustering", "natural_language_processing", "computer_vision"})

if __name__ == '__main__':
    unittest.main()