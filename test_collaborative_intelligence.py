import unittest
import asyncio
from collaborative_intelligence import MultiAgentSystem, Config
from agent import Agent
from task import Task

class TestCollaborativeIntelligence(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.agents = [
            Agent("Agent_1", "classification", "classification"),
            Agent("Agent_2", "regression", "regression"),
            Agent("Agent_3", "clustering", "clustering")
        ]
        self.mas = MultiAgentSystem(self.agents, self.config)

    def test_agent_creation(self):
        self.assertEqual(len(self.mas.agents), 3)
        self.assertEqual(self.mas.agents[0].name, "Agent_1")
        self.assertEqual(self.mas.agents[1].specialization, "regression")

    def test_task_allocation(self):
        tasks = [
            Task("Task_1", 0.5, "classification"),
            Task("Task_2", 0.7, "regression"),
            Task("Task_3", 0.9, "clustering")
        ]
        asyncio.run(self.mas.allocate_tasks(tasks))
        self.assertEqual(len(self.mas.agents[0].task_queue._queue), 1)
        self.assertEqual(len(self.mas.agents[1].task_queue._queue), 1)
        self.assertEqual(len(self.mas.agents[2].task_queue._queue), 1)

    def test_federated_learning(self):
        initial_knowledge = self.mas.agents[0].knowledge
        self.mas.federated_learning_round()
        self.assertNotEqual(initial_knowledge, self.mas.agents[0].knowledge)

    def test_collaborative_exchange(self):
        initial_knowledge = self.mas.agents[0].knowledge
        asyncio.run(self.mas.collaborative_exchange())
        self.assertNotEqual(initial_knowledge, self.mas.agents[0].knowledge)

    def test_agent_specialization_adjustment(self):
        initial_specialization = self.mas.agents[0].specialization
        self.mas._adjust_agent_specializations()
        # This test might fail sometimes due to randomness
        self.assertEqual(initial_specialization, self.mas.agents[0].specialization)

if __name__ == '__main__':
    unittest.main()