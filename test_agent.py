import unittest
import numpy as np
from agent import Agent
from task import Task

class TestAgent(unittest.TestCase):
    def setUp(self):
        self.agent = Agent("test_agent", "classification", "basic")

    def test_initialization(self):
        self.assertEqual(self.agent.id, "test_agent")
        self.assertEqual(self.agent.primary_domain, "classification")
        self.assertEqual(self.agent.specialization, "basic")
        self.assertIsInstance(self.agent.knowledge, np.ndarray)
        self.assertIsInstance(self.agent.skills, dict)

    def test_process_task(self):
        task = Task("test_task", 0.5, "classification", "basic", "prediction")
        result = self.agent.process_task(task, 0.1)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 1)

    def test_update_knowledge(self):
        task = Task("test_task", 0.5, "classification", "basic", "prediction")
        initial_knowledge = self.agent.knowledge.copy()
        self.agent.update_knowledge(task, 0.8)
        self.assertFalse(np.array_equal(initial_knowledge, self.agent.knowledge))

    def test_adapt_specialization(self):
        # Set knowledge to favor a specific domain and subdomain
        self.agent.knowledge[2, 1, :] = 0.9  # Set clustering/intermediate knowledge high
        self.agent.adapt_specialization()
        self.assertEqual(self.agent.specialization, "clustering")
        self.assertEqual(self.agent.sub_specialization, "intermediate")

    def test_propose_hypothesis(self):
        hypothesis = self.agent.propose_hypothesis()
        self.assertIn(hypothesis, self.agent.hypotheses)
        self.assertRegex(hypothesis, r"Improving (theory|implementation|optimization) in \w+/\w+ will lead to better performance")

    def test_test_hypothesis(self):
        hypothesis = self.agent.propose_hypothesis()
        initial_knowledge = self.agent.knowledge.copy()
        self.agent.test_hypothesis(hypothesis, True)
        self.assertFalse(np.array_equal(initial_knowledge, self.agent.knowledge))
        self.assertNotIn(hypothesis, self.agent.hypotheses)

    def test_update_reputation(self):
        initial_reputation = self.agent.reputation
        self.agent.update_reputation(0.8)
        self.assertGreater(self.agent.reputation, initial_reputation)

    def test_explore(self):
        domain, sub_domain = self.agent.explore()
        self.assertIn(domain, ["classification", "regression", "clustering", "natural_language_processing", "computer_vision"])
        self.assertIn(sub_domain, ["basic", "intermediate", "advanced"])

    def test_decompose_task(self):
        task = Task("test_task", 0.5, "classification", "basic", "prediction")
        subtasks = self.agent.decompose_task(task)
        self.assertEqual(len(subtasks), 3)
        for subtask in subtasks:
            self.assertIsInstance(subtask, Task)
            self.assertEqual(subtask.complexity, 0.25)

    def test_mentor(self):
        mentee = Agent("mentee", "regression", "advanced")
        task = Task("test_task", 0.5, "classification", "basic", "prediction")
        initial_mentee_knowledge = mentee.knowledge.copy()
        self.agent.mentor(mentee, task)
        self.assertFalse(np.array_equal(initial_mentee_knowledge, mentee.knowledge))

    def test_collaborate(self):
        collaborator = Agent("collaborator", "clustering", "intermediate")
        task = Task("test_task", 0.5, "classification", "basic", "prediction")
        result = self.agent.collaborate(collaborator, task)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 1)

if __name__ == '__main__':
    unittest.main()