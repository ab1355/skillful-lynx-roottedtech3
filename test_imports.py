import unittest
from collaborative_intelligence import MultiAgentSystem, Config
from agent import Agent
from task import Task

class TestImports(unittest.TestCase):
    def test_imports(self):
        config = Config()
        agent = Agent("Test_Agent", "classification", "basic")
        task = Task("Test_Task", 0.5, "classification", "basic", "prediction")
        mas = MultiAgentSystem([agent], config)

        self.assertIsInstance(config, Config)
        self.assertIsInstance(agent, Agent)
        self.assertIsInstance(task, Task)
        self.assertIsInstance(mas, MultiAgentSystem)

if __name__ == '__main__':
    unittest.main()