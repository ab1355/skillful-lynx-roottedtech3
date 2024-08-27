import asyncio
from collaborative_intelligence import MultiAgentSystem, Config
from agent import Agent
from task import Task
import numpy as np
import matplotlib.pyplot as plt

print("All imports successful")

# Create a simple configuration
config = Config()
config.num_initial_agents = 3
config.num_steps = 10

# Create agents
agents = [Agent(f"Agent_{i}", "classification", "classification") for i in range(config.num_initial_agents)]

# Create a MultiAgentSystem
mas = MultiAgentSystem(agents, config)

print("MultiAgentSystem created successfully")

# Create a simple task
task = Task("Test_Task", 0.5, "classification")

print("Task created successfully")

# Test numpy
array = np.array([1, 2, 3])
print("Numpy array:", array)

# Test matplotlib
plt.plot([1, 2, 3], [1, 2, 3])
plt.title("Test Plot")
plt.savefig("test_plot.png")
print("Matplotlib plot saved as test_plot.png")

print("All tests completed successfully")