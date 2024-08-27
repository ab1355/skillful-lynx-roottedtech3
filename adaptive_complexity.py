import numpy as np

# Add this to the MultiAgentSystem class
def adjust_task_complexity(self):
    system_performance = self.evaluate_system_performance()
    if system_performance > 0.8:
        self.config.task_complexity_range = (self.config.task_complexity_range[0] * 1.1, 
                                             self.config.task_complexity_range[1] * 1.1)
    elif system_performance < 0.6:
        self.config.task_complexity_range = (self.config.task_complexity_range[0] * 0.9, 
                                             self.config.task_complexity_range[1] * 0.9)
    print(f"Adjusted task complexity range to {self.config.task_complexity_range}")

def generate_adaptive_task(self):
    complexity = np.random.uniform(*self.config.task_complexity_range)
    return Task("Adaptive_Task", "adaptive", "dynamic", "analysis", complexity)