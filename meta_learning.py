import numpy as np

class MetaLearningAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_strategies = {
            'fast': {'learning_rate': 0.1, 'batch_size': 32},
            'slow': {'learning_rate': 0.01, 'batch_size': 128},
            'balanced': {'learning_rate': 0.05, 'batch_size': 64}
        }
        self.current_strategy = 'balanced'
        self.strategy_performance = {strategy: [] for strategy in self.learning_strategies}

    def update_learning_strategy(self):
        if all(self.strategy_performance.values()):
            best_strategy = max(self.strategy_performance, key=lambda k: np.mean(self.strategy_performance[k]))
            if best_strategy != self.current_strategy:
                print(f"Agent {self.id} switched from {self.current_strategy} to {best_strategy} strategy")
                self.current_strategy = best_strategy

    def learn(self, task_performance):
        self.strategy_performance[self.current_strategy].append(task_performance)
        self.update_learning_strategy()

# Add this to the MultiAgentSystem class
def meta_learning_update(self):
    for agent in self.agents:
        if isinstance(agent, MetaLearningAgent):
            agent.learn(np.mean(agent.performance_history[-10:]))  # Learn from recent performance