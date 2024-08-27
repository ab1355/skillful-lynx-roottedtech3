import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats

class Task:
    def __init__(self, name, domain, sub_domain, task_type, complexity):
        self.name = name
        self.domain = domain
        self.sub_domain = sub_domain
        self.task_type = task_type
        self.complexity = complexity

class Agent:
    def __init__(self, id, specialization, sub_specialization):
        self.id = id
        self.specialization = specialization
        self.sub_specialization = sub_specialization
        self.skills = {
            "problem_solving": random.random(),
            "creativity": random.random(),
            "analytical_thinking": random.random(),
            "communication": random.random()
        }
        self.knowledge = np.random.rand(100)  # 100-dimensional knowledge vector
        self.performance_history = []

class EmergentAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_history = defaultdict(list)

    def update_task_history(self, task, performance):
        self.task_history[task.domain].append(performance)

    def get_emergent_specialization(self):
        if not self.task_history:
            return None
        return max(self.task_history, key=lambda k: sum(self.task_history[k]) / len(self.task_history[k]))

class CognitiveLoadAgent(EmergentAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cognitive_load = 0
        self.max_load = 100

    def can_accept_task(self, task):
        return self.cognitive_load + task.complexity <= self.max_load

    def assign_task(self, task):
        if self.can_accept_task(task):
            self.cognitive_load += task.complexity
            self.current_task = task
            return True
        return False

    def complete_task(self, task):
        self.cognitive_load -= task.complexity

class MetaLearningAgent(CognitiveLoadAgent):
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

class AdvancedCollaborativeIntelligence:
    def __init__(self, num_agents):
        self.agents = [MetaLearningAgent(f"Agent_{i}", "general", "general") for i in range(num_agents)]
        self.tasks = []
        self.network = nx.Graph()
        self.initialize_network()
        self.config = type('Config', (), {'task_complexity_range': (0.1, 1.0)})()

    def initialize_network(self):
        for agent in self.agents:
            self.network.add_node(agent)

    def run_simulation(self, num_iterations):
        for i in range(num_iterations):
            print(f"\nIteration {i+1}/{num_iterations}")
            self.generate_tasks()
            self.assign_tasks()
            self.process_tasks()
            self.update_system()
            self.continuous_learning_update()
            self.inter_agent_teaching_round()
            self.adjust_task_complexity()
            
            # Print summary every 10 iterations
            if (i+1) % 10 == 0:
                self.print_summary()

    def generate_tasks(self):
        num_tasks = random.randint(1, len(self.agents))
        for _ in range(num_tasks):
            complexity = random.uniform(*self.config.task_complexity_range)
            task = Task(f"Task_{random.randint(1000, 9999)}", "general", "general", "analysis", complexity)
            self.tasks.append(task)

    def assign_tasks(self):
        for task in self.tasks:
            self.assign_task_with_load_balancing(task)

    def process_tasks(self):
        for agent in self.agents:
            if hasattr(agent, 'current_task'):
                performance = random.random()  # Simulated task performance
                agent.performance_history.append(performance)
                if isinstance(agent, EmergentAgent):
                    agent.update_task_history(agent.current_task, performance)
                if isinstance(agent, CognitiveLoadAgent):
                    agent.complete_task(agent.current_task)
                if isinstance(agent, MetaLearningAgent):
                    agent.learn(performance)
                delattr(agent, 'current_task')

    def update_system(self):
        self.evolve_agents()
        self.update_network()
        self.update_emergent_specializations()
        self.meta_learning_update()
        self.system_anomaly_detection()

    def crossover(self, parent1, parent2):
        child = MetaLearningAgent(f"Child_{random.randint(1000, 9999)}", parent1.specialization, parent1.sub_specialization)
        child.skills = {k: (parent1.skills[k] + parent2.skills[k]) / 2 for k in parent1.skills}
        child.knowledge = (parent1.knowledge + parent2.knowledge) / 2
        return child

    def mutate(self, agent, mutation_rate=0.1):
        for skill in agent.skills:
            if random.random() < mutation_rate:
                agent.skills[skill] += random.uniform(-0.1, 0.1)
                agent.skills[skill] = max(0, min(1, agent.skills[skill]))
        return agent

    def evolve_agents(self, num_generations=1):
        for generation in range(num_generations):
            fitnesses = [np.mean(agent.performance_history) if agent.performance_history else 0 for agent in self.agents]
            parents = random.choices(self.agents, weights=fitnesses, k=len(self.agents))
            
            new_agents = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child = self.crossover(parents[i], parents[i+1])
                    child = self.mutate(child)
                    new_agents.append(child)
            
            self.agents = sorted(self.agents + new_agents, key=lambda a: np.mean(a.performance_history) if a.performance_history else 0, reverse=True)[:len(self.agents)]
            print(f"Generation {generation + 1} complete. Best fitness: {np.mean(self.agents[0].performance_history) if self.agents[0].performance_history else 0}")

    def update_network(self):
        for agent in self.agents:
            connected_agents = list(self.network.neighbors(agent))
            if len(connected_agents) < 3:  # Ensure minimum connections
                potential_connections = set(self.agents) - set(connected_agents) - {agent}
                if potential_connections:
                    new_connection = random.choice(list(potential_connections))
                    self.network.add_edge(agent, new_connection)
            elif random.random() < 0.1:  # 10% chance to remove a connection
                connection_to_remove = random.choice(connected_agents)
                self.network.remove_edge(agent, connection_to_remove)

        # Visualize the network
        plt.figure(figsize=(10, 10))
        nx.draw(self.network, with_labels=True, node_color='lightblue', node_size=500, font_size=8, font_weight='bold')
        plt.title("Agent Collaboration Network")
        plt.savefig("agent_network.png")
        plt.close()
        print("Network visualization saved as agent_network.png")

    def update_emergent_specializations(self):
        for agent in self.agents:
            if isinstance(agent, EmergentAgent):
                new_specialization = agent.get_emergent_specialization()
                if new_specialization and new_specialization != agent.specialization:
                    print(f"Agent {agent.id} specialized from {agent.specialization} to {new_specialization}")
                    agent.specialization = new_specialization

    def assign_task_with_load_balancing(self, task):
        suitable_agents = [agent for agent in self.agents if agent.can_accept_task(task)]
        if suitable_agents:
            chosen_agent = min(suitable_agents, key=lambda a: a.cognitive_load)
            if chosen_agent.assign_task(task):
                print(f"Assigned task to Agent {chosen_agent.id}. New cognitive load: {chosen_agent.cognitive_load}")
                return True
        print("No suitable agent found for task")
        return False

    def meta_learning_update(self):
        for agent in self.agents:
            if isinstance(agent, MetaLearningAgent):
                agent.learn(np.mean(agent.performance_history[-10:]) if agent.performance_history else 0)

    def detect_anomalies(self, data, threshold=3):
        if len(data) < 2:
            return [False] * len(data)
        z_scores = np.abs(stats.zscore(data))
        return z_scores > threshold

    def system_anomaly_detection(self):
        agent_performances = [np.mean(agent.performance_history[-10:]) if agent.performance_history else 0 for agent in self.agents]
        anomalies = self.detect_anomalies(agent_performances)
        
        for i, is_anomaly in enumerate(anomalies):
            if is_anomaly:
                print(f"Anomaly detected in Agent {self.agents[i].id}")
                self.heal_agent(self.agents[i])

    def heal_agent(self, agent):
        print(f"Initiating self-healing for Agent {agent.id}")
        # Reset agent's knowledge to the average of well-performing agents
        well_performing_agents = [a for a in self.agents if np.mean(a.performance_history[-10:]) > 0.7 if a.performance_history]
        if well_performing_agents:
            average_knowledge = np.mean([a.knowledge for a in well_performing_agents], axis=0)
            agent.knowledge = average_knowledge
            print(f"Agent {agent.id} has been reset with average knowledge from well-performing agents")

    def continuous_learning_update(self):
        for agent in self.agents:
            if agent.performance_history:
                new_data = np.random.rand(len(agent.knowledge))  # Simulated new data
                learning_rate = 0.1
                agent.knowledge = (1 - learning_rate) * agent.knowledge + learning_rate * new_data
                print(f"Updated model for Agent {agent.id}")

    def inter_agent_teaching_round(self):
        agents_sorted = sorted(self.agents, key=lambda a: np.mean(a.performance_history) if a.performance_history else 0, reverse=True)
        for i in range(len(agents_sorted) // 2):
            teacher = agents_sorted[i]
            student = agents_sorted[-(i+1)]
            knowledge_transfer_rate = 0.2
            student.knowledge = (1 - knowledge_transfer_rate) * student.knowledge + knowledge_transfer_rate * teacher.knowledge
            print(f"Agent {teacher.id} taught Agent {student.id}")

    def adjust_task_complexity(self):
        system_performance = np.mean([np.mean(agent.performance_history) for agent in self.agents if agent.performance_history])
        if system_performance > 0.8:
            self.config.task_complexity_range = (self.config.task_complexity_range[0] * 1.1, 
                                                 min(self.config.task_complexity_range[1] * 1.1, 1.0))
        elif system_performance < 0.6:
            self.config.task_complexity_range = (max(self.config.task_complexity_range[0] * 0.9, 0.1), 
                                                 self.config.task_complexity_range[1] * 0.9)
        print(f"Adjusted task complexity range to {self.config.task_complexity_range}")

    def print_summary(self):
        avg_performance = np.mean([np.mean(agent.performance_history) for agent in self.agents if agent.performance_history])
        print(f"\nSystem Summary:")
        print(f"Average Agent Performance: {avg_performance:.2f}")
        print(f"Current Task Complexity Range: {self.config.task_complexity_range}")
        print(f"Number of Agents: {len(self.agents)}")
        print(f"Network Connections: {self.network.number_of_edges()}")

if __name__ == "__main__":
    num_agents = 10
    num_iterations = 100
    
    print("Initializing Advanced Collaborative Intelligence System")
    system = AdvancedCollaborativeIntelligence(num_agents)
    
    print(f"Starting simulation with {num_agents} agents for {num_iterations} iterations")
    system.run_simulation(num_iterations)
    
    print("\nSimulation complete. Final system state:")
    system.print_summary()
    print("\nNetwork visualization saved as agent_network.png")