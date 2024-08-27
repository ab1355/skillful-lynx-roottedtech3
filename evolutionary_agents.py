import random
import numpy as np

def crossover(parent1, parent2):
    child = Agent(f"Child_{random.randint(1000, 9999)}", parent1.specialization, parent1.sub_specialization)
    child.skills = {k: (parent1.skills[k] + parent2.skills[k]) / 2 for k in parent1.skills}
    child.knowledge = (parent1.knowledge + parent2.knowledge) / 2
    return child

def mutate(agent, mutation_rate=0.1):
    for skill in agent.skills:
        if random.random() < mutation_rate:
            agent.skills[skill] += random.uniform(-0.1, 0.1)
            agent.skills[skill] = max(0, min(1, agent.skills[skill]))
    return agent

# Add this to the MultiAgentSystem class
def evolve_agents(self, num_generations=5):
    for generation in range(num_generations):
        fitnesses = [np.mean(agent.performance_history) for agent in self.agents]
        parents = random.choices(self.agents, weights=fitnesses, k=len(self.agents))
        
        new_agents = []
        for i in range(0, len(parents), 2):
            child = crossover(parents[i], parents[i+1])
            child = mutate(child)
            new_agents.append(child)
        
        self.agents = sorted(self.agents + new_agents, key=lambda a: np.mean(a.performance_history), reverse=True)[:len(self.agents)]
        print(f"Generation {generation + 1} complete. Best fitness: {np.mean(self.agents[0].performance_history)}")