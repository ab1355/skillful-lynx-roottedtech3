import networkx as nx
import random

class NetworkAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.connections = set()

    def add_connection(self, other_agent):
        self.connections.add(other_agent)
        other_agent.connections.add(self)

    def remove_connection(self, other_agent):
        self.connections.remove(other_agent)
        other_agent.connections.remove(self)

# Add this to the MultiAgentSystem class
def initialize_network(self):
    self.network = nx.Graph()
    for agent in self.agents:
        self.network.add_node(agent)

def update_network(self):
    for agent in self.agents:
        if len(agent.connections) < 3:  # Ensure minimum connections
            potential_connections = set(self.agents) - set(agent.connections) - {agent}
            if potential_connections:
                new_connection = random.choice(list(potential_connections))
                agent.add_connection(new_connection)
                self.network.add_edge(agent, new_connection)
        elif random.random() < 0.1:  # 10% chance to remove a connection
            connection_to_remove = random.choice(list(agent.connections))
            agent.remove_connection(connection_to_remove)
            self.network.remove_edge(agent, connection_to_remove)

    # Visualize the network (requires matplotlib)
    nx.draw(self.network, with_labels=True)
    plt.savefig("agent_network.png")
    plt.close()