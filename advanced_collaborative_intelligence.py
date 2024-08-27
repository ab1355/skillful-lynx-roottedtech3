# Add this at the end of the file

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
    
    # You can add more detailed analysis or visualization here if needed