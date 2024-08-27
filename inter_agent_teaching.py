import numpy as np

def teach(teacher, student, knowledge_transfer_rate=0.2):
    student.knowledge = (1 - knowledge_transfer_rate) * student.knowledge + knowledge_transfer_rate * teacher.knowledge

# Add this to the MultiAgentSystem class
def inter_agent_teaching_round(self):
    agents_sorted = sorted(self.agents, key=lambda a: np.mean(a.performance_history), reverse=True)
    for i in range(len(agents_sorted) // 2):
        teacher = agents_sorted[i]
        student = agents_sorted[-(i+1)]
        teach(teacher, student)
        print(f"Agent {teacher.id} taught Agent {student.id}")