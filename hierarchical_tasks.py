class ComplexTask(Task):
    def __init__(self, name, domain, sub_domain, task_type, complexity):
        super().__init__(name, domain, sub_domain, task_type, complexity)
        self.subtasks = []

    def decompose(self):
        # Logic to break down the task into subtasks
        subtask_complexity = self.complexity / 3
        for i in range(3):
            subtask = Task(f"{self.name}_subtask_{i}", self.domain, self.sub_domain, self.task_type, subtask_complexity)
            self.subtasks.append(subtask)

# Add this to the MultiAgentSystem class
def handle_complex_task(self, complex_task):
    complex_task.decompose()
    for subtask in complex_task.subtasks:
        self.assign_task(subtask)