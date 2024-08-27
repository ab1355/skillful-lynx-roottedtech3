import random

class Task:
    def __init__(self, task_id, complexity, domain, sub_domain, task_type):
        self.id = task_id
        self.complexity = complexity
        self.domain = domain
        self.sub_domain = sub_domain
        self.task_type = task_type

    @staticmethod
    def generate_random_task(task_id, complexity_range):
        domains = ["classification", "regression", "clustering", "natural_language_processing", "computer_vision"]
        sub_domains = ["basic", "intermediate", "advanced"]
        task_types = ["prediction", "optimization", "feature_engineering", "model_selection", "hyperparameter_tuning", "data_preprocessing", "ensemble_methods", "transfer_learning"]
        
        domain = random.choice(domains)
        sub_domain = random.choice(sub_domains)
        task_type = random.choice(task_types)
        complexity = random.uniform(*complexity_range)
        
        return Task(task_id, complexity, domain, sub_domain, task_type)

    def __str__(self):
        return f"Task {self.id}: {self.domain}/{self.sub_domain} - {self.task_type} (Complexity: {self.complexity:.2f})"