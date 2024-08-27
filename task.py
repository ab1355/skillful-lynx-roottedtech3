class Task:
    def __init__(self, task_id: str, complexity: float, domain: str, sub_domain: str):
        self.id = task_id
        self.complexity = complexity
        self.domain = domain
        self.sub_domain = sub_domain

    def __repr__(self):
        return f"Task(id={self.id}, complexity={self.complexity:.2f}, domain={self.domain}, sub_domain={self.sub_domain})"