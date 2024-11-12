class Assignment:
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f'{self.lhs} = {self.rhs};'
