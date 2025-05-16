class Assignment:
    def __init__(self, lhs, rhs, op='='):
        self.lhs = lhs
        self.rhs = rhs
        self.op = op

    def __str__(self):
        return f'{self.lhs} {self.op} {self.rhs};'
