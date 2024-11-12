class AbstractKernel:
    def __init__(self, name, variables, reads, writes, it_space, body, num_flop):
        self.name = name
        self.fct_name = self.name.replace('-', '')
        self.variables = variables
        self.reads = reads
        self.writes = writes
        self.it_space = it_space
        self.body = body
        self.num_flop = num_flop

    def launch(self):
        pass

    def generate(self):
        pass


class PseudoKernel:
    def __init__(self, code):
        self.code = code
        self.reads = []
        self.writes = []
        self.num_flop = 0

    def launch(self):
        return f'{self.code}'
