from backend.base import Base

from node.kernel import AbstractKernel

from util import *


class OMPHost(Base):
    name = 'OpenMP Host'
    short_name = 'omp-host'


    class Kernel(AbstractKernel):
        def __init__(self, name, variables, reads, writes, it_space, body, has_tpe_template=True, num_flop=0):
            super().__init__(name, variables, reads, writes, it_space, body, has_tpe_template, num_flop)

        def launch(self):
            parameters = ', '.join(
                [f.name for f in self.reads if f not in self.writes]
                + [f.name for f in self.writes] +
                  [v.name for v in self.variables])

            return f'{self.fct_name}({parameters});'

        def generate(self):
            parameters = ', '.join(
                [f'const {f.tpe} *__restrict__ {f.name}' for f in self.reads if f not in self.writes]
                + [f'{f.tpe} *__restrict__ {f.name}' for f in self.writes]
                + [f'{v.tpe} {v.name}' for v in self.variables])

            body_in_loops = f'{self.body}'
            for loop in self.it_space:
                body_in_loops = \
                    f'for ({loop[0].tpe} {loop[0]} = {loop[1]}; {loop[0]} < {loop[2]}; ++{loop[0]}) {"{"}{newline}' + \
                    body_in_loops + newline + \
                    f'{"}"}'

            return \
                (f'template<typename tpe>{newline}' if self.has_tpe_template else '') + \
                f'inline void {self.fct_name}({parameters}) {"{"}{newline}' + \
                f'#pragma omp parallel for schedule (static){newline}' + \
                body_in_loops + newline + \
                f'{"}"}{newline}'
