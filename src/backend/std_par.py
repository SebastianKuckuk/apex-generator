import os

from backend.base import Base

from node.application import AbstractApplication
from node.kernel import AbstractKernel

from util import *


class StdPar(Base):
    name = 'std::par'
    short_name = 'std-par'


    class Kernel(AbstractKernel):
        def __init__(self, name, variables, reads, writes, it_space, body, has_tpe_template=True, num_flop=0):
            super().__init__(name, variables, reads, writes, it_space, body, has_tpe_template, num_flop)

        def launch(self):
            parameters = ', '.join(
                [f.name for f in self.reads if f not in self.writes]
                + [f.name for f in self.writes]
                + [v.name for v in self.variables])

            return f'{self.fct_name}({parameters});'

        def generate(self):
            parameters = ', '.join(
                [f'const {f.tpe} * const __restrict__ {f.name}' for f in self.reads if f not in self.writes]
                + [f'{f.tpe} *__restrict__ {f.name}' for f in self.writes]
                + [f'{v.tpe} {v.name}' for v in self.variables])

            it_field = (self.reads + self.writes)[0]

            def remap_it(it, d):
                mod = ' * '.join(f'{s}' for s in it_field.sizes[0: d + 1])
                div = ' * '.join(f'{s}' for s in it_field.sizes[0: d])
                if len(mod) > 0 and d < len(it_field.sizes) - 1:
                    it = f'{it} % ({mod})'
                if len(div) > 0:
                    it = f'({it}) / ({div})'
                return it

            it_cond = ' && '.join((f'{self.it_space[d][0]} >= {self.it_space[d][1]} && ' if 0 != self.it_space[d][1] else '')
                                  + f'{self.it_space[d][0]} < {self.it_space[d][2]}'
                                  for d in range(len(self.it_space)))

            body_w_cond = \
                f'if ({it_cond}) {"{"}{newline}' + \
                f'{self.body}' + newline + \
                f'{"}"}'

            if 1 == len(self.it_space):
                tids = f'const {self.it_space[0][0].tpe} {self.it_space[0][0]} = &{it_field.name}_item - {it_field.name};'
            else:
                tids = \
                    f'const {self.it_space[0][0].tpe} idx = &{it_field.name}_item - {it_field.name};{newline}' + \
                    newline.join(f'const {self.it_space[d][0].tpe} {self.it_space[d][0]} = {remap_it("idx", d)};'
                                 for d in range(len(self.it_space)))

            lambda_fct = \
                f'[=](const {it_field.tpe} &{it_field.name}_item) {"{"} //{newline}' + \
                tids + newline + \
                body_w_cond + newline + \
                f'{"}"});'

            return \
                (f'template<typename tpe>{newline}' if self.has_tpe_template else '') + \
                f'inline void {self.fct_name}({parameters}) {"{"}{newline}' + \
                f'std::for_each(std::execution::par_unseq, {it_field.name}, {it_field.name} + {it_field.totalSize()}, //{newline}' + \
                lambda_fct + newline + \
                f'{"}"}'


    class Application(AbstractApplication):
        def __init__(self, backend, app, sizes, parameters, kernels):
            super().__init__(backend, app, sizes, parameters, kernels)

        def generate(self):
            return f'#include <algorithm>{newline}' + \
                f'#include <execution>{newline}' + \
                newline + \
                f'#include "{self.app}-util.h"{newline}' + \
                2 * newline + \
                self.kernelDecls() + \
                2 * newline + \
                self.mainStart() + \
                newline + \
                self.mainAllocateAndInit() + \
                newline + \
                self.mainMiddle() + \
                newline + \
                self.mainEnd() + \
                2 * newline + \
                self.mainWrapper()
