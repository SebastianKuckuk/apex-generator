from backend.backend import Backend

from node.application import AbstractApplication
from node.field import AbstractField
from node.kernel import AbstractKernel

from util import *


class OMPTarget(Backend):
    '''Super class to OMPTargetExpl and OMPTargetMM'''

    file_extension = 'cpp'

    @staticmethod
    def synchronize():
        return None


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

            body_in_loops = f'{self.body}'
            for loop in self.it_space:
                body_in_loops = \
                    f'for ({loop[0].tpe} {loop[0]} = {loop[1]}; {loop[0]} < {loop[2]}; ++{loop[0]}) {"{"}{newline}' + \
                    body_in_loops + newline + \
                    f'{"}"}'

            collapse = f' collapse({len(self.it_space)})' if len(self.it_space) > 1 else ''
            return \
                (f'template<typename tpe>{newline}' if self.has_tpe_template else '') + \
                f'inline void {self.fct_name}({parameters}) {"{"}{newline}' + \
                f'#pragma omp target teams distribute parallel for{collapse}{newline}' + \
                body_in_loops + newline + \
                f'{"}"}{newline}'


class OMPTargetExpl(OMPTarget):
    name = 'OpenMP Target Explicit Memory'
    short_name = 'omp-target-expl'


    class Field(AbstractField):
        def __init__(self, name, tpe, sizes):
            super().__init__(name, name, tpe, sizes, has_device_ptr=False)

        def h_declare(self):
            return f'{self.tpe} *{self.name};'

        def h_allocate(self):
            return f'{self.name} = new {self.tpe}[{self.totalSize()}];'

        def h_free(self):
            return f'delete[] {self.name};'

        def copyToDevice(self):
            return f'#pragma omp target enter data map(to : {self.name}[0 : {self.totalSize()}])'

        def copyToHost(self):
            return f'#pragma omp target exit data map(from : {self.name}[0 : {self.totalSize()}])'


    class Application(AbstractApplication):
        def __init__(self, backend, app, sizes, parameters, kernels):
            super().__init__(backend, app, sizes, parameters, kernels)

        def generate(self):
            return f'#include "{self.app}-util.h"{newline}' + \
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


class OMPTargetMM(OMPTarget):
    name = 'OpenMP Target Managed Memory'
    short_name = 'omp-target-mm'


    class Field(AbstractField):
        def __init__(self, name, tpe, sizes):
            super().__init__(name, name, tpe, sizes, has_device_ptr=False)

        def h_declare(self):
            return f'{self.tpe} *{self.name};'

        def h_allocate(self):
            return f'{self.name} = new {self.tpe}[{self.totalSize()}];'

        def h_free(self):
            return f'delete[] {self.name};'

        def copyToDevice(self):
            return None

        def copyToHost(self):
            return None


    class Application(AbstractApplication):
        def __init__(self, backend, app, sizes, parameters, kernels):
            super().__init__(backend, app, sizes, parameters, kernels)

        def generate(self):
            return f'#include "{self.app}-util.h"{newline}' + \
                2 * newline + \
                f'#pragma omp requires unified_shared_memory{newline}' + \
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
