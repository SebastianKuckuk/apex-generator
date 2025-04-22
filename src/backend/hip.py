from backend.backend import Backend

from node.application import AbstractApplication
from node.field import AbstractField
from node.kernel import AbstractKernel

from util import *


class Hip(Backend):
    '''Super class to HipExpl and HipMM'''

    file_extension = 'hip'

    def_block_sizes = {
        1: [256],
        2: [16, 16],
        3: [16, 4, 4]
    }

    @staticmethod
    def synchronize():
        return f'checkHipError(hipDeviceSynchronize(), true);'


    class Kernel(AbstractKernel):
        def __init__(self, name, variables, reads, writes, it_space, body, has_tpe_template=True, num_flop=0):
            super().__init__(name, variables, reads, writes, it_space, body, has_tpe_template, num_flop)

        def launch(self):
            parameters = ', '.join(
                [f.d_name for f in self.reads if f not in self.writes]
                + [f.d_name for f in self.writes]
                + [v.name for v in self.variables])

            num_dims = len(self.it_space)
            sizes = [s[2] for s in self.it_space]

            block_size = ', '.join(str(s) for s in Hip.def_block_sizes[num_dims])
            num_blocks = ', '.join(f'ceilingDivide({s}, {Hip.def_block_sizes[num_dims][d]})' for d, s in enumerate(sizes))

            return f'{self.fct_name}<<<dim3({num_blocks}), dim3({block_size})>>>({parameters});'

        def generate(self):
            parameters = ', '.join(
                [f'const {f.tpe} * const __restrict__ {f.name}' for f in self.reads if f not in self.writes]
                + [f'{f.tpe} *__restrict__ {f.name}' for f in self.writes]
                + [f'{v.tpe} {v.name}' for v in self.variables])

            tids = newline.join(f'const {self.it_space[d][0].tpe} {self.it_space[d][0]} = blockIdx.{s} * blockDim.{s} + threadIdx.{s};'
                                for d, s in enumerate(dim_to_char[0: len(self.it_space)]))

            it_cond = ' && '.join((f'{self.it_space[d][0]} >= {self.it_space[d][1]} && ' if 0 != self.it_space[d][1] else '')
                                  + f'{self.it_space[d][0]} < {self.it_space[d][2]}'
                                  for d in range(len(self.it_space)))

            return \
                (f'template<typename tpe>{newline}' if self.has_tpe_template else '') + \
                f'__global__ void {self.fct_name}({parameters}) {"{"}{newline}' + \
                tids + newline + \
                newline + \
                f'if ({it_cond}) {"{"}{newline}' + \
                f'{self.body}' + newline + \
                f'{"}"}{newline}' + \
                f'{"}"}{newline}'


    class Application(AbstractApplication):
        def __init__(self, backend, app, sizes, parameters, kernels):
            super().__init__(backend, app, sizes, parameters, kernels)

        def generate(self):
            block_size = ', '.join(str(s) for s in Hip.def_block_sizes[len(self.sizes)])
            num_blocks = ', '.join(f'ceilingDivide({s}, blockSize.{d})' for s, d in zip(self.sizes, dim_to_char[0: len(self.sizes)]))

            return f'#include "{self.app}-util.h"{newline}' + \
                newline + \
                (f'#include "../../hip-util.h"{newline}' if Backend.genToApex else f'#include "../../../hip-util.h"{newline}') + \
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


class HipExpl(Hip):
    name = 'HIP Explicit Memory'
    short_name = 'hip-expl'


    class Field(AbstractField):
        def __init__(self, name, tpe, sizes):
            super().__init__(name, f'd_{name}', tpe, sizes, has_device_ptr=True)

        def h_declare(self):
            return f'{self.tpe} *{self.name};'

        def d_declare(self):
            return f'{self.tpe} *{self.d_name};'

        def h_allocate(self):
            return f'checkHipError(hipHostMalloc((void **) &{self.name}, sizeof({self.tpe}) * {self.totalSize()}));'

        def d_allocate(self):
            return f'checkHipError(hipMalloc((void **) &{self.d_name}, sizeof({self.tpe}) * {self.totalSize()}));'

        def h_free(self):
            return f'checkHipError(hipHostFree({self.name}));'

        def d_free(self):
            return f'checkHipError(hipFree({self.d_name}));'

        def copyToDevice(self):
            return f'checkHipError(hipMemcpy({self.d_name}, {self.name}, sizeof({self.tpe}) * {self.totalSize()}, hipMemcpyHostToDevice));'

        def copyToHost(self):
            return f'checkHipError(hipMemcpy({self.name}, {self.d_name}, sizeof({self.tpe}) * {self.totalSize()}, hipMemcpyDeviceToHost));'


class HipMM(Hip):
    name = 'HIP Managed Memory'
    short_name = 'hip-mm'


    class Field(AbstractField):
        def __init__(self, name, tpe, sizes):
            super().__init__(name, name, tpe, sizes, has_device_ptr=False)

        def h_declare(self):
            return f'{self.tpe} *{self.name};'

        def h_allocate(self):
            return f'checkHipError(hipMallocManaged((void **) &{self.name}, sizeof({self.tpe}) * {self.totalSize()}));'

        def h_free(self):
            return f'checkHipError(hipFree({self.name}));'

        def copyToDevice(self):
            return f'checkHipError(hipMemPrefetchAsync({self.name}, sizeof({self.tpe}) * {self.totalSize()}, 0));'

        def copyToHost(self):
            return f'checkHipError(hipMemPrefetchAsync({self.name}, sizeof({self.tpe}) * {self.totalSize()}, hipCpuDeviceId));'
