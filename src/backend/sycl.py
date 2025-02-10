from backend.backend import Backend

from node.application import AbstractApplication
from node.field import AbstractField
from node.kernel import AbstractKernel

from util import *


class Sycl(Backend):
    '''Super class to SyclBuffer, SyclExpl and SyclMM'''

    file_extension = 'cpp'

    def_block_sizes = {
        1: [256],
        2: [16, 16],
        3: [16, 4, 4]
    }

    @staticmethod
    def synchronize():
        return f'q.wait();'


class SyclBuffer(Sycl):
    name = 'SYCL Buffer'
    short_name = 'sycl-buffer'


    class Field(AbstractField):
        def __init__(self, name, tpe, sizes):
            super().__init__(name, f'b_{name}', tpe, sizes, has_device_ptr=False)

        def h_declare(self):
            return f'{self.tpe} *{self.name};'

        def d_declare(self):
            return f'{self.tpe} *{self.d_name};'

        def h_allocate(self):
            return f'{self.name} = new {self.tpe}[{self.totalSize()}];'

        def d_allocate(self):
            return f'sycl::buffer {self.d_name}({self.name}, sycl::range({self.totalSize()}));'  # TODO: nD

        def h_free(self):
            return f'delete[] {self.name};'

        def d_free(self):
            return None

        def copyToDevice(self):
            return None

        def copyToHost(self):
            return None


    class Kernel(AbstractKernel):
        def __init__(self, name, variables, reads, writes, it_space, body, has_tpe_template=True, num_flop=0):
            super().__init__(name, variables, reads, writes, it_space, body, has_tpe_template, num_flop)

        def launch(self):
            parameters = ', '.join(
                [f.d_name for f in self.reads if f not in self.writes]
                + [f.d_name for f in self.writes]
                + [v.name for v in self.variables])

            return f'{self.fct_name}(q, {parameters});'

        def generate(self):
            parameters = ', '.join(
                ['sycl::queue &q']
                + [f'sycl::buffer<{f.tpe}> &{f.d_name}' for f in self.reads if f not in self.writes]
                + [f'sycl::buffer<{f.tpe}> &{f.d_name}' for f in self.writes]
                + [f'{v.tpe} {v.name}' for v in self.variables])

            it_cond = ' && '.join((f'{self.it_space[d][0]} >= {self.it_space[d][1]} && ' if 0 != self.it_space[d][1] else '')
                                  + f'{self.it_space[d][0]} < {self.it_space[d][2]}'
                                  for d in range(len(self.it_space)))

            def access_mode(f):
                if f in self.reads and f in self.writes:
                    return 'sycl::read_write'
                if f in self.reads:
                    return 'sycl::read_only'
                return 'sycl::write_only'
            access_buffers = newline.join(f'auto {f.name} = {f.d_name}.get_access(h, {access_mode(f)});' for f in self.reads + self.writes)

            dims = len(self.it_space)
            if 1 == dims:
                parallel_for = \
                    f'h.parallel_for({self.it_space[0][2]}, [=](auto {self.it_space[0][0]}) {"{"}{newline}' + \
                    f'if ({it_cond}) {"{"}{newline}' + \
                    f'{self.body}' + newline + \
                    f'{"}"}{newline}' + \
                    f'{"}"});'
            else:
                sizes = [i[2] for i in self.it_space]
                block_size = Sycl.def_block_sizes[len(sizes)]
                threads_total = ', '.join(reversed(list(f'ceilToMultipleOf({s}, {d})' for s, d in zip(sizes, block_size))))  # dim_to_char[0 : len(self.sizes)]))
                block_size = ', '.join(f'{b}' for b in reversed(block_size))

                its = newline.join(f'const auto {self.it_space[d][0]} = item.get_global_id({dims - d - 1});' for d in range(dims))

                # TODO alternative h.parallel_for(sycl::range<2>(ny, nx), [=](sycl::item<2> item) {

                parallel_for = \
                    f'h.parallel_for(sycl::nd_range<{dims}>(sycl::range<{dims}>({threads_total}), sycl::range<{dims}>({block_size})), [=](sycl::nd_item<{dims}> item) {"{"}{newline}' + \
                    its + newline + \
                    newline + \
                    f'if ({it_cond}) {"{"}{newline}' + \
                    f'{self.body}' + newline + \
                    f'{"}"}{newline}' + \
                    f'{"}"});'

            queue_op = \
                f'q.submit([&](sycl::handler &h) {"{"}{newline}' + \
                access_buffers + newline + \
                newline + \
                parallel_for + newline + \
                f'{"}"});'

            return \
                (f'template<typename tpe>{newline}' if self.has_tpe_template else '') + \
                f'inline void {self.fct_name}({parameters}) {"{"}{newline}' + \
                queue_op + newline + \
                f'{"}"}{newline}'


    class Application(AbstractApplication):
        def __init__(self, backend, app, sizes, parameters, kernels):
            super().__init__(backend, app, sizes, parameters, kernels)

        def generate(self):
            return f'#include "{self.app}-util.h"{newline}' + \
                newline + \
                f'#include "../../sycl-util.h"{newline}' + \
                2 * newline + \
                self.kernelDecls() + \
                2 * newline + \
                self.mainStart() + \
                newline + \
                f'sycl::queue q(sycl::property::queue::in_order{"{}"}); // in-order queue to remove need for waits after each kernel{newline}' + \
                newline + \
                self.mainAllocateAndInit() + \
                newline + \
                f'{"{"}{newline}' + \
                newline.join(f.d_allocate() for f in self.fields) + newline + \
                newline + \
                self.mainMiddle() + \
                f'{"}"} // implicit D-H copy of destroyed buffers{newline}' + \
                newline + \
                self.mainEnd() + \
                2 * newline + \
                self.mainWrapper()


class SyclNoBuffer(Sycl):
    '''Super class to SyclExpl and SyclMM'''


    class Kernel(AbstractKernel):
        def __init__(self, name, variables, reads, writes, it_space, body, has_tpe_template=True, num_flop=0):
            super().__init__(name, variables, reads, writes, it_space, body, has_tpe_template, num_flop)

        def launch(self):
            parameters = ', '.join(
                [f.d_name for f in self.reads if f not in self.writes]
                + [f.d_name for f in self.writes]
                + [v.name for v in self.variables])
            return f'{self.fct_name}(q, {parameters});'

        def generate(self):
            parameters = ', '.join(
                ['sycl::queue &q']
                + [f'const {f.tpe} * const __restrict__ {f.name}' for f in self.reads if f not in self.writes]
                + [f'{f.tpe} *__restrict__ {f.name}' for f in self.writes]
                + [f'{v.tpe} {v.name}' for v in self.variables])

            it_cond = ' && '.join((f'{self.it_space[d][0]} >= {self.it_space[d][1]} && ' if 0 != self.it_space[d][1] else '')
                                  + f'{self.it_space[d][0]} < {self.it_space[d][2]}'
                                  for d in range(len(self.it_space)))

            dims = len(self.it_space)
            if 1 == dims:
                parallel_for = \
                    f'h.parallel_for({self.it_space[0][2]}, [=](auto {self.it_space[0][0]}) {"{"}{newline}' + \
                    f'if ({it_cond}) {"{"}{newline}' + \
                    f'{self.body}' + newline + \
                    f'{"}"}{newline}' + \
                    f'{"}"});'
            else:
                sizes = [i[2] for i in self.it_space]
                block_size = Sycl.def_block_sizes[len(sizes)]
                threads_total = ', '.join(reversed(list(f'ceilToMultipleOf({s}, {d})' for s, d in zip(sizes, block_size))))  # dim_to_char[0 : len(self.sizes)]))
                block_size = ', '.join(f'{b}' for b in reversed(block_size))

                its = newline.join(f'const auto {self.it_space[d][0]} = item.get_global_id({dims - d - 1});' for d in range(dims))

                # TODO alternative h.parallel_for(sycl::range<2>(ny, nx), [=](sycl::item<2> item) {

                parallel_for = \
                    f'h.parallel_for(sycl::nd_range<{dims}>(sycl::range<{dims}>({threads_total}), sycl::range<{dims}>({block_size})), [=](sycl::nd_item<{dims}> item) {"{"}{newline}' + \
                    its + newline + \
                    newline + \
                    f'if ({it_cond}) {"{"}{newline}' + \
                    f'{self.body}' + newline + \
                    f'{"}"}{newline}' + \
                    f'{"}"});'

            queue_op = \
                f'q.submit([&](sycl::handler &h) {"{"}{newline}' + \
                parallel_for + newline + \
                f'{"}"});'

            return \
                (f'template<typename tpe>{newline}' if self.has_tpe_template else '') + \
                f'inline void {self.fct_name}({parameters}) {"{"}{newline}' + \
                queue_op + newline + \
                f'{"}"}{newline}'


    class Application(AbstractApplication):
        def __init__(self, backend, app, sizes, parameters, kernels):
            super().__init__(backend, app, sizes, parameters, kernels)

        def generate(self):
            return f'#include "{self.app}-util.h"{newline}' + \
                newline + \
                f'#include "../../sycl-util.h"{newline}' + \
                2 * newline + \
                self.kernelDecls() + \
                2 * newline + \
                self.mainStart() + \
                newline + \
                f'sycl::queue q(sycl::property::queue::in_order{"{}"}); // in-order queue to remove need for waits after each kernel{newline}' + \
                newline + \
                self.mainAllocateAndInit() + \
                newline + \
                self.mainMiddle() + \
                newline + \
                self.mainEnd() + \
                2 * newline + \
                self.mainWrapper()


class SyclExpl(SyclNoBuffer):
    name = 'SYCL Explicit Memory'
    short_name = 'sycl-expl'


    class Field(AbstractField):
        def __init__(self, name, tpe, sizes):
            super().__init__(name, f'd_{name}', tpe, sizes, has_device_ptr=True)

        def h_declare(self):
            return f'{self.tpe} *{self.name};'

        def d_declare(self):
            return f'{self.tpe} *{self.d_name};'

        def h_allocate(self):
            return f'{self.name} = sycl::malloc_host<{self.tpe}>({self.totalSize()}, q);'

        def d_allocate(self):
            return f'{self.d_name} = sycl::malloc_device<{self.tpe}>({self.totalSize()}, q);'

        def h_free(self):
            return f'sycl::free({self.name}, q);'

        def d_free(self):
            return f'sycl::free({self.d_name}, q);'

        def copyToDevice(self):
            return f'q.memcpy({self.d_name}, {self.name}, sizeof({self.tpe}) * {self.totalSize()}); q.wait();'

        def copyToHost(self):
            return f'q.memcpy({self.name}, {self.d_name}, sizeof({self.tpe}) * {self.totalSize()}); q.wait();'


class SyclMM(SyclNoBuffer):
    name = 'SYCL Managed Memory'
    short_name = 'sycl-mm'


    class Field(AbstractField):
        def __init__(self, name, tpe, sizes):
            super().__init__(name, name, tpe, sizes, has_device_ptr=False)

        def h_declare(self):
            return f'{self.tpe} *{self.name};'

        def h_allocate(self):
            return f'{self.name} = sycl::malloc_shared<{self.tpe}>({self.totalSize()}, q);'

        def h_free(self):
            return f'sycl::free({self.name}, q);'

        def copyToDevice(self):
            return None  # TODO prefetch f'q.memcpy({self.d_name}, {self.name}, sizeof({self.tpe}) * {self.totalSize()});'

        def copyToHost(self):
            return None  # TODO prefetch f'q.memcpy({self.name}, {self.d_name}, sizeof({self.tpe}) * {self.totalSize()});'
