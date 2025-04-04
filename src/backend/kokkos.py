import os
import sympy as sp

from backend.backend import Backend

from node.application import AbstractApplication
from node.field import AbstractField
from node.kernel import AbstractKernel

from util import *


class Kokkos(Backend):
    '''Super class to KokkosSerial, KokkosOMPHost and KokkosCuda'''

    file_extension = 'cpp'

    @classmethod
    def default_code_file(cls, app):
        return f'{app.name}-kokkos.{cls.file_extension}'

    @staticmethod
    def synchronize():
        return f'Kokkos::fence();'


    class Field(AbstractField):
        def __init__(self, name, tpe, sizes):
            super().__init__(name, name, tpe, sizes, has_device_ptr=True)

            self.h_name = f'h_{name}'

        def h_declare(self):
            return None

        def d_declare(self):
            return None

        def h_allocate(self):
            return f'auto {self.h_name} = Kokkos::create_mirror_view({self.d_name});'

        def d_allocate(self):
            return f'Kokkos::View<{self.tpe} {"*" * len(self.sizes)}> {self.d_name}("{self.d_name}", {comma.join(str(s) for s in self.sizes)});'

        def h_free(self):
            return None

        def d_free(self):
            return None

        def copyToDevice(self):
            return f'Kokkos::deep_copy({self.d_name}, {self.h_name});'

        def copyToHost(self):
            return f'Kokkos::deep_copy({self.h_name}, {self.d_name});'

        def access(self, iterators):
            return Kokkos.FieldAccess(self, iterators, True)

        def d_access(self, iterators):
            return Kokkos.FieldAccess(self, iterators, False)


    class FieldAccess(sp.Symbol):
        def __new__(cls, field, indices, host, description=''):
            obj = sp.Symbol.__new__(cls, f'{field.name if host else field.d_name}({comma.join(str(i) for i in indices)})')
            obj.description = description
            return obj

        def __init__(self, field, index, host):
            self.field = field
            self.index = index
            self.host = host


    class Kernel(AbstractKernel):
        def __init__(self, name, variables, reads, writes, it_space, body, has_tpe_template=True, num_flop=0):
            super().__init__(name, variables, reads, writes, it_space, body, has_tpe_template, num_flop)

        def launch(self):
            parameters = ', '.join(
                [f.d_name for f in self.reads if f not in self.writes]
                + [f.d_name for f in self.writes]
                + [v.name for v in self.variables])

            return f'{self.fct_name}({parameters});'

        def generate(self):
            parameters = ', '.join(
                [f'const Kokkos::View<{f.tpe} {"*" * len(f.sizes)}> &{f.name}' for f in self.reads if f not in self.writes]
                + [f'Kokkos::View<{f.tpe} {"*" * len(f.sizes)}> &{f.name}' for f in self.writes]
                + [f'{v.tpe} {v.name}' for v in self.variables])

            dims = len(self.it_space)
            if 1 == dims:
                for_bounds = f'Kokkos::RangePolicy<>({self.it_space[0][1]}, {self.it_space[0][2]})'
            else:
                for_bounds = f'Kokkos::MDRangePolicy<Kokkos::Rank<{dims}>, Kokkos::Schedule<Kokkos::Static> >(' + \
                    '{' + ', '.join(f'{self.it_space[d][1]}' for d in range(dims)) + '}, ' + \
                    '{' + ', '.join(f'{self.it_space[d][2]}' for d in range(dims)) + '})'

            its = ', '.join(f'const {i[0].decl()}' for i in self.it_space)

            parallel_for = \
                f'Kokkos::parallel_for({for_bounds}, //{newline}' + \
                f'KOKKOS_LAMBDA({its}) {"{"} //{newline}' + \
                f'{self.body} \\' + newline + \
                f'{"}"});'

            return \
                (f'template<typename tpe>{newline}' if self.has_tpe_template else '') + \
                f'inline void {self.fct_name}({parameters}) {"{"}{newline}' + \
                parallel_for + newline + \
                f'{"}"}{newline}'


    class Application(AbstractApplication):
        def __init__(self, backend, app, sizes, parameters, kernels):
            super().__init__(backend, app, sizes, parameters, kernels)

        def generate(self):
            size_list = ', '.join(f'{s}' for s in self.sizes)
            field_list = ', '.join(f'{f.h_name}.data()' for f in self.fields)

            if len(self.parameters) > 0:
                param_list = ', ' + ', '.join(f'{p}' for p in self.parameters)
            else:
                param_list = ''

            return f'#include <Kokkos_Core.hpp>{newline}' + \
                newline + \
                f'#include "{self.app}-util.h"{newline}' + \
                2 * newline + \
                self.kernelDecls() + \
                2 * newline + \
                self.mainStart() + \
                newline + \
                f'int c = 1;{newline}' + \
                f'Kokkos::initialize(c, argv);{newline}' + \
                f'{"{"}{newline}' + \
                newline.join(f.d_allocate() for f in self.fields if f.has_device_ptr) + newline + \
                newline + \
                newline.join(f.h_allocate() for f in self.fields) + newline + \
                newline + \
                f'// init{newline}' + \
                f'init{self.app.title().replace("-", "")}({field_list}, {size_list}{param_list});{newline}' + \
                self.toDeviceCopies() + \
                newline + \
                self.mainMiddle() + \
                newline + \
                self.toHostCopies() + \
                f'// check solution{newline}' + \
                f'checkSolution{self.app.title().replace("-", "")}({field_list}, {size_list}, nIt + nItWarmUp{param_list});{newline}' + \
                f'{"}"}{newline}' + \
                f'Kokkos::finalize();{newline}' + \
                newline + \
                f'return 0;{newline}' + \
                f'{"}"}{newline}' + \
                2 * newline + \
                self.mainWrapper()


class KokkosSerial(Kokkos):
    name = 'Kokkos Host Serial'
    short_name = 'kokkos-serial'


class KokkosOMPHost(Kokkos):
    name = 'Kokkos Host OpenMP'
    short_name = 'kokkos-omp-host'


class KokkosCuda(Kokkos):
    name = 'Kokkos CUDA'
    short_name = 'kokkos-cuda'
