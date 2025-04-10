from backend.util_header import UtilHeader

from node.assignment import Assignment
from node.kernel import PseudoKernel
from node.variable import Variable

from util import *


class SquareRoot:
    name = 'square-root'
    name_as_postfix = name.title().replace("-", "")
    group = 'benchmark'
    metric = 'compute'
    additional_parameters = []
    default_parameters = ['double', 64, 1, 2]
    # default_parameters = ['double', 1024 * 1024, 1, 2]
    num_rep = 64 * 1024
    dimensionality = 1

    @classmethod
    def compose_app(cls, backend):
        iterators = [Variable(f'i{i}', 'size_t') for i in range(cls.dimensionality)]
        sizes = [Variable('nx', 'size_t')]
        it_space = [[i, 0, s] for i, s in zip(iterators, sizes)]

        src = backend.Field('src', 'tpe', sizes)
        dest = backend.Field('dest', 'tpe', sizes)

        fields = [dest, src]

        if backend == UtilHeader:
            kernels = [
                backend.Kernel(f'init{cls.name_as_postfix}', sizes, [], fields, it_space,
                                newline.join(str(a) for a in [
                                    Assignment(src.access(iterators), f'(tpe)1'),
                                    Assignment(dest.access(iterators), f'(tpe)0')]),
                                num_flop=0),
                backend.generate_check_kernel(cls, fields, sizes, [], it_space, '(tpe)1', src.access(iterators)),
                backend.generate_parse_kernel(cls, sizes, [])
            ]

        else:
            kernels = [
                backend.Kernel(cls.name, sizes, [src], [dest], it_space,
                                f'{Assignment("tpe acc", src.access(iterators))}{newline}' +
                                newline +
                                f'for (auto r = 0; r < {cls.num_rep}; ++r){newline}' +
                                f'acc = sqrt(acc);{newline}' +
                                newline +
                                f'{Assignment(dest.access(iterators), "acc")}',
                                num_flop=cls.num_rep),
                PseudoKernel(f'std::swap({src.d_name}, {dest.d_name});')
            ]

        return backend.Application(backend, cls.name, sizes, [], kernels)

    @classmethod
    def sizes_to_bench(cls):
        return sorted([*{*[int(pow(2, 0.1 * i)) for i in range(1_0, 25_0 + 1, 1)]}])

    @classmethod
    def params_to_bench(cls):
        return [[]]
