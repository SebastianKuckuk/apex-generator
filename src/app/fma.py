from backend.util_header import UtilHeader

from node.assignment import Assignment
from node.variable import Variable

from util import *


class FMA:
    name = 'fma'
    name_as_postfix = name.title().replace("-", "")
    group = 'benchmark'
    metric = 'flops'
    default_type = 'float'
    default_parameters = ['double', 64, 1, 2]
    # default_parameters = ['double', 1024 * 1024, 1, 2]
    num_rep = 1024 * 1024
    dimensionality = 1

    @classmethod
    def compose_app(cls, backend):
        iterators = [Variable(f'i{i}', 'size_t') for i in range(cls.dimensionality)]
        sizes = [Variable('nx', 'size_t')]
        it_space = [[i, 0, s] for i, s in zip(iterators, sizes)]

        data = backend.Field('data', 'tpe', sizes)

        if backend == UtilHeader:
            kernels = [
                backend.Kernel(f'init{cls.name_as_postfix}', sizes, [], [data], it_space, Assignment(data.access(iterators), '(tpe)1'), num_flop=0),
                backend.generate_check_kernel(cls, [data], sizes, [], it_space, '(tpe)1', data.access(iterators)),
                backend.generate_parse_kernel(cls, sizes, [])
            ]

        else:
            kernels = [
                backend.Kernel(cls.name, sizes, [], [data], it_space,
                                f'tpe a = (tpe)0.5, b = (tpe)1;{newline}' +
                                f'// dummy op to prevent compiler from solving loop analytically{newline}' +
                                f'if (1 == nx) {"{"}{newline}' +
                                f'auto tmp = b; b = a; a = tmp;{newline}' +
                                f'{"}"}{newline}' +
                                newline +
                                f'{Assignment("tpe acc", iterators[0])}{newline}' +
                                newline +
                                f'for (auto r = 0; r < {cls.num_rep}; ++r){newline}' +
                                f'acc = a * acc + b;{newline}' +
                                newline +
                                f'// dummy check to prevent compiler from eliminating loop{newline}' +
                                f'if ((tpe)0 == acc)' +
                                f'    {Assignment(data.access(iterators), "acc")}',
                                num_flop=2 * cls.num_rep)
            ]

        return backend.Application(backend, cls.name, sizes, [], kernels)

    @classmethod
    def sizes_to_bench(cls):
        return sorted([*{*[int(pow(2, 0.1 * i)) for i in range(1_0, 25_0 + 1, 1)]}])
