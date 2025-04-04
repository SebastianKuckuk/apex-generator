from backend.util_header import UtilHeader

from node.assignment import Assignment
from node.variable import Variable

from util import *


class FMAStrided:
    name = 'fma-strided'
    name_as_postfix = name.title().replace("-", "")
    group = 'benchmark'
    metric = 'flops'
    default_type = 'float'
    additional_parameters = ['stride']
    default_parameters = ['double',  # datatype
                          64,  # nx
                          1,  # stride
                          0, 4]  # nIt, nWarmup
    num_rep = 64 * 1024
    dimensionality = 1

    @classmethod
    def compose_app(cls, backend):
        iterators = [Variable(f'i{i}', 'size_t') for i in range(cls.dimensionality)]
        sizes = [Variable('nx', 'size_t')]
        it_space = [[i, 0, s] for i, s in zip(iterators, sizes)]

        stride = Variable('stride', 'size_t')
        parameters = [stride]

        it_space_kernel = [[i, 0, s * stride] for i, s in zip(iterators, sizes)]

        data = backend.Field('data', 'tpe', sizes)

        if backend == UtilHeader:
            kernels = [
                backend.Kernel(f'init{cls.name_as_postfix}', [*sizes, *parameters], [], [data], it_space, Assignment(data.access(iterators), '(tpe)1'), num_flop=0),
                backend.generate_check_kernel(cls, [data], sizes, parameters, it_space, '(tpe)1', data.access(iterators)),
                backend.generate_parse_kernel(cls, sizes, parameters)
            ]

        else:
            kernels = [
                backend.Kernel(cls.name, sizes + parameters, [], [data], it_space_kernel,
                               f'tpe a = (tpe)0.5, b = (tpe)1;{newline}' +
                               f'// dummy op to prevent compiler from solving loop analytically{newline}' +
                               f'if (1 == nx) {"{"}{newline}' +
                               f'auto tmp = b; b = a; a = tmp;{newline}' +
                               f'{"}"}{newline}' +
                               newline +
                               f'{Assignment("tpe acc", iterators[0])}{newline}' +
                               newline +
                               f'if (0 == {iterators[0]} % {stride}){newline}' +
                               f'for (auto r = 0; r < {cls.num_rep}; ++r){newline}' +
                               f'acc = a * acc + b;{newline}' +
                               newline +
                               f'// dummy check to prevent compiler from eliminating loop{newline}' +
                               f'if ((tpe)0 == acc)' +
                               f'    {Assignment(data.access([i / stride for i in iterators]), "acc")}',
                               num_flop=2 * cls.num_rep)
            ]

        return backend.Application(backend, cls.name, sizes, parameters, kernels)

    @classmethod
    def sizes_to_bench(cls):
        return sorted([*{*[int(pow(2, 0.1 * i)) for i in range(1_0, 20_0 + 1, 1)]}])

    @classmethod
    def params_to_bench(cls):
        # stride
        return [[1], [2], [4], [8], [16], [32]]
