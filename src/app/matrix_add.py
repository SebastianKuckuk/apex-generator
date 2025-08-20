from backend.util_header import UtilHeader

from node.assignment import Assignment
from node.kernel import PseudoKernel
from node.variable import Variable

from util import *


class MatrixAdd:
    name = 'matrix-add'
    name_as_postfix = name.title().replace("-", "")
    group = 'benchmark'
    metric = 'bandwidth'
    additional_parameters = []
    default_parameters = ['double', 4096, 4096, 2, 10]
    dimensionality = 2

    @classmethod
    def compose_app(cls, backend):
        iterators = [Variable(f'i{i}', 'size_t') for i in range(cls.dimensionality)]
        sizes = [
            Variable('nx', 'size_t'),
            Variable('ny', 'size_t'),
        ]
        it_space = [[i, 0, s] for i, s in zip(iterators, sizes)]

        a = backend.Field('a', 'tpe', sizes)
        b = backend.Field('b', 'tpe', sizes)
        c = backend.Field('c', 'tpe', sizes)

        fields = [a, b, c]

        if backend == UtilHeader:
            init_vals = {
                a: '(tpe)1',
                b: '(tpe)2',
                c: '(tpe)0'
            }

            kernels = [
                backend.Kernel(f'init{cls.name_as_postfix}', sizes, [], fields, it_space, 
                               newline.join(f'{s}' for s in [Assignment(f.access(iterators), v) for f, v in init_vals.items()]), num_flop=0),
                backend.generate_check_kernel(cls, fields, sizes, [], it_space, 1 + 2 * Variable('nIt', 'size_t'), a.access(iterators)),
                backend.generate_parse_kernel(cls, sizes, [])
            ]

        else:
            kernels = [
                backend.Kernel(cls.name, sizes, [a, b], [c], it_space,
                                Assignment(c.access(iterators), a.access(iterators) + b.access(iterators)),
                                num_flop=1),
                PseudoKernel(f'std::swap({c.d_name}, {a.d_name});')
            ]

        return backend.Application(backend, cls.name, sizes, [], kernels)

    @classmethod
    def sizes_to_bench(cls):
        return sorted([*{*[int(pow(2, 0.1 * i)) for i in range(1_0, 15_0 + 1, 1)]}])

    @classmethod
    def params_to_bench(cls):
        return [[]]
