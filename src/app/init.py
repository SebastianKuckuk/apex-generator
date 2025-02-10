from backend.util_header import UtilHeader

from node.assignment import Assignment
from node.variable import Variable


class Init:
    name = 'init'
    name_as_postfix = name.title().replace("-", "")
    group = 'benchmark'
    metric = 'bandwidth'
    default_parameters = ['double', 64 * 1024 * 1024, 2, 10]
    dimensionality = 1

    @classmethod
    def compose_app(cls, backend):
        iterators = [Variable(f'i{i}', 'size_t') for i in range(cls.dimensionality)]
        sizes = [Variable('nx', 'size_t')]
        it_space = [[i, 0, s] for i, s in zip(iterators, sizes)]

        data = backend.Field('data', 'tpe', sizes)

        if backend == UtilHeader:
            kernels = [
                backend.Kernel(f'init{cls.name_as_postfix}', sizes, [], [data], it_space, Assignment(data.access(iterators), '(tpe)0'), num_flop=0),
                backend.generate_check_kernel(cls, [data], sizes, it_space, iterators[0], data.access(iterators)),
                backend.generate_parse_kernel(cls, sizes, [])
            ]

        else:
            kernels = [
                backend.Kernel(cls.name, sizes, [], [data], it_space, Assignment(data.access(iterators), iterators[0]), num_flop=0)
            ]

        return backend.Application(backend, cls.name, sizes, [], kernels)

    @classmethod
    def sizes_to_bench(cls):
        return sorted([*{*[int(pow(2, 0.1 * i)) for i in range(1_0, 30_0 + 1, 1)]}])
