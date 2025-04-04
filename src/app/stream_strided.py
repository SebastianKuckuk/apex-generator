from backend.util_header import UtilHeader

from node.assignment import Assignment
from node.kernel import PseudoKernel
from node.variable import Variable

from util import newline


class StreamStrided:
    name = 'stream-strided'
    name_as_postfix = name.title().replace("-", "")
    group = 'benchmark'
    metric = 'bandwidth'
    additional_parameters = ['stride_read', 'stride_write']
    default_parameters = ['double',  # datatype
                          64 * 1024 * 1024,  # nx
                          1, 1,  # stride_read, stride_write
                          2, 10  # nIt, nWarmup
                          ]
    dimensionality = 1

    @classmethod
    def compose_app(cls, backend):
        iterators = [Variable(f'i{i}', 'size_t') for i in range(cls.dimensionality)]
        sizes = [Variable('nx', 'size_t')]
        it_space = [[i, 0, s] for i, s in zip(iterators, sizes)]

        stride_read, stride_write = Variable('strideRead', 'size_t'), Variable('strideWrite', 'size_t')
        parameters = [stride_read, stride_write]

        ext_it_space = [[i, 0, f'{s} * std::max({stride_read}, {stride_write})'] for i, s in zip(iterators, sizes)]

        src = backend.Field('src', 'tpe', [f'{s} * std::max({stride_read}, {stride_write})' for s in sizes])
        dest = backend.Field('dest', 'tpe', [f'{s} * std::max({stride_read}, {stride_write})' for s in sizes])

        fields = [dest, src]

        if backend == UtilHeader:
            check = f'total += {src.access(iterators)};'

            for loop in ext_it_space:
                check = f'''\
                    for ({loop[0].tpe} {loop[0]} = {loop[1]}; {loop[0]} < {loop[2]}; ++{loop[0]}) {"{"}
                    {check}
                    {"}"}'''

            check = f'''\
                tpe total = 0;
                {check}

                if (total <= 0 || total > {sizes[0]} * nIt)
                    std::cerr << "{cls.name_as_postfix} check failed " << " (expected value between 0+ and " << {sizes[0]} * nIt << " but got " << total << ")" << std::endl;
            '''

            kernels = [
                backend.Kernel(f'init{cls.name_as_postfix}', [*sizes, *parameters], [], fields, ext_it_space,
                               newline.join(str(a) for a in [
                                   Assignment(src.access(iterators), f'(tpe)0'),
                                   Assignment(dest.access(iterators), f'(tpe)0')]),
                               num_flop=0),
                backend.Kernel(f'checkSolution{cls.name_as_postfix}', [*sizes, Variable('nIt', 'size_t'), *parameters], fields, [], [], check, num_flop=0),
                backend.generate_parse_kernel(cls, sizes, parameters)
            ]

        else:
            kernels = [
                backend.Kernel(cls.name, sizes + parameters, [src], [dest], it_space,
                               Assignment(dest.access([i * stride_write for i in iterators]),
                                          src.access([i * stride_read for i in iterators]) + 1), num_flop=1),
                PseudoKernel(f'std::swap({src.d_name}, {dest.d_name});')
            ]

        return backend.Application(backend, cls.name, sizes, parameters, kernels)

    @classmethod
    def sizes_to_bench(cls):
        return sorted([*{*[int(pow(2, 0.1 * i)) for i in range(1_0, 26_0 + 1, 1)]}])

    @classmethod
    def params_to_bench(cls):
        # stride_read, stride_write
        params = [[1, 1]]

        for stride_read in [2, 4, 8, 16]:
            params.append([stride_read, 1])
        for stride_write in [2, 4, 8, 16]:
            params.append([1, stride_write])

        return params
