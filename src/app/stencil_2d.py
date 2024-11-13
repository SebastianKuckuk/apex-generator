from backend.util_header import UtilHeader

from node.assignment import Assignment
from node.kernel import PseudoKernel
from node.variable import Variable


class Stencil2D:
    name = 'stencil-2d'
    name_as_postfix = name.title().replace("-", "")
    group = 'benchmark'
    metric = 'bandwidth'
    default_parameters = ['double', 4096, 4096, 2, 10]
    dimensionality = 2

    @classmethod
    def compose_app(cls, backend):
        iterators = [Variable(f'i{i}', 'size_t') for i in range(cls.dimensionality)]
        sizes = [
            Variable('nx', 'size_t'),
            Variable('ny', 'size_t'),
        ]
        it_space = [[i, 1, s - 1] for i, s in zip(iterators, sizes)]
        full_it_space = [[i, 0, s] for i, s in zip(iterators, sizes)]

        u = backend.Field('u', 'tpe', sizes)
        uNew = backend.Field('uNew', 'tpe', sizes)

        fields = [u, uNew]

        if backend == UtilHeader:
            on_boundary = ' || '.join(f'0 == {iterators[d]} || {sizes[d]} - 1 == {iterators[d]}' for d in range(cls.dimensionality))
            init = f'''\
                if ({on_boundary}) {'{'}
                    {Assignment(u.access(iterators), f'(tpe)0')}
                    {Assignment(uNew.access(iterators), f'(tpe)0')}
                {'}'} else {'{'}
                    {Assignment(u.access(iterators), f'(tpe)1')}
                    {Assignment(uNew.access(iterators), f'(tpe)1')}
                {'}'}
                '''

            check = f'''\
                    const tpe localRes = {2 * cls.dimensionality * u.access(iterators) 
                                          - (u.access([iterators[0] - 1, iterators[1]])
                                             + u.access([iterators[0] + 1, iterators[1]])
                                             + u.access([iterators[0], iterators[1] - 1])
                                             + u.access([iterators[0], iterators[1] + 1]))};
                    res += localRes * localRes;'''

            for loop in it_space:
                check = f'''\
                    for ({loop[0].tpe} {loop[0]} = {loop[1]}; {loop[0]} < {loop[2]}; ++{loop[0]}) {"{"}
                    {check}
                    {"}"}'''

            check = f'''\
                tpe res = 0;
                {check}

                res = sqrt(res);

                std::cout << "  Final residual is " << res << std::endl;
            '''
            kernels = [
                backend.Kernel(f'init{cls.name_as_postfix}', sizes, [], fields, full_it_space, init, num_flop=0),
                backend.Kernel(f'checkSolution{cls.name_as_postfix}', [*sizes, Variable('nIt', 'size_t')], fields, [], [], check, num_flop=0),
            ]

        else:
            kernels = [
                backend.Kernel(cls.name, sizes, [u], [uNew], it_space,
                                Assignment(uNew.access(iterators), 0.25 * (
                                    u.access([iterators[0] - 1, iterators[1]])
                                    + u.access([iterators[0] + 1, iterators[1]])
                                    + u.access([iterators[0], iterators[1] - 1])
                                    + u.access([iterators[0], iterators[1] + 1]))),
                                num_flop=1 + 3 * 2),  # 1 MUL and 3 FMA
                PseudoKernel(f'std::swap({u.d_name}, {uNew.d_name});')
            ]

        return backend.Application(backend, cls.name, sizes, kernels)

    @classmethod
    def sizes_to_bench(cls):
        return sorted([*{*[int(pow(2, 0.1 * i)) for i in range(1_0, 15_0 + 1, 1)]}])
