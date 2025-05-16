from backend.backend import Backend
from backend.base import Base

from node.application import AbstractApplication
from node.assignment import Assignment
from node.variable import Variable

from util import *


class UtilHeader(Base):
    name = 'Util'
    short_name = 'util'

    file_extension = 'h'

    @classmethod
    def generate_check_kernel(cls, app, fields, sizes, parameters, it_space, expected, to_compare):
        check = f'''\
            if ((tpe)({expected}) != {to_compare}) {'{'}
            std::cerr << "{app.name_as_postfix} check failed for element " << {' << ", " << '.join(str(i[0]) for i in it_space)} << " (expected " << {expected} << " but got " << {to_compare} << ")" << std::endl;
            return;
        {'}'}'''
        return cls.Kernel(f'checkSolution{app.name_as_postfix}', [*sizes, Variable('nIt', 'size_t'), *parameters], fields, [], it_space, check)

    @classmethod
    def generate_parse_kernel(cls, app, sizes, parameters):
        init_sizes = newline.join(str(Assignment(s, app.default_parameters[i + 1])) for i, s in enumerate(sizes))
        init_param = newline.join(str(Assignment(p, app.default_parameters[i + 1 + len(sizes)])) for i, p in enumerate(parameters))
        init_n_it = newline.join([str(Assignment('nItWarmUp', app.default_parameters[-2])),
                                  str(Assignment('nIt', app.default_parameters[-1]))])

        parse_sizes = newline.join(f'if (argc > i) {s} = atoi(argv[i]);{newline}++i;' for s in sizes)
        parse_param = newline.join(f'if (argc > i) {p} = atoi(argv[i]);{newline}++i;' for p in parameters) # TODO: support types other than integers
        parse_n_it = newline.join([f'if (argc > i) {n} = atoi(argv[i]);{newline}++i;' for n in ['nItWarmUp', 'nIt']])

        body = f'''\
            // default values
            {init_sizes}
            {init_param}
            {init_n_it}

            // override with command line arguments
            int i = 1;
            if (argc > i) tpeName = argv[i];
            ++i;
            {parse_sizes}
            {parse_param}
            {parse_n_it}'''

        kernel_parameters = [Variable('argc', 'int'), Variable('argv', 'char**'),
                             Variable('tpeName', 'char*&'),
                             *[Variable(s.name, f'{s.tpe}&') for s in sizes],
                             *[Variable(p.name, f'{p.tpe}&') for p in parameters],
                             Variable('nItWarmUp', 'size_t&'), Variable('nIt', 'size_t&')]

        return cls.Kernel(f'parseCLA_{len(sizes)}d', kernel_parameters, [], [], [], body, has_tpe_template=False)


    class Application(AbstractApplication):
        def __init__(self, backend, app, sizes, parameters, kernels):
            super().__init__(backend, app, sizes, parameters, kernels)

        def generate(self):
            return f'#pragma once{newline}{newline}' + \
                (f'#include "../../util.h"{newline}' if Backend.genToApex else f'#include "../../../util.h"{newline}') + \
                2 * newline + \
                self.kernelDecls()
