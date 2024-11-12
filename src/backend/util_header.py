from backend.base import Base

from node.application import AbstractApplication
from node.variable import Variable

from util import *


class UtilHeader(Base):
    name = 'Util'
    short_name = 'util'

    file_extension = 'h'

    @classmethod
    def generate_check_kernel(cls, app, fields, sizes, it_space, expected, to_compare):
        check = f'''\
            if ((tpe)({expected}) != {to_compare}) {'{'}
            std::cerr << "Init check failed for element " << {' << ", " << '.join(str(i[0]) for i in it_space)} << " (expected " << {expected} << " but got " << {to_compare} << ")" << std::endl;
            return;
        {'}'}'''
        return cls.Kernel(f'checkSolution{app.name_as_postfix}', [*sizes, Variable('nIt', 'size_t')], fields, [], it_space, check, num_flop=0)


    class Application(AbstractApplication):
        def __init__(self, backend, app, sizes, kernels):
            super().__init__(backend, app, sizes, kernels)

        def generate(self):
            return f'#include "../../util.h"{newline}' + \
                2 * newline + \
                self.kernelDecls()
