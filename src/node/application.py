import math

from node.kernel import PseudoKernel

from util import *


class AbstractApplication:
    def __init__(self, backend, app, sizes, parameters, kernels):
        self.backend = backend
        self.app = app
        self.sizes = sizes

        if parameters is None:
            self.parameters = []
        elif isinstance(parameters, list):
            self.parameters = parameters
        else:
            self.parameters = [parameters]

        if isinstance(kernels, list):
            self.kernels = kernels
        else:
            self.kernels = [kernels]

        self.fields = sorted({f for k in self.kernels for f in k.reads + k.writes}, key=lambda f: f.name)

    def generate(self):
        pass

    def kernelDecls(self):
        return newline.join(kernel.generate() for kernel in self.kernels if not isinstance(kernel, PseudoKernel))

    def fieldAllocates(self):
        code = newline.join(f.h_declare() + newline + f.h_allocate() for f in self.fields)
        code_device = newline.join(f.d_declare() + newline + f.d_allocate() for f in self.fields if f.has_device_ptr)

        return code + (2 * newline + code_device if 0 != len(code_device) else '')

    def fieldFrees(self):
        code_device = newline.join(f.d_free() for f in self.fields if f.has_device_ptr)
        code = newline.join(f.h_free() for f in self.fields)

        return (code_device + 2 * newline if 0 != len(code_device) else '') + code

    def toDeviceCopies(self):
        copies = [f.copyToDevice() for f in self.fields]
        code = newline.join([c for c in copies if c is not None])

        return (newline + code + newline if '' != code else '')

    def toHostCopies(self):
        copies = [f.copyToHost() for f in self.fields]
        code = newline.join([c for c in copies if c is not None])

        return (code + 2 * newline if '' != code else '')

    def synchronize(self):
        code = self.backend.synchronize()

        return (code + newline if code is not None else '')

    def mainStart(self):
        size_list = ', '.join(f'{s}' for s in self.sizes)
        if len(self.parameters) > 0:
            param_list = ', ' + ', '.join(f'{p}' for p in self.parameters)
            param_decls = newline.join(f'{p.tpe} {p.name};' for p in self.parameters) + newline
        else:
            param_list = ''
            param_decls = ''

        return \
            f'template<typename tpe>{newline}' + \
            f'inline int realMain(int argc, char *argv[]) {"{"}{newline}' + \
            f'char* tpeName;{newline}' + \
            f'size_t {size_list}, nItWarmUp, nIt;{newline}' + \
            param_decls + \
            f'parseCLA_{len(self.sizes)}d(argc, argv, tpeName, {size_list}{param_list}, nItWarmUp, nIt);{newline}'

    def mainAllocateAndInit(self):
        size_list = ', '.join(f'{s}' for s in self.sizes)
        field_list = ', '.join(f.name for f in self.fields)

        return \
            self.fieldAllocates() + newline + \
            newline + \
            f'// init{newline}' + \
            f'init{self.app.title().replace("-", "")}({field_list}, {size_list});{newline}' + \
            self.toDeviceCopies()

    def mainMiddle(self):
        total_size = f'{math.prod(self.sizes)}'

        num_flop = sum(k.num_flop for k in self.kernels)
        num_byte = ' + '.join(f'sizeof({f.tpe})' for k in self.kernels for f in k.reads + k.writes)

        return \
            f'// warm-up{newline}' + \
            f'for (size_t i = 0; i < nItWarmUp; ++i) {"{"}{newline}' + \
            newline.join(k.launch() for k in self.kernels) + newline + \
            f'{"}"}{newline}' + \
            self.synchronize() + \
            newline + \
            f'// measurement{newline}' + \
            f'auto start = std::chrono::steady_clock::now();{newline}' + \
            newline + \
            f'for (size_t i = 0; i < nIt; ++i) {"{"}{newline}' + \
            newline.join(k.launch() for k in self.kernels) + newline + \
            f'{"}"}{newline}' + \
            self.synchronize() + \
            newline + \
            f'auto end = std::chrono::steady_clock::now();{newline}' + \
            newline + \
            f'printStats<tpe>(end - start, nIt, {total_size}, tpeName, {num_byte}, {num_flop});{newline}'

    def mainEnd(self):
        size_list = ', '.join(f'{s}' for s in self.sizes)
        field_list = ', '.join(f.name for f in self.fields)

        if len(self.parameters) > 0:
            param_list = ', ' + ', '.join(f'{p}' for p in self.parameters)
        else:
            param_list = ''

        return self.toHostCopies() + \
            f'// check solution{newline}' + \
            f'checkSolution{self.app.title().replace("-", "")}({field_list}, {size_list}, nIt + nItWarmUp{param_list});{newline}' + \
            newline + \
            self.fieldFrees() + newline + \
            newline + \
            f'return 0;{newline}' + \
            f'{"}"}{newline}'

    def mainWrapper(self):
        types = ['int', 'long', 'float', 'double']

        switch = newline.join(f'if ("{tpe}" == tpeName){newline}return realMain<{tpe}>(argc, argv);'
                              for tpe in types)

        body = \
            f'if (argc < 2) {"{"}{newline}' + \
            f'std::cout << "Missing type specification" << std::endl;{newline}' + \
            f'return -1;{newline}' + \
            f'{"}"}{newline}' + \
            f'{newline}' + \
            f'std::string tpeName(argv[1]);{newline}' + \
            f'{newline}' + \
            switch + newline + \
            newline + \
            f'std::cout << "Invalid type specification (" << argv[1] << "); supported types are" << std::endl;{newline}' + \
            f'std::cout << "  {", ".join(types)}" << std::endl;{newline}' + \
            f'return -1;{newline}'

        return \
            f'int main(int argc, char *argv[]) {"{"}{newline}' + \
            body + newline + \
            f'{"}"}{newline}'
