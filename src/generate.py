import sys

from apps import get_default_apps

from backends import get_default_backends
from backend.makefile import Makefile
from backend.util_header import UtilHeader


if len(sys.argv) < 4:
    print(f'Usage: python {sys.argv[0]} machine app backend')
    exit(1)

cla_machine = sys.argv[1]  # 'nvidia.alex.a40'
cla_app = sys.argv[2]      # 'all'
cla_backend = sys.argv[3]  # 'all'

apps = get_default_apps()
backends = get_default_backends(cla_machine)

for app in apps[cla_app]:
    print(f'Generating {app.group}/{app.name} ...')

    print(f'  ... for {UtilHeader.__name__}')

    UtilHeader.print_code_file(app, app.compose_app(UtilHeader).generate())

    print(f'  ... for {Makefile.__name__}')

    Makefile.print_code_file(app, Makefile.generate(cla_machine, app, backends['all']), format=False)

    for backend in backends[cla_backend]:
        print(f'  ... for {backend.__name__}')

        code = app.compose_app(backend).generate()

        backend.print_code_file(app, code)

    print()
