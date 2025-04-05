import sys

from apps import get_default_apps

from backends import get_default_backends
from backend.makefile import Makefile
from backend.util_header import UtilHeader


def generate(cla_machine, cla_app, cla_backend, apps, backends):
    for app in apps[cla_app]:
        print(f'Generating {app.group}/{app.name} ...')

        print(f'  ... for {UtilHeader.__name__}')

        UtilHeader.print_code_file(cla_machine, app, app.compose_app(UtilHeader).generate())

        print(f'  ... for {Makefile.__name__}')

        Makefile.print_code_file(cla_machine, app, Makefile.generate(cla_machine, app, backends['all']), format=False)

        for backend in backends[cla_backend]:
            print(f'  ... for {backend.__name__}')

            code = app.compose_app(backend).generate()

            backend.print_code_file(cla_machine, app, code)

    print('Finished generating')
    print()

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(f'Usage: python {sys.argv[0]} machine app backend')
        exit(1)

    for cla_machine in sys.argv[1].split(','):          # 'nvidia.alex.a40'
        for cla_app in sys.argv[2].split(','):          # 'all'
            for cla_backend in sys.argv[3].split(','):  # 'all'
                apps = get_default_apps()
                backends = get_default_backends(cla_machine)

                generate(cla_machine, cla_app, cla_backend, apps, backends)
