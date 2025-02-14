import sys

from apps import get_default_apps

from backends import get_default_backends


def execute(cla_machine, cla_app, cla_backend, apps, backends):
    for app in apps[cla_app]:
        print(f'Executing {app.group}/{app.name} ...')

        for backend in backends[cla_backend]:
            print(f'  ... for {backend.__name__}')

            backend.exec_bin(app)

    print('Finished executing')
    print()

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(f'Usage: python {sys.argv[0]} machine app backend')
        exit(1)

    cla_machine = sys.argv[1]  # 'nvidia.alex.a40'
    cla_app = sys.argv[2]      # 'all'
    cla_backend = sys.argv[3]  # 'all'

    apps = get_default_apps()
    backends = get_default_backends(cla_machine)

    execute(cla_machine, cla_app, cla_backend, apps, backends)
