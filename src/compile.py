import sys
from threading import Thread

from apps import get_default_apps

from backends import get_default_backends


def compile(cla_machine, cla_app, cla_backend, cla_parallel, apps, backends):
    threads = []

    for app in apps[cla_app]:
        print(f'Compiling {app.group}/{app.name} ...')

        for backend in backends[cla_backend]:
            print(f'  ... with {backend.__name__}')

            if cla_parallel:
                thread = Thread(target=backend.compile_bin, args=[cla_machine, app])
                thread.start()
                threads.append(thread)
            else:
                backend.compile_bin(cla_machine, app)

    for thread in threads:
        thread.join()

    print('Finished compiling')
    print()


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print(f'Usage: python {sys.argv[0]} machine app backend parallel')
        exit(1)

    cla_parallel = sys.argv[4].lower() in ['true', '1', 'on']

    for cla_machine in sys.argv[1].split(','):          # 'nvidia.alex.a40'
        for cla_app in sys.argv[2].split(','):          # 'all'
            for cla_backend in sys.argv[3].split(','):  # 'all'
                apps = get_default_apps()
                backends = get_default_backends(cla_machine)

                compile(cla_machine, cla_app, cla_backend, cla_parallel, apps, backends)
