import sys
from threading import Thread

from apps import get_default_apps

from backends import get_default_backends


if len(sys.argv) < 5:
    print(f'Usage: python {sys.argv[0]} machine app backend parallel')
    exit(1)

cla_machine = sys.argv[1]  # 'nvidia.alex.a40'
cla_app = sys.argv[2]      # 'all'
cla_backend = sys.argv[3]  # 'all'

cla_parallel = sys.argv[4].lower() in ['true', '1', 'on']

apps = get_default_apps()
backends = get_default_backends(cla_machine)

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

    print()

for thread in threads:
    thread.join()

print('Finished')
