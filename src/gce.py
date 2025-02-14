import sys

from apps import get_default_apps

from backends import get_default_backends

from compile import compile
from execute import execute
from generate import generate


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print(f'Usage: python {sys.argv[0]} machine app backend parallel')
        exit(1)

    cla_machine = sys.argv[1]  # 'nvidia.alex.a40'
    cla_app = sys.argv[2]      # 'all'
    cla_backend = sys.argv[3]  # 'all'

    cla_parallel = sys.argv[4].lower() in ['true', '1', 'on']

    apps = get_default_apps()
    backends = get_default_backends(cla_machine)

    generate(cla_machine, cla_app, cla_backend, apps, backends)
    compile(cla_machine, cla_app, cla_backend, cla_parallel, apps, backends)
    execute(cla_machine, cla_app, cla_backend, apps, backends)
