import math
import os
import re
import subprocess
import sys
import time

import pandas as pd

from apps import get_default_apps

from backends import get_default_backends
from backend.backend import Backend


def benchmark(machine, app, backends, gpu_for_filename, num_repeat=3, show_plot=False):
    print(f'Benchmarking {app.group}/{app.name} ...')

    # set up output folder and create it if necessary
    output_folder = Backend.default_measurement_dir(machine, app)
    output_folder.mkdir(parents=True, exist_ok=True)

    # target output file and read it if already existent
    output_file = output_folder / Backend.default_measurement_file(machine, app)
    columns = ['index', 'gpu', 'backend', 'nx', 'ny', 'nz', 'nIt', 'nWarmUp', 'type', *app.additional_parameters, 'time', 'mlups', 'bandwidth', 'compute']
    index = 'index'

    if output_file.is_file():
        df = pd.read_csv(output_file, header=0, names=columns, index_col=index)
    else:  # output file doesn't exist already -> prepare new data frame
        df = pd.DataFrame(columns=columns)
        df.set_index(index, inplace=True)

    # prepare samples
    # sizes_to_bench = [64, 1024] # dummy test input for debugging
    sizes_to_bench = app.sizes_to_bench()
    params_to_bench = app.params_to_bench()

    types = ['double']

    # prepare environment variables
    env = os.environ.copy()
    env['OMP_PROC_BIND'] = 'close'
    env['OMP_PLACES'] = 'cores'

    last_save = time.time()

    for backend in backends:
        measurements = []

        def save_df():
            nonlocal df, measurements
            df = pd.concat([df, pd.DataFrame(measurements)], ignore_index=True)
            measurements = []

            nonlocal last_save
            last_save = time.time()

            df.to_csv(output_file)

        for tpe in types:
            print(f'  ... for {tpe} ...')

            for params in params_to_bench:
                print(f'   ... with {params} ...')

                for (i, size) in enumerate(sizes_to_bench):
                    print(f'\r   ... with {backend.name.ljust(30)} --- {round(100 * i / len(sizes_to_bench))}%', end='')

                    num_cells = size**app.dimensionality
                    n_warm = 2
                    n_it = 2**(min(10, 1 + max(0, 46 - 2 * int(math.log2(num_cells)))))

                    if 'flops' == app.metric: # overwrite n_warm and n_it for compute heavy benchmarks that feature hot inner loop
                        n_warm = 2
                        n_it = 8

                    not_measured = df.loc[
                        (df['gpu'] == gpu_for_filename) &
                        (df['backend'] == backend.name) &
                        (df['nx'] == size) &
                        (df['ny'] == size if app.dimensionality > 1 else 1) &
                        (df['nz'] == size if app.dimensionality > 2 else 1) &
                        (df['nIt'] == n_it) &
                        (df['nWarmUp'] == n_warm) &
                        (df['type'] == tpe)
                    ]
                    for (param_name, param_value) in zip(app.additional_parameters, params):
                        not_measured = not_measured.loc[(not_measured[param_name] == param_value)]
                    not_measured = not_measured.empty

                    if not_measured:
                        local_results = []
                        for _ in range(num_repeat):
                            out = subprocess.check_output([backend.default_bin_dir(machine, app) / backend.default_bin_file(machine, app),
                                                        tpe, *[f'{size}' for _ in range(app.dimensionality)],
                                                        *[f'{param}' for param in params], f'{n_warm}', f'{n_it}'],
                                                        env=env)
                            out = out.decode("utf-8")

                            elapsed = float(re.findall(r'elapsed time: *(\d+(?:\.\d+)?|\d+(?:\.\d+)?e-\d+) ms', out)[0])
                            mlups = float(re.findall(r'MLUP/s: *(\d+(?:\.\d+)?|\d+(?:\.\d+)?e-\d+)\n', out)[0])
                            bandwidth = float(re.findall(r'bandwidth: *(\d+(?:\.\d+)?|\d+(?:\.\d+)?e-\d+) GB/s', out)[0])
                            compute = float(re.findall(r'compute: *(\d+(?:\.\d+)?|\d+(?:\.\d+)?e-\d+) GFLOP/s', out)[0])

                            local_results.append([elapsed, mlups, bandwidth, compute])

                        local_results.sort(key=lambda m: m[0])

                        measurements.append({
                            'gpu': gpu_for_filename,
                            'backend': backend.name,
                            'nx': size,
                            'ny': size if app.dimensionality > 1 else 1,
                            'nz': size if app.dimensionality > 2 else 1,
                            'nIt': n_it,
                            'nWarmUp': n_warm,
                            'type': tpe,
                            **dict(zip(app.additional_parameters, params)),
                            'time': local_results[0][0],
                            'mlups': local_results[0][1],
                            'bandwidth': local_results[0][2],
                            'compute': local_results[0][3]
                        })

                        if time.time() - last_save >= 120 and len(measurements) > 0:
                            save_df()

        print(f'\r   ... with {backend.name.ljust(30)} --- done')

        df.sort_values(['gpu', 'backend', 'type', *app.additional_parameters, 'nz', 'ny', 'nx'], ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)

        save_df()

    print(f'Wrote results to \'{output_file}\'')

    # write to file - xlsx
    xlsx_file = output_file.with_suffix('.xlsx')
    print(f'Wrote results to \'{xlsx_file}\'')

    df.to_excel(xlsx_file)

def eval_gpu(machine):
    # evaluate GPU running on
    if machine.startswith('nvidia'):
        out = subprocess.check_output(['nvidia-smi', '-L'])
        out = out.decode('utf-8').strip()
        gpu = re.findall(r'GPU 0: (.*) \(UUID: GPU', out)[0]
        gpu_for_filename = gpu.replace(' ', '-').replace('NVIDIA-', '').replace('GeForce-', '')
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'])
        out = out.decode('utf-8').strip()
        gpu_cc = float(out)

        print(f'Running on GPU {gpu} ({gpu_for_filename}), compute capability {gpu_cc}')

    elif machine.startswith('amd'):
        out = subprocess.check_output(['rocm-smi', '-d', '0', '--showproductname'])
        out = out.decode('utf-8').strip()
        gpu = re.findall(r'Card Series: (.*)\n', out)[0].strip()
        gpu_for_filename = gpu.replace(' ', '-').replace('AMD-', '').replace('Instinct-', '').replace('-OAM', '')
        gfx = re.findall(r'GFX Version: (.*)\n', out)[0].strip()

        print(f'Running on GPU {gpu} ({gpu_for_filename}), gfx {gfx}')

    else:
        gpu_for_filename = 'none'

    return gpu_for_filename

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(f'Usage: python {sys.argv[0]} machine app backend')
        exit(1)

    for cla_machine in sys.argv[1].split(','):          # 'nvidia.alex.a40'
        gpu_for_filename = eval_gpu(cla_machine)

        for cla_app in sys.argv[2].split(','):          # 'all'
            for cla_backend in sys.argv[3].split(','):  # 'all'
                apps = get_default_apps()
                backends = get_default_backends(cla_machine)

                for app in apps[cla_app]:
                    benchmark(cla_machine, app, backends[cla_backend], gpu_for_filename)
