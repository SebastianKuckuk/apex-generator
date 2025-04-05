from matplotlib import pyplot as plt
import pandas as pd
import sys

from apps import get_default_apps

from backend.backend import Backend


def plot(machine, app, show_plot=False):
    print(f'Running {app.group}/{app.name} ...')

    # configure the folder containing the measurements
    data_folder = Backend.default_measurement_dir(machine, app)
    data_folder.mkdir(parents=True, exist_ok=True)

    # configure measurement data file and read it
    data_file = data_folder / Backend.default_measurement_file(machine, app)
    columns = ['index', 'gpu', 'backend', 'nx', 'ny', 'nz', 'nIt', 'nWarmUp', 'type', *app.additional_parameters, 'time', 'mlups', 'bandwidth', 'compute']
    index = 'index'

    df = pd.read_csv(data_file, header=0, names=columns, index_col=index)

    # add auxiliary columns
    df['numCells'] = df['nx'] * df['ny'] * df['nz']
    df['version'] = df[['backend', 'gpu', *app.additional_parameters]].apply(
        lambda x: f'{x[0]} ({x[1]})' + (' - ' + ', '.join([f'{p}' for p in x[2:]]) if len(x) > 2 else ''), axis=1)

    # filter by gpu
    gpus = df['gpu'].unique()
    df = df[df.apply(lambda x: x['gpu'] in gpus, axis=1)]

    # filter by back end
    backends = df['backend'].unique()
    df = df[df.apply(lambda x: x['backend'] in backends, axis=1)]

    # remove unnecessary columns and group by group_col
    group_col = 'version'
    df = df[[group_col, 'gpu', 'backend', 'numCells', 'bandwidth']]
    df.set_index('numCells', inplace=True)
    # df = df.groupby(group_col)

    # plot resulting data frame
    plot_file_name = data_file.with_suffix('.pdf')
    print(f'Plotting results to \'{plot_file_name}\'')

    df.groupby(group_col)['bandwidth'].plot(legend=True, logx=True, figsize=[11.7, 8.3])
    plt.savefig(plot_file_name)

    if show_plot:
        plt.show()

    # plot per back end
    for backend in backends:
        print(f'Plotting for back end {backend}')
        plt.clf()
        filtered = df[backend == df['backend']]
        filtered.groupby('version')['bandwidth'].plot(legend=True, logx=True, figsize=[11.7, 8.3])
        plt.savefig(data_folder / f'{app.name}-{backend.replace(" ", "")}.pdf')

    # plot per gpu
    for gpu in gpus:
        print(f'Plotting for GPU {gpu}')
        plt.clf()
        filtered = df[gpu == df['gpu']]
        filtered.groupby('version')['bandwidth'].plot(legend=True, logx=True, figsize=[11.7, 8.3])
        plt.savefig(data_folder / f'{app.name}-{gpu.replace(" ", "")}.pdf')


if len(sys.argv) < 3:
    print(f'Usage: python {sys.argv[0]} machine app')
    exit(1)

cla_machine = sys.argv[1]  # 'nvidia.alex.a40'
cla_app = sys.argv[2]      # 'all'

apps = get_default_apps()

for app in apps[cla_app]:
    plot(cla_machine, app)
