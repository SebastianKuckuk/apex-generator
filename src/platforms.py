import os


def platform(machine, backend):
    # Supported machines:
    #   nvidia.alex.a40, nvidia.alex.a100,
    #   amd.testfront.aquavan

    compiler, flags, libs = None, None, None

    if backend in ['Base', 'OpenMP Host']:
        # default for all machines
        compiler = 'g++'
        flags = ['-O3', '-march=native', '-std=c++17']
        if backend in ['OpenMP Host']:
            flags.append('-fopenmp')

    elif backend.startswith('CUDA'):
        if machine.startswith('nvidia'):
            compiler = 'nvc++'
            flags = ['-O3', '-fast', '-std=c++17']

    elif backend.startswith('HIP'):
        if machine.startswith('amd'):
            compiler = 'hipcc'
            flags = ['-x', 'hip', '-O3', '-fast', '-std=c++17', '-munsafe-fp-atomics']

    elif backend.startswith('SYCL'):
        if machine.startswith('nvidia'):
            compiler = 'icpx'
            flags = ['-O3', '-march=native', '-std=c++17', '-fsycl', '-fsycl-targets=nvptx64-nvidia-cuda', '-Xsycl-target-backend']
            flags.append({
                'nvidia.alex.a100': '--sycl-gpu-arch=sm_80',
                'nvidia.alex.a40': '--sycl-gpu-arch=sm_86'
            }[machine])

    elif backend.startswith('OpenACC'):
        if machine.startswith('nvidia'):
            compiler = 'nvc++'
            flags = ['-O3', '-std=c++17', '-acc=gpu', '-target=gpu']
            if 'OpenACC Managed Memory' == backend:
                flags.append('-gpu=managed')

    elif backend.startswith('OpenMP Target'):
        if machine.startswith('nvidia'):
            compiler = 'nvc++'
            flags = ['-O3', '-std=c++17', '-mp=gpu', '-target=gpu']
            if 'OpenMP Target Managed Memory' == backend:
                flags.append('-gpu=managed')
        elif machine in ['amd.testfront.aquavan1']:
            compiler = '/opt/rocm/bin/amdclang++'
            flags = ['-fopenmp']
            flags.append({
                'OpenMP Target Explicit Memory': '--offload-arch=gfx942',
                'OpenMP Target Managed Memory': '--offload-arch=gfx942:xnack+',
            }[backend])

    elif backend.startswith('std::par'):
        if machine.startswith('nvidia'):
            compiler = 'nvc++'
            flags = ['-O3', '-std=c++17', '-stdpar=gpu', '-target=gpu']  # TODO: -gpu=cc80
        elif machine in ['amd.testfront.aquavan1']:
            compiler = '/opt/rocm/bin/amdclang++'
            flags = ['--hipstdpar', f'--hipstdpar-path={os.environ["WORK"]}/roc-stdpar/include', '--offload-arch=gfx942:xnack+']

    elif 'Kokkos Host Serial' == backend:
        # default for all machines
        compiler = 'g++'
        flags = ['-O3', '-march=native', '-std=c++17',
                 f'-I{os.environ["HOME"]}/kokkos/install-serial/include',
                 f'-L{os.environ["HOME"]}/kokkos/install-serial/lib64']
        if 'Kokkos Host OpenMP' == backend:
            flags.append('-fopenmp')
        libs = ['-lkokkoscore', '-ldl']

    elif 'Kokkos Host OpenMP' == backend:
        # default for all machines
        compiler = 'g++'
        flags = ['-O3', '-march=native', '-std=c++17', '-fopenmp',
                 f'-I{os.environ["HOME"]}/kokkos/install-omp/include',
                 f'-L{os.environ["HOME"]}/kokkos/install-omp/lib64']
        libs = ['-lkokkoscore', '-ldl']

    elif 'Kokkos CUDA' == backend:
        if machine.startswith('nvidia'):
            compiler = f'{os.environ["HOME"]}/kokkos/install-cuda/bin/nvcc_wrapper'
            flags = ['-O3', '-march=native', '-std=c++17',
                     {'nvidia.alex.a100': '-arch=sm_80',
                      'nvidia.alex.a40': '-arch=sm_86'
                      }[machine],
                     '--expt-extended-lambda', '--expt-relaxed-constexpr',
                     f'-I{os.environ["HOME"]}/kokkos/install-cuda/include',
                     f'-L{os.environ["HOME"]}/kokkos/install-cuda/lib64']
            libs = ['-lkokkoscore', '-ldl']

    return compiler, flags, libs
