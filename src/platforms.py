def platform(machine, backend):
    # Supported machines:
    #   nvidia
    #       .docker
    #           .a40
    #           .a100
    #           .h100
    #           .h200
    #       .alex
    #           .a40
    #           .a100
    #       .helma
    #           .h100
    #           .h200
    #   amd
    #       .testfront
    #           .aquavan1.mi300x
    #           .aquavan2.mi300a

    if machine not in [
        'nvidia.docker.a40', 'nvidia.docker.a100', 'nvidia.docker.h100', 'nvidia.docker.h200',
        'nvidia.alex.a40', 'nvidia.alex.a100',
        'nvidia.helma.h100', 'nvidia.helma.h200',
        'amd.testfront.aquavan1.mi300x', 'amd.testfront.aquavan2.mi300a']:
        raise ValueError(f'Unknown machine: {machine}')


    def get_sm():
        if machine.startswith('nvidia'):
            if machine.endswith('h100') or machine.endswith('h200'):
                return '90'
            elif machine.endswith('a100'):
                return '80'
            elif machine.endswith('a40'):
                return '86'

        raise ValueError(f'Unknown machine: {machine}')


    def def_kokkos_path():
        if machine.startswith('nvidia.docker'):
            return '/root/kokkos'
        elif machine.startswith('nvidia'):
            return '$(WORK)/kokkos'
        elif machine.startswith('amd'):
            return '$(WORK)/kokkos'

        raise ValueError(f'Unknown machine: {machine}')
    

    def def_kokkos_lib_path():
        if machine.startswith('nvidia.docker'):
            return 'lib'
        elif machine.startswith('nvidia'):
            return 'lib64'
        elif machine.startswith('amd'):
            return 'lib64'

        raise ValueError(f'Unknown machine: {machine}')


    compiler, flags, libs = None, None, None

    if backend in ['Base', 'OpenMP Host']:
        # default for all machines
        compiler = 'g++'
        flags = ['-O3', '-march=native', '-std=c++17']
        if backend in ['OpenMP Host']:
            flags.append('-fopenmp')

    elif backend.startswith('CUDA'):
        if machine.startswith('nvidia'):
            if machine.startswith('nvidia.helma') or machine.startswith('nvidia.docker'):
                compiler = 'nvcc'
                flags = ['-O3', '-std=c++17', f'-arch=sm_{get_sm()}', '--expt-extended-lambda', '--expt-relaxed-constexpr']
            else:
                compiler = 'nvc++'
                flags = ['-O3', '-fast', '-std=c++17']

    elif backend.startswith('HIP'):
        if machine.startswith('amd'):
            compiler = 'hipcc'
            flags = ['-x', 'hip', '-O3', '-std=c++17', '-munsafe-fp-atomics']

    elif backend.startswith('SYCL'):
        if machine.startswith('nvidia'):
            compiler = 'icpx'
            flags = ['-O3', '-march=native', '-std=c++17', '-fsycl', '-fsycl-targets=nvptx64-nvidia-cuda', '-Xsycl-target-backend']
            flags.append(f'--cuda-gpu-arch=sm_{get_sm()}')

    elif backend.startswith('OpenACC'):
        if machine.startswith('nvidia'):
            compiler = 'nvc++'
            flags = ['-O3', '-std=c++17', '-acc=gpu', '-target=gpu']
            if 'OpenACC Managed Memory' == backend:
                if machine.startswith('nvidia.helma') or machine.startswith('nvidia.docker'):
                    flags.append('-gpu=mem:unified')
                else:
                    flags.append('-gpu=managed')

    elif backend.startswith('OpenMP Target'):
        if machine.startswith('nvidia'):
            compiler = 'nvc++'
            flags = ['-O3', '-std=c++17', '-mp=gpu', '-target=gpu']
            if 'OpenMP Target Managed Memory' == backend:
                if machine.startswith('nvidia.helma') or machine.startswith('nvidia.docker'):
                    flags.append('-gpu=mem:unified')
                else:
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
            flags = ['-O3', '-std=c++17', '-stdpar=gpu', '-target=gpu', f'-gpu=cc{get_sm()}']
        elif machine in ['amd.testfront.aquavan1']:
            compiler = '/opt/rocm/bin/amdclang++'
            flags = ['--hipstdpar', f'--hipstdpar-path={os.environ["WORK"]}/roc-stdpar/include', '--offload-arch=gfx942:xnack+']

    elif 'Kokkos Host Serial' == backend:
        # default for all machines
        compiler = 'g++'
        flags = ['-O3', '-march=native', '-std=c++17',
                 f'-I{def_kokkos_path()}/install-serial/include',
                 f'-L{def_kokkos_path()}/install-serial/{def_kokkos_lib_path()}']
        if 'Kokkos Host OpenMP' == backend:
            flags.append('-fopenmp')
        libs = ['-lkokkoscore', '-ldl']

    elif 'Kokkos Host OpenMP' == backend:
        # default for all machines
        compiler = 'g++'
        flags = ['-O3', '-march=native', '-std=c++17', '-fopenmp',
                 f'-I{def_kokkos_path()}/install-omp/include',
                 f'-L{def_kokkos_path()}/install-omp/{def_kokkos_lib_path()}']
        libs = ['-lkokkoscore', '-ldl']

    elif 'Kokkos CUDA' == backend:
        if machine.startswith('nvidia'):
            compiler = f'{def_kokkos_path()}/install-cuda/bin/nvcc_wrapper'
            flags = ['-O3', '-march=native', '-std=c++17', f'-arch=sm_{get_sm()}',
                     '--expt-extended-lambda', '--expt-relaxed-constexpr',
                     f'-I{def_kokkos_path()}/install-cuda/include',
                     f'-L{def_kokkos_path()}/install-cuda/{def_kokkos_lib_path()}']
            libs = ['-lkokkoscore', '-ldl', '-lcuda']

    return compiler, flags, libs
