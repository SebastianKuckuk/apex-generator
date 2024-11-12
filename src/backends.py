from backend.base import Base
from backend.omp_host import OMPHost
from backend.omp_target import OMPTargetExpl, OMPTargetMM
from backend.openacc import OpenAccExpl, OpenAccMM
from backend.cuda import CudaExpl, CudaMM
from backend.hip import HipExpl, HipMM
from backend.sycl import SyclBuffer, SyclExpl, SyclMM
from backend.std_par import StdPar
from backend.kokkos import KokkosSerial, KokkosOMPHost, KokkosCuda


def get_default_backends(machine=None):
    all = [Base, OMPHost,
           OMPTargetExpl, OMPTargetMM,
           OpenAccExpl, OpenAccMM]

    if machine is None or machine.startswith('nvidia'):
        all.extend([CudaExpl, CudaMM])
    if machine is None or machine.startswith('amd'):
        all.extend([HipExpl, HipMM])

    all.extend([SyclBuffer, SyclExpl, SyclMM,
                StdPar,
                KokkosSerial, KokkosOMPHost, KokkosCuda])

    backends = {'all': all}

    for backend in backends['all']:
        backends[backend.short_name] = [backend]

    return backends
