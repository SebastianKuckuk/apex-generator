from app.init import Init
from app.stream import Stream
from app.stencil_1d import Stencil1D
from app.stencil_2d import Stencil2D
from app.stencil_3d import Stencil3D

from app.fma import FMA
from app.square_root import SquareRoot


def get_default_apps():
    apps = {'all': [Init, Stream,
                    Stencil1D, Stencil2D, Stencil3D,
                    FMA, SquareRoot]}

    for app in apps['all']:
        apps[app.name] = [app]

    return apps
