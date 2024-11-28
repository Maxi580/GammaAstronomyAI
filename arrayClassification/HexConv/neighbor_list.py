from ctapipe_io_magic import MAGICEventSource
from ctapipe.io import EventSource, DataLevel
from ctapipe.core import Provenance, Container, Field
from ctapipe.core.traits import Bool, UseEnum
from ctapipe.coordinates import CameraFrame
from ctapipe.instrument import (
    TelescopeDescription,
    SubarrayDescription,
    OpticsDescription,
    CameraDescription,
    CameraGeometry,
    CameraReadout,
    SizeType,
    ReflectorShape,
    FocalLengthKind,
)
from pkg_resources import resource_filename


def get_neighbor_list() -> list[list[int]]:
    """Returns a list of 1039 lists, that has the index of neighbours of each idx"""
    f = resource_filename("ctapipe_io_magic", "resources/MAGICCam.camgeom.fits.gz")
    Provenance().add_input_file(f, role="CameraGeometry")
    return CameraGeometry.from_table(f).neighbors
