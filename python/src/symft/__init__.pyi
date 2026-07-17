from os import PathLike
from typing import Any, Union

from ._native import (
    BoolSamples as _BoolSamples,
    Circuit as Circuit,
    CompiledCountsSampler as CompiledCountsSampler,
    CompiledMeasurementSampler as CompiledMeasurementSampler,
    PackedSamples as _PackedSamples,
    SymFTError as SymFTError,
    active_batch_backend as active_batch_backend,
    active_cuda_backend as active_cuda_backend,
    active_simd_backend as active_simd_backend,
    cuda_enabled as cuda_enabled,
)

__version__: str

def read_stim_file(
    path: Union[str, bytes, PathLike[str], PathLike[bytes]],
) -> Circuit: ...
def sample(
    circuit: Union[Circuit, str, bytes],
    shots: int = ...,
    **kwargs: Any,
) -> Union[_BoolSamples, _PackedSamples]: ...
