from skipcore.inference.torch_knn import TorchKNNBackend
from skipcore.inference.knn_base import KNNBackendBase

__all__ = ["TorchKNNBackend", "KNNBackendBase"]

# Try to import faiss backends
try:
    from skipcore.inference.faiss_gpu import FaissGPUBackend
    __all__.append("FaissGPUBackend")
except ImportError:
    pass

try:
    from skipcore.inference.faiss_cpu import FaissCPUBackend
    __all__.append("FaissCPUBackend")
except ImportError:
    pass
