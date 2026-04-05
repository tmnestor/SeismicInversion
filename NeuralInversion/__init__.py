"""Neural inference for seismic inversion: hybrid Newton-Net + amortized VI.

Provides a Transformer-based encoder that produces warm-start estimates and
calibrated uncertainty for layered earth model parameters from frequency-domain
reflectivity data R(omega, p).  The warm start feeds into the existing
Newton-Levenberg-Marquardt solver for rapid quadratic convergence.
"""

from .hybrid_inversion import (
    HybridInversionResult,
    hybrid_invert_taup,
    load_trained_network,
    network_predict,
)
from .inference_config import (
    ArchitectureConfig,
    DataGenConfig,
    HybridInversionConfig,
    InferenceNetConfig,
    TrainingConfig,
    load_inference_config,
    save_inference_config,
)
from .inference_data import (
    ReflectivityDataset,
    generate_training_data,
    prepare_input,
)
from .inference_net import (
    ContinuousPositionalEncoding,
    SeismicInferenceNet,
    gaussian_nll_loss,
)
from .inference_train import evaluate_inference_net, train_inference_net

__all__ = [
    "ArchitectureConfig",
    "ContinuousPositionalEncoding",
    "DataGenConfig",
    "HybridInversionConfig",
    "HybridInversionResult",
    "InferenceNetConfig",
    "ReflectivityDataset",
    "SeismicInferenceNet",
    "TrainingConfig",
    "evaluate_inference_net",
    "gaussian_nll_loss",
    "generate_training_data",
    "hybrid_invert_taup",
    "load_inference_config",
    "load_trained_network",
    "network_predict",
    "prepare_input",
    "save_inference_config",
    "train_inference_net",
]
