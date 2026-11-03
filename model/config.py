"""Global configuration for the chest X-ray TB pipeline.

Paths are anchored to this package directory so scripts work from any working directory.
"""

from pathlib import Path

MODEL_ROOT: Path = Path(__file__).resolve().parent

# Image and preprocessing parameters
IMAGE_HEIGHT: int = 512
IMAGE_WIDTH: int = 512
CHANNELS: int = 1

# Training hyperparameters (U-Net)
BATCH_SIZE: int = 4
NUM_EPOCHS: int = 100
LEARNING_RATE: float = 1e-4

# Dataset and split configuration
DATA_DIR: Path = MODEL_ROOT / "data"
CSV_PATH: Path = DATA_DIR / "dataset.csv"
VALIDATION_SPLIT: float = 0.1
TEST_SPLIT: float = 0.1

# Output directories
OUTPUT_DIR: Path = MODEL_ROOT / "outputs"
CHECKPOINT_DIR: Path = OUTPUT_DIR / "checkpoints"
LOG_DIR: Path = OUTPUT_DIR / "logs"
PREDICTION_DIR: Path = OUTPUT_DIR / "predictions"
CLASSIFIER_DIR: Path = OUTPUT_DIR / "classifier"
EVALUATION_DIR: Path = OUTPUT_DIR / "evaluation"

DEFAULT_SEGMENTATION_MODEL: Path = CHECKPOINT_DIR / "best_model.h5"
DEFAULT_CLASSIFIER: Path = CLASSIFIER_DIR / "tb_classifier.joblib"

# Inference / decision parameters
MASK_THRESHOLD: float = 0.5  # probability threshold for the U-Net output
MIN_COMPONENT_PX: int = 16  # drop connected components smaller than this (at model resolution)
MIN_LESION_AREA_FRACTION: float = 0.001  # below this fraction of the image, treat as "no lesion detected"
RISK_LOW_MAX: float = 0.35  # tb_probability < 0.35 -> low
RISK_HIGH_MIN: float = 0.65  # tb_probability >= 0.65 -> high

# Quality control (an analysis that fails QC is reported with low reliability and is not patient-facing)
QC_MIN_IMAGE_SIDE: int = 256  # smaller inputs are too degraded for the 512 model
QC_MIN_CONTRAST_STD: float = 10.0  # grayscale std below this -> blank / washed-out image
QC_ASPECT_RANGE: tuple[float, float] = (0.6, 1.7)  # width / height outside -> not a frontal CXR crop
QC_TINY_ROI_FRACTION: float = 0.003  # segmented region smaller than this -> texture features unreliable
QC_HUGE_ROI_FRACTION: float = 0.35  # larger than this -> segmentation failure, not a lesion
QC_LOW_CONFIDENCE: float = 0.6  # max pixel probability below this -> low-confidence segmentation

# Miscellaneous
SHUFFLE_BUFFER_SIZE: int = 256
RANDOM_SEED: int = 42
