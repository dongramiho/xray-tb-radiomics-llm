"""Global configuration for the chest X-ray segmentation project."""

from pathlib import Path

# Image and preprocessing parameters
IMAGE_HEIGHT: int = 512
IMAGE_WIDTH: int = 512
CHANNELS: int = 1

# Training hyperparameters
BATCH_SIZE: int = 4
NUM_EPOCHS: int = 100
LEARNING_RATE: float = 1e-4

# Dataset and split configuration
CSV_PATH: Path = Path("data/shenzhen_dataset.csv")
VALIDATION_SPLIT: float = 0.1
TEST_SPLIT: float = 0.1

# Output directories
OUTPUT_DIR: Path = Path("outputs")
CHECKPOINT_DIR: Path = OUTPUT_DIR / "checkpoints"
LOG_DIR: Path = OUTPUT_DIR / "logs"
PREDICTION_DIR: Path = OUTPUT_DIR / "predictions"

# Miscellaneous
SHUFFLE_BUFFER_SIZE: int = 256
RANDOM_SEED: int = 42
