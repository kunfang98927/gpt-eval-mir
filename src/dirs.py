# Description: This file contains the paths to the directories used in the project.

import os

# Root directory of the project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "../../../../Perf2Score/data/raw/")
METADATA_DIR = os.path.join(PROJECT_ROOT, "data/metadata/")

# Experiments directory
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments/")
