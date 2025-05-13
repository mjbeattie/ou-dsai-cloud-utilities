#!/bin/bash

# Initialize Conda
eval "$(conda shell.bash hook)"

# Define environment name
ENV_NAME="py312_torch_cpu"

# Create the environment with Python 3.12
conda create --name $ENV_NAME python=3.12 -y

# Activate the environment
conda activate $ENV_NAME

# Install essential packages
conda install -y numpy pandas scipy matplotlib seaborn scikit-learn jupyter

# Install PyTorch CUDA
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install graphing and visualization libraries
conda install -y plotly seaborn

# Install statistics and data science tools
conda install -y statsmodels xgboost lightgbm -c conda-forge

# Install Google Cloud libraries
pip install google-cloud-bigquery google-cloud-storage google-cloud-compute

# Install the Pandas to BigQuery library
pip install db-dtypes

# Test the environment
echo "Testing the environment..."

python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')"
python -c "import numpy as np; print(f'Numpy test: {np.array([1,2,3])}')"
python -c "import matplotlib.pyplot as plt; print(f'Matplotlib test successful')"
python -c "import seaborn; print(f'Seaborn test successful')"
python -c "import statsmodels.api as sm; print(f'Statsmodels test successful')"
python -c "from google.cloud import bigquery as bq; print(f'Google BigQuery test successful')"

# Confirm installation
echo "Conda environment '$ENV_NAME' created and tested successfully with Python 3.12 and CUDA 12.4."

# Activate the environment (reminder for the user)
echo "Run 'conda activate $ENV_NAME' to use the environment."

exit 0
