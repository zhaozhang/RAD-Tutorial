#!/bin/bash

module load pytorch-conda/2.8
RAD_ENV="$HOME/python_venvs/rad-delta-py311"
python -m venv --system-site-packages "$RAD_ENV"
"$RAD_ENV/bin/python" -m pip install gdown==6.2.0
"$RAD_ENV/bin/python" -m ipykernel install --user --name rad-delta-py311 --display-name 'RAD Delta (PyTorch 2.8)'
