#!/usr/bin/bash

#assumes correct python is the default one
python -m venv --copies ${ENVNAME}

source ${ENVNAME}/bin/activate
pip install uv
uv pip install matplotlib==3.10.6 numpy==1.26.1 pandas==2.3.3 transformers==4.57.1 huggingface-hub==0.35.3 datasets==4.2.0 nilearn==0.12.1 nibabel==5.3.2 scipy==1.15.3

uv pip install torch==2.8.0 torchvision==0.23.0

uv pip install pycox==0.3.0

uv pip install --no-deps ucimlrepo

#resolve known dependencies
uv pip install --upgrade pandas==2.2.0
uv pip install --upgrade pyarrow==22.0.0
uv pip install --upgrade --no-deps spatialdata==0.6.1 #this installs numpy==2.3.5 raises error afterwards

