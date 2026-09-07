#!/usr/bin/bash

source ${ENVNAME}/bin/activate

#pip install --no-build-isolation flash_attn==1.0.4
uv pip install scgpt ipython torchtext==0.18.0
uv pip install anndata==0.12.5 wandb==0.22.3
uv pip install torch==2.3.0 transformers==4.57.3 torchvision==0.18.0
uv pip install captum==0.9.0
uv pip install accelerate==0.32.0
uv pip install --no-deps peft==0.11.1

#printf "\nsparse_dataset=SparseDataset" >> ${ENVNAME}/lib/python3.11/site-packages/anndata/_core/sparse_dataset.py

#mkdir -p ${ENVNAME}/lib/python3.11/site-packages/scgpt
git clone git@github.com:bowang-lab/scGPT.git
cd scGPT
pip install --no-deps .
cd ..
rm -rf scGPT

cd ${ENVNAME}/lib/python3.11/site-packages/scgpt
mkdir -p pretrained

uv pip install gdown

gdown --folder "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y" -O pretrained

gdown --folder "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y" -O .
