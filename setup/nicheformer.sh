#!/usr/bin/bash

source ${ENVNAME}/bin/activate

uv pip install anndata numba

git clone https://huggingface.co/theislab/Nicheformer

mv ./Nicheformer ./nicheformer

cd nicheformer 

sed -i 's/from .configuration_nicheformer/from nicheformer.configuration_nicheformer/g' __init__.py
sed -i 's/from .modeling_nicheformer/from nicheformer.modeling_nicheformer/g' __init__.py

cd ..

mv nicheformer ${ENVNAME}/lib/python3.11/site-packages/.

mkdir -p ${ENVNAME}/lib/python3.11/site-packages/nicheformer/pretrainedmodel

cd ${ENVNAME}/lib/python3.11/site-packages/nicheformer/pretrainedmodel

rm config.json model.safetensors model.h5ad vocab.json

wget https://huggingface.co/theislab/Nicheformer/resolve/main/config.json
wget http://huggingface.co/theislab/Nicheformer/resolve/main/model.safetensors
wget http://huggingface.co/theislab/Nicheformer/resolve/main/model.h5ad
wget http://huggingface.co/theislab/Nicheformer/resolve/main/vocab.json

cd ..

wget https://huggingface.co/theislab/Nicheformer/resolve/main/config.json
wget http://huggingface.co/theislab/Nicheformer/resolve/main/model.safetensors
wget http://huggingface.co/theislab/Nicheformer/resolve/main/model.h5ad
wget http://huggingface.co/theislab/Nicheformer/resolve/main/vocab.json
