#!/usr/bin/bash

source ${ENVNAME}/bin/activate

git clone https://huggingface.co/ctheodoris/Geneformer
cd Geneformer
pip install .

cd ..
rm -rf Geneformer

rm -rf ${ENVNAME}/lib/python3.11/site-packages/seaborn/*
touch ${ENVNAME}/lib/python3.11/site-packages/seaborn/__init__.py
echo "def set(): return" > ${ENVNAME}/lib/python3.11/site-packages/seaborn/__init__.py

#download tokens
cd  ${ENVNAME}/lib/python3.11/site-packages/geneformer
rm *.pkl
wget https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer/ensembl_mapping_dict_gc104M.pkl
wget https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer/gene_median_dictionary_gc104M.pkl
wget https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer/gene_name_id_dict_gc104M.pkl
wget https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer/token_dictionary_gc104M.pkl
cd gene_dictionaries_30m
rm *.pkl
wget https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer/gene_dictionaries_30m/ensembl_mapping_dict_gc30M.pkl
wget https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer/gene_dictionaries_30m/gene_median_dictionary_gc30M.pkl
wget https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer/gene_dictionaries_30m/gene_name_id_dict_gc30M.pkl
wget https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl

cd ..
mkdir -p fine_tuned_models
cd fine_tuned_models
mkdir -p Geneformer-V2-104M
cd Geneformer-V2-104M
wget https://huggingface.co/ctheodoris/Geneformer/resolve/main/Geneformer-V2-104M/config.json
wget http://huggingface.co/ctheodoris/Geneformer/resolve/main/Geneformer-V2-104M/model.safetensors
wget http://huggingface.co/ctheodoris/Geneformer/resolve/main/Geneformer-V2-104M/training_args.bin
wget http://huggingface.co/ctheodoris/Geneformer/resolve/main/Geneformer-V2-104M/generation_config.json
cd ..
mkdir -p Geneformer-V2-316M
cd Geneformer-V2-316M
wget https://huggingface.co/ctheodoris/Geneformer/resolve/main/Geneformer-V2-316M/config.json
wget http://huggingface.co/ctheodoris/Geneformer/resolve/main/Geneformer-V2-316M/model.safetensors
wget http://huggingface.co/ctheodoris/Geneformer/resolve/main/Geneformer-V2-316M/training_args.bin
wget http://huggingface.co/ctheodoris/Geneformer/resolve/main/Geneformer-V2-316M/generation_config.json
