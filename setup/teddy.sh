#!/usr/bin/bash

source ${ENVNAME}/bin/activate

git clone https://huggingface.co/Merck/TEDDY

cd TEDDY

uv pip install poetry

PYVERSION="$(python -c 'import platform; print(platform.python_version())')"
sed -i "s/3\.11\.10/>=$PYVERSION/g" pyproject.toml
sed -i "s/3\.11\.10/$PYVERSION/g" poetry.lock

printf "[project]\n name = \"myproject\"\n version = \"0.1.0\"\n" >> poetry.lock

poetry build
uv pip install dist/teddy-0.1.0-py3-none-any.whl

cd ..
rm -rf TEDDY

cd ${ENVNAME}/lib/python3.11/site-packages/teddy/models/teddy_g/
cd 160M/
rm ./*
wget https://huggingface.co/Merck/TEDDY/resolve/main/teddy/models/teddy_g/160M/added_tokens.json
wget https://huggingface.co/Merck/TEDDY/resolve/main/teddy/models/teddy_g/160M/config.json
wget https://huggingface.co/Merck/TEDDY/resolve/main/teddy/models/teddy_g/160M/model.safetensors
wget https://huggingface.co/Merck/TEDDY/resolve/main/teddy/models/teddy_g/160M/special_tokens_map.json
wget https://huggingface.co/Merck/TEDDY/resolve/main/teddy/models/teddy_g/160M/tokenizer_config.json
wget https://huggingface.co/Merck/TEDDY/resolve/main/teddy/models/teddy_g/160M/vocab.txt
cd ../70M/
rm ./*
wget https://huggingface.co/Merck/TEDDY/resolve/main/teddy/models/teddy_g/70M/added_tokens.json
wget https://huggingface.co/Merck/TEDDY/resolve/main/teddy/models/teddy_g/70M/config.json
wget https://huggingface.co/Merck/TEDDY/resolve/main/teddy/models/teddy_g/70M/model.safetensors
wget https://huggingface.co/Merck/TEDDY/resolve/main/teddy/models/teddy_g/70M/special_tokens_map.json
wget https://huggingface.co/Merck/TEDDY/resolve/main/teddy/models/teddy_g/70M/tokenizer_config.json
wget https://huggingface.co/Merck/TEDDY/resolve/main/teddy/models/teddy_g/70M/vocab.txt
cd ../400M/
rm ./*
wget https://huggingface.co/Merck/TEDDY/resolve/main/teddy/models/teddy_g/400M/added_tokens.json
wget https://huggingface.co/Merck/TEDDY/resolve/main/teddy/models/teddy_g/400M/config.json
wget https://huggingface.co/Merck/TEDDY/resolve/main/teddy/models/teddy_g/400M/model.safetensors
wget https://huggingface.co/Merck/TEDDY/resolve/main/teddy/models/teddy_g/400M/special_tokens_map.json
wget https://huggingface.co/Merck/TEDDY/resolve/main/teddy/models/teddy_g/400M/tokenizer_config.json
wget https://huggingface.co/Merck/TEDDY/resolve/main/teddy/models/teddy_g/400M/vocab.txt

cd ../../../../../..
