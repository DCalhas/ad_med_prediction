#!/usr/bin/bash

source ${ENVNAME}/bin/activate

uv pip install bitsandbytes==0.49.1

sed -i '1s/^/import torch\ntorch.cuda.init()\n/' ${ENVNAME}/lib/python3.11/site-packages/bitsandbytes/__init__.py

