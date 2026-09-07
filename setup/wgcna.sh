#!/usr/bin/bash

source ${ENVNAME}/bin/activate

pip install --no-deps pywgcna==1.20.2
sed -i 's/sns.set_style* / /g' ${ENVNAME}/venv/lib/python3.11/site-packages/PyWGCNA/wgcna.py

pip install --no-deps gseapy==1.0.1
pip install --no-deps pyvis==0.3.1
pip install jsonpickle
pip install --no-deps reactome2py==3.0.0
pip install --no-deps biomart==0.9.2
