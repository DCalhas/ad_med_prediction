#~/usr/bin/bash

source ${ENVNAME}/bin/activate

git clone https://github.com/biomap-research/scFoundation.git

cd scFoundation/model/models/
wget https://huggingface.co/genbio-ai/scFoundation/resolve/main/models.ckpt

#install requirements
uv pip install --no-deps local-attention einops scanpy

uv pip install --no-deps hyper_connections==0.2.1

cd ..

sed -i 's/from load /from scfoundation.load /g' *.py
sed -i 's/ pretrainmodels/ scfoundation.pretrainmodels/g' *.py
sed -i 's/pretrainmodels import select_model/pretrainmodels.select_model import select_model/g' *.py
sed -i 's/ pretrainmodels/ scfoundation.pretrainmodels/g' pretrainmodels/*.py
sed -i 's/ \.performer/ scfoundation.pretrainmodels.performer/g' pretrainmodels/*.py
sed -i 's/ \.mae_autobin/ scfoundation.pretrainmodels.mae_autobin/g' pretrainmodels/*.py
sed -i 's/ \.transformer/ scfoundation.pretrainmodels.transformer/g' pretrainmodels/*.py

printf 'import scfoundation.load\nimport scfoundation.pretrainmodels' > __init__.py
printf 'import scfoundation.pretrainmodels.performer\nimport scfoundation.pretrainmodels.reversible\nimport scfoundation.pretrainmodels.transformer\nimport scfoundation.pretrainmodels.mae_autobin \nimport scfoundation.pretrainmodels.pytorchTransformer \nimport scfoundation.pretrainmodels.select_model' > pretrainmodels/__init__.py

cd ..	
mv model scfoundation
mv scfoundation ${ENVNAME}/lib/python3.11/site-packages/.
cd ..
rm -rf scFoundation
