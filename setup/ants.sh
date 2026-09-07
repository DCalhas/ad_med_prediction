#!/usr/bin/bash

source ${ENVNAME}/bin/activate

wget -P /tmp https://github.com/ANTsX/ANTs/releases/download/v2.6.3/ants-2.6.3-almalinux8-X64-gcc.zip

unzip /tmp/ants-2.6.3-almalinux8-X64-gcc.zip

mkdir -p ${ENVNAME}/bin
mkdir -p ${ENVNAME}/lib64

mv ants-2.6.3/bin/* ${ENVNAME}/bin/.
mv ants-2.6.3/lib64/* ${ENVNAME}/lib64/.

rm -rf ants-2.6.3-almalinux8-X64-gcc.zip ants-2.6.3
