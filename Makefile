# Makefile for DIFERRE environment setup

ENVNAME := venv
ENVDIR := $(shell pwd)/${ENVNAME}

.PHONY: all base geneformer scgpt teddy scfoundation ants fsl

BOLD := \033[1m
MAGENTA := \033[0;35m
RESET := \033[0m

define cecho
@printf "\033[1;35m%s\033[0m\n" "$1"
endef

all: base bitsandbytes geneformer scgpt teddy scfoundation nicheformer wgcna

base: ${ENVDIR}/lib/python3.11/site-packages/torch 

bitsandbytes: base ${ENVDIR}/lib/python3.11/site-packages/bitsandbytes 

geneformer: base ${ENVDIR}/lib/python3.11/site-packages/geneformer

scgpt: base ${ENVDIR}/lib/python3.11/site-packages/scgpt

teddy: base ${ENVDIR}/lib/python3.11/site-packages/teddy

scfoundation: base ${ENVDIR}/lib/python3.11/site-packages/scfoundation

nicheformer: base ${ENVDIR}/lib/python3.11/site-packages/nicheformer

ants: base ${ENVDIR}/bin/N4BiasFieldCorrection

fsl: base ${ENVDIR}/bin/flirt

dcm: base ${ENVDIR}/bin/dcm2niix

wgcna: base geneformer scgpt teddy scfoundation nicheformer ${ENVDIR}/lib/python3.11/site-packages/PyWGCNA

captum: base geneformer scgpt teddy scfoundation nicheformer wgcna ${ENVDIR}/lib/python3.11/site-packages/captum

${ENVDIR}/lib/python3.11/site-packages/torch:
	$(call cecho,I: Installing python base packages ...)
	@ENVNAME=${ENVDIR} bash setup/base.sh
	$(call cecho,I: Base packages installed.)

${ENVDIR}/lib/python3.11/site-packages/bitsandbytes:
	$(call cecho,I: Installing python bitsandbytes...)
	@ENVNAME=${ENVDIR} bash setup/bitsandbytes.sh
	$(call cecho,I: Bitsandbytes installed.)

${ENVDIR}/lib/python3.11/site-packages/geneformer:
	$(call cecho,I: Installing GeneFormer ...)
	@ENVNAME=${ENVDIR} bash setup/geneformer.sh
	$(call cecho,I: GeneFormer installed.)

${ENVDIR}/lib/python3.11/site-packages/scgpt:
	$(call cecho,I: Installing scGPT ...)
	@ENVNAME=${ENVDIR} bash setup/scgpt.sh
	$(call cecho,I: scGPT installed.)

${ENVDIR}/lib/python3.11/site-packages/teddy:
	$(call cecho,I: Installing TEDDY ...)
	@ENVNAME=${ENVDIR} bash setup/teddy.sh
	$(call cecho,I: TEDDY installed.)

${ENVDIR}/lib/python3.11/site-packages/scfoundation:
	$(call cecho,I: Installing scFoundation ...)
	@ENVNAME=${ENVDIR} bash setup/scfoundation.sh
	$(call cecho,I: scFoundation installed.)

${ENVDIR}/lib/python3.11/site-packages/nicheformer:
	$(call cecho,I: Installing NicheFormer ...)
	@ENVNAME=${ENVDIR} bash setup/nicheformer.sh
	$(call cecho,I: NicheFormer installed.)

${ENVDIR}/bin/N4BiasFieldCorrection:
	$(call cecho,I: Installing ANTs ...)
	@ENVNAME=${ENVDIR} bash setup/ants.sh
	$(call cecho,I: ANTs installed.)

${ENVDIR}/bin/flirt:
	$(call cecho,I: Installing FSL ...)
	echo ${ENVDIR}
	@ENVNAME=${ENVDIR} bash setup/fsl.sh ${ENVDIR}/_fsl
	$(call cecho,I: FSL installed.)

${ENVDIR}/bin/dcm2niix:
	$(call cecho,I: Installing DCM2NIIX ...)
	echo ${ENVDIR}
	@ENVNAME=${ENVDIR} bash setup/dcm.sh
	$(call cecho,I: DCM2NIIX installed.)

${ENVDIR}/lib/python3.11/site-packages/PyWGCNA:
	$(call cecho,I: Installing WGCNA ...)
	echo ${ENVDIR}
	@ENVNAME=${ENVDIR} bash setup/wgcna.sh
	$(call cecho,I: WGCNA installed.)

${ENVDIR}/lib/python3.11/site-packages/captum:
	$(call cecho,I: Installing captum package ...)
	echo ${ENVDIR}
	@ENVNAME=${ENVDIR} bash setup/captum.sh
	$(call cecho,I: captum package installed.)

clean:
	rm -rf ${ENVDIR}
