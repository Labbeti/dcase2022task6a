
<div align="center">

# IRIT-UPS DCASE 2022 TASK6A SYSTEM: STOCHASTIC DECODING METHODS FOR AUDIO CAPTIONING

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.9+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.10.1-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.6.2-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra 1.1-89b8cd?style=for-the-badge&labelColor=gray"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>


Automated Audio Captioning experiment source code on **Clotho** dataset for DCASE2022 task6a challenge.

</div>

## TLDR
**Installation with conda** :
```bash
git clone https://github.com/Labbeti/dcase2022task6a
cd dcase2022task6a
conda create -n env_task6a -f environment_full.yaml
conda activate env_task6a
pip install -e aac_datasets --no-dependencies
pip install -e . --no-dependencies
```
**Reproduce results of the submission** :
```bash
# Download & prepare Clotho
python -m dcase2022task6a.prepare
# Train a model on Clotho
python -m dcase2022task6a.train pl.beam_size=9
# Select the path where the training has saved data
logdir="/absolute/path/to/train/logdir"
# Test decoding methods
python -m dcase2022task6a.train trainer=test resume=${logdir} pl.top_k=4 pl.generator=1234
python -m dcase2022task6a.train trainer=test resume=${logdir} pl.top_p=0.3 pl.generator=1234
python -m dcase2022task6a.train trainer=test resume=${logdir} pl.typical_p=0.8 pl.generator=1234
```

## Installation details
This repository contains `environment_full.yaml` and a `requirements.txt` files for installing dependencies via conda or pip. 
The `environment_full.yaml` contains the exact same environment than used for development.

### External Requirements
- **Java >= 1.8.0** to compute the **SPICE** metric. (you can specify a path with `path.java` option)
	- On Ubuntu : `sudo apt install default-jre`
- **unzip** for extract the JAR file from the SPICE zip file.

### Dataset and models preparation
You can install the datasets with `python -m dcase2022task6a.prepare`. The default root path is `./data/`, but you can change it with the option `data.root=/my/root`.
You can choose a dataset with the option `data=DATASET`.
This script also install language models for NLTK, spaCy and LanguageTool to process captions and download PANN pre-trained models.

Example : (download basic models and Clotho v2.1)
```shell
python -m dcase2022task6a.prepare data=clotho
```

Note : Clotho is fast to install (several minutes), but AudioCaps requires to download and extract youtube audios with ffmpeg and can take several days.

### Other installation mode with pip
You can also use the `requirements.txt` file for install dependencies :
```bash
pip install https://github.com/Labbeti/dcase2022task6a
```

The installation is simplier and faster than with conda, but the packages like pytorch, torchaudio, etc will be installed from pip instead of conda, which means the results can be different.
The program can also be slower due to conda optimizations like numpy with MKL.

## Usage

### Example
```shell
python -m dcase2022task6a.train pl=cnn10_transformer data=clotho epochs=50
```
For training CNN10Transformer model with Clotho dataset during 50 epochs.
The testing is automatically done at the end of the training.

## External authors
- Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, Mark D. Plumbley for **PANN models** (CNN10, CNN14...)
	- Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, Mark D. Plumbley. "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition." arXiv preprint arXiv:1912.10211 (2019).
	- [source code](https://github.com/qiuqiangkong/audioset_tagging_cnn)
