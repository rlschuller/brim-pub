# Breast MRI

*Work produced as part of the activities of the project "Rede de Inteligência Artificial para Diagnóstico por Imagem (RIADI), funded by a [FAPERJ](https://www.faperj.br) grant E-26/290.024/2021 (Edital de Inteligência Artificial)*.

*Participating institutions of RIADI:* [IMPA](https://impa.br), [Dasa](https://dasa.com.br), [LNCC](https://www.gov.br/lncc/pt-br), [UFRJ](https://ufrj.br) and [UniRio](https://www.unirio.br).

## Contributors

*Main contributors:*

* [Otávio Moreira (IMPA/Columbia U.)](https://stat.columbia.edu/people/name/otavio-moreira/), [Rodrigo Schuller (IMPA)](https://github.com/rlschuller), [Marcelo Carneiro (IMPA)](https://impa.br/alunos/): student participants of RIADI and main contributors to this repository.
* [Dr. Fernanda Philadelpho, MD (Dasa)](https://altadiagnosticos.com.br/dra-fernanda-philadelpho/), Brest radiologist, postdoctoral member of the RIADI project, medical lead of the Breast MRI subproject. 
* [Gustavo Trancoso](https://advdinamico.com.br/socios/gustavo-de-morais-trancoso-e443c38c) was in charge of image segmentations
* [Roberto Imbuzeiro Oliveira](https://sites.google.com/view/roboliv/) was the coordinator of RIADI.   

*Also acknowledged:* [Dr Heron Werner, MD (Dasa)](https://altadiagnosticos.com.br/dr-heron-werner-jr/), [Gérson Ribeiro (Dasa)](https://www.linkedin.com/in/gerson-ribeiro-0b65a439/?originalSubdomain=br)   

 ## About

This repository contains data and AI algorithms for segmentation and the analysis of findings of breast nuclear magnetic resonance exams. The main goal is to identify and distinguish findings in the [BIRADS 3/4 levels](https://radiopaedia.org/articles/breast-imaging-reporting-and-data-system-bi-rads-2). 

The *data*  consists of anonymized 3d bread MRI images from the practice of [Dr. Fernanda Philadelpho, MD](https://altadiagnosticos.com.br/dra-fernanda-philadelpho/) which contained findings evaluated at BIRADS 3/4 levels. Dr. Philadelpho is the coordinator of breast radiology for [Dasa](https://dasa.com.br) in Rio de Janeiro. Images were segmented and annotated by [Gustavo Trancoso](https://advdinamico.com.br/socios/gustavo-de-morais-trancoso-e443c38c) . 

The *algorithms* can be found in the `models/` folder at the link below; se subsequent sections for download and docker instructions. They were trained to reproduce the evaluations of Dr. Philadelpho (i.e. those are interpreted as "ground truth" for our purposes). In their current state, they produce a relatively large of false positives (ie. false classifications at the BIRADS 4 level). More study and (probably) more data are needed to obtain a method with better specificity.

*Data and algorithms are available under a Creative Commons License [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).*

## Download instructions

Clone the project, and place the `data/` and `models/` folders from [this link](https://drive.google.com/drive/folders/1T1AX0BR5ac2ylEmZmhvMFdpN8lwlELk-?usp=drive_link) under the root of the project.

## Docker instructions

To build the project, process the MRI exams and train the best performing model `src/models/DLPT/`, install docker with CUDA support and run

```
docker compose up
```

## Native setup instructions

### Installing pyenv and poetry

Install pyenv with
```sh
curl https://pyenv.run | bash
```

Add
```bash
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```
at the end of *~/.bashrc* to enable pyenv.

Install Poetry with
```sh
curl -sSL https://install.python-poetry.org | python3 -
```
Add
```bash
export PATH="$HOME/.local/bin:$PATH"
```
at the end of *~/.bashrc* to enable Poetry.

Restart bash session for the changes to take effect, or run
```sh
. ~/.bashrc
```

Install Python 3.10 and activate it run
```sh
pyenv install 3.10
```
If you get any errors, your system may be missing some packages. Try to fix that by installing:
```sh
sudo apt install \
    build-essential \
    curl \
    libbz2-dev \
    libffi-dev \
    liblzma-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    libxml2-dev \
    libxmlsec1-dev \
    llvm \
    make \
    tk-dev \
    wget \
    xz-utils \
    zlib1g-dev \
    libzbar0
```

### Running the pipeline

Execute the `./src/entrypoint.sh` script, from the root folder of the project.

