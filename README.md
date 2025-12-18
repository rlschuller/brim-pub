# Breast MRI

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

