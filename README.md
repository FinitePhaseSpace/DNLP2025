# DNLP2025

## Setup

Install poetry https://python-poetry.org/

Run "poetry install" to install dependencies and setup the venv.

To add poetry to the path (if it's not added automatically) run:

```
echo 'export PATH="/home/jovyan/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

PyCharm Venv:

```
Adding the newly generated Poetry venv in PyCharm: https://www.jetbrains.com/help/pycharm/poetry.html#existing-poetry-environment
```

## Scripts

To download datasets and train the tokenizers run:

```
poetry run train-tokenizers
```

## Running Tests

```
poetry run pytest tests
```
