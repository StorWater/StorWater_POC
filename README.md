# Water Consumption Data Hack



## Quickstart

1. Check that Python is installed with `python -V` (I use `Python 3.7.6`). If not, [download Python](https://www.python.org/downloads/release/python-376/) and install. Mostly likely works with different versions, I could not install shap with 3.8
2. Clone/download project: `git clone https://github.com/StorWater/water_hackathlon.git .`. Go to project location `cd water_hackathlon`
3. Create new virtual environment: `make create_environment` or `python -m venv .venv_waterhack`
4. Activate it: `source .\.venv_waterhack\Scripts\activate`
5. Install packages with `make requirements` or `pip install -r requirements.txt`
6. Ready to go! Launch juptter lab: `jupyter lab`, open via [http://localhost:8888/](http://localhost:8888/)



## Project Organization

(to be updated)


    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   └── raw            <- The original, immutable data dump.
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── waterhack                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module      
    │   ├── make_dataset.py
    │   ├── build_features.py
    │   ├── predict_model.py
    │   ├── train_model.py
    │   └── visualize.py <- Scripts to create exploratory and results oriented visualizations


## Setup of jupyter notebooks

This is the configuration of the extensions I am using:

```
set NODE_OPTIONS=--max-old-space-size=4096
jupyter labextension install jupyterlab-plotly@4.9 --no-build
jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.9 --no-build
jupyter labextension install @ryantam626/jupyterlab_code_formatter --no-build
pip install jupyterlab_code_formatter
jupyter serverextension enable --py jupyterlab_code_formatter --no-build
jupyter lab build
```

