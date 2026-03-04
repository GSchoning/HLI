# HLI

Hierarchical Latent Inversion of Airborne Electromagnetic data.

This project uses SimPEG and Dask for probabilistic and deterministic inversion workflows for Airborne Electromagnetic (AEM) data.

## Project Structure

* `libraries/`: Contains the core Python package for data parsing (`gex_parser.py`, `des_parser.py`) and inversion tools (`SigNULL.py`, `HIES.py`, etc.).
* `notebooks/`: Jupyter Notebooks demonstrating workflows (`FINAL_WORKFLOW.ipynb`, `CONDAINE_EX.ipynb`, etc.).

## Setup

It is recommended to use Conda to manage your dependencies. You can create the required environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate hli_env
```

After activating the environment, you can run the provided Jupyter notebooks by starting Jupyter from the project root:

```bash
jupyter notebook
```

The notebooks are configured to append the parent directory to the Python path, ensuring that the `libraries` package can be imported correctly from within the `notebooks` directory.
