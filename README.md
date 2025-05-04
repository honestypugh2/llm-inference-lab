# LLM Inference: Introduction to Optimization and Efficiency Lab

This lab is designed to introduce participants to techniques for optimizing and improving the efficiency of large language model (LLM) inference. It covers practical methods for reducing latency, improving throughput, and leveraging tools and frameworks to achieve efficient inference.

## Objectives

- Understand the basics of LLM inference and its challenges.
- Learn optimization techniques for LLM inference, including quantization and distillation.
- Have the ability to explore tools like Ray, ONNX and Hugging Face Optimum for efficient inference.
- Set up and run inference pipelines in various environments, including GitHub Codespaces and local setups.

## Prerequisites
Software dependencies to support the work.

- Python 3.11: some package dependencies used in our work may not go beyond 3.11
- Install [vscode](https://code.visualstudio.com/)
- Install [pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation) 
- Install [poetry](https://python-poetry.org/docs/#installation)

Here is a really great Medium article on setup with pyenv and poetry: [Proper Python setup with pyenv & Poetry](https://douwevandermeij.medium.com/proper-python-setup-with-pyenv-poetry-4d8baea329a8)

## Code Pre-requisites

Ensure the following dependencies are installed in your environment:

- Python 3.11 or higher
- `pandas` (v2.2.3)
- `polars` (v1.29.0)
- `torch` (v2.7.0)
- `transformers` (v4.51.3)
- `datasets` (v3.5.1)
- `onnx` (v1.17.0)
- `optimum` (v1.24.0)
- `ipykernel` (v6.29.5)
- `huggingface-hub` (v0.30.2)

These dependencies are already listed in the `pyproject.toml` file and can be installed using Poetry.

## Getting Started

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd llm-inference-lab
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Activate the environment:
   ```bash
   poetry shell
   ```

4. Open the Jupyter Notebook: ~/llm_inference_lab/notebooks/inference_optimize.ipynb

5. In Jupyter Notebook, select Kernel to be the poetry virtual environment that should have been created following Step 2 located at `/llm-inference-lab/.venv/bin/python`.

## Environment Setup

### GitHub Codespaces

1. Open the repository in GitHub Codespaces.
2. Ensure the `.devcontainer` configuration is present to set up the environment automatically.
3. Run the Jupyter Notebook from the terminal:
   ```bash
   jupyter notebook notebooks/inference_optimize.ipynb
   ```

### VSCode Remote Containers

1. Open the repository in VSCode.
2. Install the "Remote - Containers" extension.
3. Reopen the folder in the container by selecting `Remote-Containers: Reopen in Container` from the Command Palette.
4. Run the Jupyter Notebook as described in the "Getting Started" section.

### Local Setup

1. Ensure Python 3.11 or higher is installed on your system.
2. Install Poetry:
   ```bash
   pip install poetry
   ```
3. Follow the steps in the "Getting Started" section to set up the environment and run the notebook.

### Additional VSCode Extensions:
- Jupyter
- Jupyter Cell Tags
- Jupyter Notebook Renderers

With these steps, you are ready to explore optimization techniques for LLM inference!
