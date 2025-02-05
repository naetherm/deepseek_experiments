# Deepseek Model Experiments

This guide explains how to set up and run experiments with the Deepseek language model using Ollama.

**Note**: As this is a personal playground it, and the direction it takes, can shift any time.

## Prerequisites

### Installing Ollama
1. First, install Ollama by following the official installation guide at [ollama.com](https://ollama.com)
2. Start the Ollama service:
```bash
ollama serve
```

### Setting up Deepseek
Install the Deepseek model through Ollama. The default version uses the 7B parameter model:
```bash
ollama pull deepseek-r1
```

For different model sizes, specify the parameter count in the model name. For example, to use the 1.5B parameter variant:
```bash
ollama pull deepseek-r1:1.5b
```

To verify your installation, test the model in the terminal:
```bash
ollama run deepseek-r1:1.5b
```

## Environment Setup

### Creating a Virtual Environment
This project uses PDM for dependency management. Follow these steps to set up your environment:

```bash
# Create a new virtual environment
pdm venv create

# Select and activate the virtual environment
pdm use
pdm venv activate
```

### Installing Dependencies
Install all required packages:
```bash
pdm install
pdm sync
```

## Running

As this is highly driven by `streamlit`, a lot of examples are streamlit applications.
So, e.g. to start the document-qa example you can now call:
```bash
poetry run streamlit run deepseek-experiment/app_document_rag.py
```
for starting the corresponding app. 

## Examples

### Document (PDF) QA System

```bash
poetry run streamlit run deepseek-experiment/app_document_rag.py
```

### Simple Chatbot

```bash
poetry run streamlit run deepseek_experiments/app_chat.py
```

### Chatbot with System Prompt for Python Development

```bash
poetry run streamlit run deepseek_experiments/app_python_coder.py
```