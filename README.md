# Deepseek Experiments

Before continuing, we first have to install deepseek through `ollama`.
For a detailed instruction on how to install `ollama`, please go to the corresponding webpage: https://ollama.com

And start `ollama`:
```bash
ollama serve
```

Next, we need to install and run ollama using the deepseek-r1 (7B) model:
```bash
ollama pull deepseek-r1
```

If you want to check out a different model, you have to specify the number of parameters as an additional element of the model name, e.g. trying the 1.5B parameter variant:
```bash
ollama pull deepseek-r1:1.5b
```

To check if everything works as expected, you can try the model in the terminal:
```bash
ollama run deepseek-r1:1.5b
```

## Virtual Environment

Typically, we use `poetry` and `pdm`, so here we can use pdm as well to create a virtual environment for us like:
```bash
pdm venv create
pdm use # If not already set correctly, choose the one in the current pwd
pdm venv activate
```

## More requirements

Install all remaining extra packages:
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
