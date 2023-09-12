### The LLM Chatbot service

## Prerequisites
The application requires Python 3.10
Here's a link to download 3.10.5 https://www.python.org/downloads/release/python-3105/

# Running Locally

## Downloading models

On application start, the app will attempt to download the models for your current environment (defaulted to `local`). The models for each environment can be found within the `src/config.py` file. If the app determines that the requested model already exists, it will attempt to load the model into memory. **WARNING** Laptops will only be able to run `*.GGML` models due to system memory constraints.

You can download different models by updating the `download_link` field for a given `EnvrionmentConfiguration`. An example for how to retrieve the `download_link` for a given model is shown in the video below:

![download_link_example](ReadMe_Model_URL_Example.mov)

All models will be downloaded to `/src/model/downloaded_models/{folder_name}/{quantization_model}`.

## Application Standalone

1. Setup a virtual env
   1. `pip install virtualenv`
   2. `python3 -m venv copilot`
   3. `source copilot/bin/activate`
2. `pip install wheel`
3. `pip install -r requirements.txt`
4. Run `uvicorn src.main:app --reload` to start the app
   * The first time you run this it will attempt to download the model. This means it can take up to 15 minutes for the app to startup depending on internet speeds.
5. Chat with the app! `curl --request POST \
  --url http://127.0.0.1:8000/api/v1/inference \
  --header 'Content-Type: application/json' \
  --data '{
	"query": "Write a short story about Grace Hopper"
}'`
# Troubleshooting

## PIP Install Errors for hnswlib

https://github.com/pypa/packaging-problems/issues/648#issuecomment-1564437323

`export HNSWLIB_NO_NATIVE=1` and then run `pip install -r requirements.txt`


## PIP Install Errors for ChromaDb

If you run into the issue
`ERROR: Could not build wheels for chroma-hnswlib, which is required to install pyproject.toml-based projects`
Per [stackoverflow link](https://stackoverflow.com/questions/73969269/error-could-not-build-wheels-for-hnswlib-which-is-required-to-install-pyprojec)

First, run `export HNSWLIB_NO_NATIVE=1`
Then run `pip install chromadb

## Local Docker Build Out of Memory

This can be attributed to Docker not removing unused images. You can run `docker system prune -a -f` to free the necessary memory.
