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
   2. `python3 -m venv grace-hopper`
   3. `source grace-hopper/bin/activate`
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



# Conceptual Overview

This sample project leverages a collection of open source frameworks, APIs, and programming concepts which may be unfamiliar to some. Below are some explanations of the elements of the solution:

## The Retrieval Augmented Generation (RAG) Pattern

The architectural pattern in which domain-specific context is used to populate a searchable data store - and which is then retrieved on-demand by a retrieval mechanism before sending to a Large Language Model - is a common pattern.  Read more about it [here](https://towardsdatascience.com/build-industry-specific-llms-using-retrieval-augmented-generation-af9e98bb6f68)

## Use of Python for AI Applications

Python is the primary language of choice for developing applications that leverage Machine Learning / AI.  There are a plethora of libraries as fundamental as [numpy](https://numpy.org), [SciPy](https://scipy.org), [scikit-learn](https://scikit-learn.org/stable/), [pytorch](https://pytorch.org) and more recently [langchain](https://python.langchain.com/docs/get_started/introduction.html) and [huggingface](https://huggingface.co/docs/hub/models-libraries).  The variety of available libraries and the power of the capabilities they expose in accomplishing AI-related tasks is unparalleled (regardless of your feelings about whitespace) and thus python is a common choice for rapidly building AI-powered apps.  

This application is built in python, and uses many libraries (found in `requirements.txt`). The following libraries are key enablers of most LLM-based applications, and are used by this project:

- [Huggingface libraries](https://huggingface.co/docs/hub/models-libraries)
- [Langchain, for LLM orchestration, prompt engineering and retrieval support](https://python.langchain.com/docs/get_started/introduction.html)
- [FastAPI, for rapid web-based API development in python](https://fastapi.tiangolo.com)
- [Chroma, a lightweight vector database](https://www.trychroma.com)

In addition, the project uses many of the python language's features to accomplish tasks.  If interested, feel free to read more about some core python concepts:

- [Modules / how code is organized for imports](https://docs.python.org/3/tutorial/modules.html)
- [pip's pyproject.toml](https://pip.pypa.io/en/stable/reference/build-system/pyproject-toml/)
- [pytest, for unit tests](https://www.tutorialspoint.com/pytest/pytest_introduction.htm)
- [Python decorators, and their use in application startup](https://realpython.com/primer-on-python-decorators/)

## Use of Docker for Application Deployment / Delivery

Docker is a framework for wrapping up an application and its runtime requirements, and orchestrating its deployment.  Docker's "Containerized App" allows an application to be wrapped up in a singular object that can be run anywhere Docker can run - regardless of implementation language - in a way that allows the app to be replicated, load-balanced, and more easily self-healing.  Most modern apps leverage containers in their back-end.  Read more about Docker [here](https://docs.docker.com/get-started/overview/).
