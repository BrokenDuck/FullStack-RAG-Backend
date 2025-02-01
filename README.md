# AI Chatbot

This git submodule contains the backend for the **Retrieval-Augmented Generation (RAG) full-stack chatbot application**. The backend is build with **Python, FastAPI, Weaviate and Cohere's api models**.

The main components of the backend are:
- A fully asynchronous python REST API
    - The api exposes endpoints to embed pdfs and perform RAG queries on the pdfs
    - The api performs fully asynchronous calls to deployed Cohere models and the database. This allows for a significant performance increase as the python GIL is released during database queries and api calls.
- A vector database to perform efficient hybrid queries on the pdfs chunks

## Structure

- `app/`: Contains the Python FastAPI backend code and docker files to run the backend.
- `data/`: Contains a set of sample pdfs to use in the application
- `notebooks/`: Contains a set of notebooks to detail usage of Weaviate and Cohere

## REST API Endpoints

We define three endpoints for the REST API:
- `/query`: allows the client to ask a question which will be answered and grounded in a set of documents retrieved from the database, the response is sent in one large JSON. The endpoint supports selection between simple and advanced retrieval techniques from the database.
- `/stream`: allows the client to ask a question which will be answered and grounded in a set of documents retrieved from the database, the response is streamed using server-sent events. The endpoint supports selection between simple and advanced retrieval techniques from the database.
- `/uploadfiles`: allows the client to upload multiple files using the FormData api. Every file will be chunked, embedded and uploaded to the database.

## Instalation

### Requirements

A [docker](https://www.docker.com/) instalation.

### Running the backend

1. Create a `.env` file in the backend folder copying the `.env-sample` file provided and set the required environment variable:
    - `COMMAND_R_URL`: Your Cohere chat completion model endpoint url
    - `COMMAND_R_API_KEY`: Your Cohere chat completion model api key
    - `EMBED_ENGLISH_URL`: Your Cohere text and image embedding model url
    - `EMBED_ENGLISH_API_KEY`: Your Cohere text and image embedding model api key
    - (Optional) `RERANK_ENGLISH_URL`: Your cohere rerank model endpoint url (needed for more advanced document retrieval strategies)
    - (Optional) `RERANK_ENGLISH_API_KEY`: Your cohere rerank model api key (needed for more advanced document retrieval strategies)

2. Build the backend docker image:
    ```bash
    docker build -t backend-image .
    ```

3. Run docker compose to run the backend:
    ```bash
    docker compose up -d
    ```

4. Create a python virtual environment and install the dependencies:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

4. Define the object collection in the weaviate database. You should look at the database notebook at `notebooks/database.ipynb`.

5. You are all set. To see a full description of the backend api, including endpoint request and response formats, go to `http://127.0.0.1:8000/docs`. A full openapi specification (that can be imported into Postman) can be found in `specification.json`.

## Developing the backend

The backend has the following file structure:

```
.
├── config.py
├── main.py
├── models.py
├── query.py
├── statics
│   ├── prompts.py
│   └── tools.py
├── tests
│   └── test_main.py
└── upload.py
```

Here is a brief description of every file and it's purpose:
- `config.py`: reads environment variable using pydantic settings and instatiates Cohere and Weaviate clients
- `main.py`: main entry point of the REST api, defines the endpoints and the CORS middleware for security
- `models.py`: defines the pydantic models (JSON formats) for the POST request and response payloads
- `query.py`: defines the python async functions to query the database and generate the answer with the cohere chat completion model
- `upload.py`: defines the python async functions to chunk pdfs, embed the chunks and upload them to the database
- `static/prompts.py`: defines the prompts for the chat completion model (can be fine tuned for prompt engineering)
- `static/tools.py`: defines the tools for the advanced embeddings retrieval models (only required for advanced retrieval)
- `tests/test_main.py`: defines a basic test to check the root api endpoint

### FastAPI REST API

After modifying the files, run the FastAPI in development mode:
```bash
fastapi dev app/main.py
```

To run the test, first install pytest and then run the tests
```bash
pip install pytest httpx pytest-asyncio
pytest app/tests/
```

### Weaviate Database

To change the settings of the Weaviate database, modify the `docker-compose.yml` file. Afterwards, rebuild and run the docker containers with:
```bash
docker compose up --build -d
```
