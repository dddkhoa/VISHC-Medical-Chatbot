# Vishc Medical ChatBot 🤖
Note: Thank you for your interest in our application.
        Please be aware that this is only a Proof of Concept system
        and may contain bugs or unfinished features.
## Demo Pipeline
1. Vector Database: https://weaviate.io/
2. Embedding: https://huggingface.co/kaiserrr/Bilingual-BioSimCSE-BioLinkBERT-base
3. Retriever: Hybrid Search (BM25+Semantic Search, alpha=0.5)
4. Generator: gpt-turbo-3.5 (https://platform.openai.com/docs/models/gpt-3-5)

## Requirements

- Python 3.10
- Docker

## Installation

### Set up virtual environment

```shell
python -m venv venv
. venv/bin/activate
```

### Install dependencies

- Install poetry: https://python-poetry.org/
- Install dependencies

```shell
poetry install
```

### Install `pre-commit` hooks (for development only)

```shell
pre-commit install
```

## Running

### Set up Weaviate Vector Database locally with Docker
Currently the Docker instance is using CPU-only for embeddings,
which is quite slow. To enhance the performance, you can
edit the `docker-compose.yml` file to use GPU as follow:
```shell
t2v-transformers:
  image: vishc-medbot-inference
  environment:
    ENABLE_CUDA: '1'  # Set 0 to use CPU-only
    NVIDIA_VISIBLE_DEVICES: 'all'
```
Then run the following:
```shell
docker build -f vishcMedBot.Dockerfile -t vishc-medbot-inference .  # Build and use the finetuned embedding
docker-compose up -d  # Start Weaviate instance
```

### Insert API keys
Create a file called `.env` in the parent folder and 
add the following keys
```shell
OPENAI_API_KEY="your-openai-api-key-here"
HUGGINGFACEHUB_API_TOKEN = "your-huggingface-hub-api-key-here"
```

### Interactive Chat with Streamlit

Inside the virtual environment, run

```shell
streamlit run run_streamlit.py
```
There's a sample PDF file in `docs/` for testing purpose.