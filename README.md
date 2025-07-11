# LangGraph Learning

This project documents my learning journey with LangGraph, featuring incremental updates saved in separate files.

## Table of Contents

- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

## Installation

To get started with this project, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/dev-arctik/LangGraph_Learning.git
    cd LangGraph_Learning
    ```

2. **Install Poetry** (if not already installed):
    ```bash
    # On macOS and Linux
    curl -sSL https://install.python-poetry.org | python3 -
    
    # On Windows (PowerShell)
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
    ```

3. **Install dependencies using Poetry**:
    ```bash
    poetry install
    ```

4. **Activate the Poetry virtual environment**:
    ```bash
    poetry shell
    ```

## Setup

To run the code effectively, create a `.env` file in the root directory to store your API keys and configuration settings. This is necessary for using the various APIs involved in the project.

You can use the provided `.env.example` file as a template:

```bash
cp .env.example .env
```

Then edit the `.env` file with your actual API keys.

You will get the langsmith api (optional) on [langsmith Website](https://smith.langchain.com/)

The `.env` file should contain the following variables:

```plaintext
# OpenAI
OPENAI_API_KEY="your_openai_api_key_here"

# LangSmith
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your_langchain_api_key_here"
LANGCHAIN_PROJECT="your_project_name_here"

# Tavily
TAVILY_API_KEY="your_tavily_api_key_here"
```
Replace 'your_openai_api_key_here' with your actual OpenAI API key.

Note: If you choose to use a different LLM (Language Learning Model), refer to the LangChain documentation and adjust the code as needed.


## Usage

You can run individual files with the following command:
```bash
poetry run python [file_name.py]
```

Alternatively, if you have activated the Poetry shell:
```bash
python [file_name.py]
```

## Acknowledgments

I am learning from the [LangChain Academy](https://academy.langchain.com/) and Blogs by langchain team. Special thanks to the LangChain community for their support and resources.
