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

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    ```

3. **Activate the virtual environment**:

    - On macOS and Linux:
      ```bash
      source venv/bin/activate
      ```
      
    - On Windows:
      ```bash
      venv\Scripts\activate
      ```

4. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Setup

To run the code effectively, create a `.env` file in the root directory to store your API keys and configuration settings. This is necessary for using the various APIs involved in the project.

You will get the langsmith api (optional) on [langsmith Website](https://smith.langchain.com/)

Create a `.env` file and add the following lines:

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
python [file_name.py]
```

## Acknowledgments

I am learning from the [LangChain Academy](https://academy.langchain.com/).
