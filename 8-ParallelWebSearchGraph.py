# Import necessary libraries
import os
import dotenv
from langchain_openai import ChatOpenAI
from typing import Annotated, TypedDict
import operator

# LangGraph components to build the graph
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults

# Import custom function for image saving and display
from utils.graph_img_generation import save_and_show_graph

# Load environment variables from a .env file
dotenv.load_dotenv()

# Retrieve the API keys for OpenAI and Tavily from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Initialize the OpenAI language model with parameters
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0)

# Define the structure of the state dictionary
class State(TypedDict):
    question: str
    answer: str
    context: Annotated[list, operator.add]

# Define the function to retrieve documents from the web
def search_web(state):
    """Retrieve search results from the web using TavilySearchResults."""
    tavily_search = TavilySearchResults(max_results=3)  # Initialize Tavily search
    search_docs = tavily_search.invoke(state['question'])  # Retrieve documents
    
    # Format search results for readability
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    # Update state with the search results
    return {"context": [formatted_search_docs]} 

# Define the function to retrieve documents from Wikipedia
def search_wikipedia(state):
    """Retrieve search results from Wikipedia."""
    search_docs = WikipediaLoader(query=state['question'], load_max_docs=2).load()  # Load Wikipedia docs
    
    # Format Wikipedia results for readability
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    # Update state with the Wikipedia results
    return {"context": [formatted_search_docs]} 

# Define the function to generate an answer based on context
def generate_answer(state):
    """Generate an answer based on the accumulated context."""
    context = state["context"]
    question = state["question"]

    # Template for the AI to generate an answer
    answer_template = """Answer the question {question} using this context: {context}"""
    answer_instructions = answer_template.format(question=question, context=context)
    
    # Generate the answer by sending the question and context to the language model
    answer = llm.invoke([SystemMessage(content=answer_instructions)]+[HumanMessage(content="Answer the question.")])
    
    # Update state with the generated answer
    return {"answer": answer}

# Initialize the StateGraph and define each node
builder = StateGraph(State)

# Add nodes for parallel execution
builder.add_node("search_web", search_web)  # Node for web search
builder.add_node("search_wikipedia", search_wikipedia)  # Node for Wikipedia search
builder.add_node("generate_answer", generate_answer)  # Node to generate an answer

# Define the flow: 
# Start the search processes in parallel, then direct both outputs to the answer generator
builder.add_edge(START, "search_wikipedia")
builder.add_edge(START, "search_web")
builder.add_edge("search_wikipedia", "generate_answer")
builder.add_edge("search_web", "generate_answer")
builder.add_edge("generate_answer", END)

# Compile the graph to create the parallel execution structure
parallel_websearch_graph = builder.compile()

# Use the utility function to save and optionally show the generated graph
save_and_show_graph(parallel_websearch_graph, filename="Parallel_WebSearchGraph_image", show_image=False)

# test the graph
result = parallel_websearch_graph.invoke({"question": "Tell me about hindu mythology and Chakra"})
print(result['answer'].content)