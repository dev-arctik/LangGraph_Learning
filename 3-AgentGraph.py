import os
import dotenv

from langchain_openai import ChatOpenAI

# to build graph
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

# To create image
from utils.graph_img_generation import save_and_show_graph

# for printing messages
from langchain_core.messages import HumanMessage

# loading the env file
dotenv.load_dotenv()

# storing the API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



# defining the LLM
llm = ChatOpenAI(model = "gpt-4o-mini",openai_api_key=OPENAI_API_KEY)



# defining the tools
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def subtract(a: int, b: int) -> int:
    """Subtract a and b.

    Args:
        a: first int
        b: second int
    """
    return a - b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, subtract, multiply, divide]


# For this ipynb we set parallel tool calling to false as math generally is done sequentially, and this time we have 3 tools that can do math
# the OpenAI model specifically defaults to parallel tool calling for efficiency, see https://python.langchain.com/docs/how_to/tool_calling_parallel/
# binding tools with llm
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)




# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}



# Build the graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")
agent_graph = builder.compile()


# Use the utility function to save and optionally show the graph
save_and_show_graph(agent_graph, filename="agentgraph_image", show_image=False)



# To run the Graphs
user_msg = "7+3+4-2."
messages = [HumanMessage(content=user_msg)]
messages = agent_graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()