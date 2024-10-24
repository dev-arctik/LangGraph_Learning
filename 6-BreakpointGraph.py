# Breakpoints are a simple way to stop the graph to ask for approval before moving to the next nodes

import os
import dotenv

from langchain_openai import ChatOpenAI

# to build graph
from langgraph.graph import StateGraph, START, MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import tools_condition, ToolNode

# To create image
from utils.graph_img_generation import save_and_show_graph

# for printing messages
from langchain_core.messages import HumanMessage

# for adding checkpoint in memory
from langgraph.checkpoint.memory import MemorySaver

# loading the env file
dotenv.load_dotenv()

# storing the API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



# defining the LLM
llm = ChatOpenAI(model = "gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

# defining a memory location
memory = MemorySaver()



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

# Compile graph with memory
breakpoint_graph = builder.compile(interrupt_before=["tools"], checkpointer=memory)


# Use the utility function to save and optionally show the graph
save_and_show_graph(breakpoint_graph, filename="BreakpointGraph_image", show_image=False)



# Specify a thread AKA session
config = {"configurable": {"thread_id": "1"}}

# Start the conversation loop
while True:

    # Take user input
    user_msg = input("Enter your message (or type 'exit' to quit): ")

    # Break the loop if the user types 'exit'
    if user_msg.lower() == 'exit':
        print("Exiting the session...")
        break

    # Input
    initial_input = {"messages": HumanMessage(content=user_msg)}

    # Run the graph until the first interruption
    for event in breakpoint_graph.stream(initial_input, config, stream_mode="values"):
        event['messages'][-1].pretty_print()

    # Get user feedback
    user_approval = input("Do you want to call the tool? (yes/no): ")

    # Check approval
    if user_approval.lower() == "yes":
        
        # If approved, continue the graph execution
        for event in breakpoint_graph.stream(None, config, stream_mode="values"):
            event['messages'][-1].pretty_print()
            
    else:
        print("Operation cancelled by user.")
