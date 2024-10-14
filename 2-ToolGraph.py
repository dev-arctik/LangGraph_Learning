from langchain_openai import ChatOpenAI

# to build graph
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

# To create image
from utils.graph_img_generation import save_and_show_graph

# for printing messages
from langchain_core.messages import HumanMessage

# tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

llm = ChatOpenAI()
llm_with_tools = llm.bind_tools([multiply])


# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", END)
graph = builder.compile()


# Save the graph image as a PNG file in the GraphImages directory
# Use the utility function to save and optionally show the graph
save_and_show_graph(graph, filename="toolgraph_image", show_image=False)

user_msg = "multiply 2 and 3 and 6."
messages = [HumanMessage(content=user_msg)]
messages = graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()