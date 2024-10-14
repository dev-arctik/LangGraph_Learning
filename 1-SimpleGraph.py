from typing import TypedDict, Literal
import random

# for langGraph
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    graph_state: str


def node_1(state):
    print("---Node 1---")
    return {"graph_state": state['graph_state'] + " I am"}

def node_2(state):
    print("---Node 2---")
    return {"graph_state": state['graph_state'] + " happy :)  "}

def node_3(state):
    print("---Node 3---")
    return {"graph_state": state['graph_state'] + " Sad :(  "}


def decide_mood(state) -> Literal["node_2", "node_3"]:
    # 50 / 50 split between nodes 2 and 3
    if random.random() < 0.5:
        return "node_2"
    return "node_3"


# Build Graph
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Compile graph
graph = builder.compile()


# Save the graph image as a PNG file in the GraphImages directory
img_path = "./GraphImages/simplegraph_image.png"
with open(img_path, "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

print(f"Graph saved as '{img_path}'")

# Open and display the image using Pillow
from PIL import Image as PILImage
img = PILImage.open(img_path)

# To open the image
# img.show()


response = graph.invoke({"graph_state" : "Hi, this is Lance."})

print(response)