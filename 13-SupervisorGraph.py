# 14-SupervisorGraph.py
# We will create a graph with a supervisor node which will direct the user to the required node

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langgraph.graph.message import add_messages, AnyMessage
from typing import TypedDict, Annotated, List, Literal, Any
from langgraph.checkpoint.memory import MemorySaver

from config.secret_keys import OPENAI_API_KEY
from config.config import get_llm

from utils.graph_img_generation import save_and_show_graph

# define LLM
llm = get_llm()

# define Custom State
class CustomState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    next_node: str

# define Model for structured output
class SupervisorModel(BaseModel):
    next_node: Literal['ASSISTANT', 'MATH_EXPERT', 'SCIENCE_EXPERT', 'HISTORY_EXPERT'] = Field(
        ...,
        description="The next node to which the user should be directed. It can be 'ASSISTANT', 'MATH_EXPERT', 'SCIENCE_EXPERT', or 'HISTORY_EXPERT'.",
    )

# define math tools
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def add(a: int, b: int) -> int:
    """Add a and b.

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

def divide(a: int, b: int) -> float:  # Fixed: should return float
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

tools = [add, subtract, multiply, divide]

# define NODES

# SUPERVISOR NODE
def supervisor(state):
    """
    Supervisor node that directs the user to Math Expert, Science Expert, or History Expert.
    """
    print("----------INSIDE SUPERVISOR----------")

    supervisor_prompt = """
    You are an intelligent routing supervisor responsible for directing users to the most appropriate expert based on their question.

    Analyze the user's message and determine which expert can best help them:

    - MATH_EXPERT: Choose for mathematical calculations, equations, algebra, geometry, statistics, calculus, or any numerical problem-solving
    - SCIENCE_EXPERT: Choose for physics, chemistry, biology, earth sciences, astronomy, or scientific concepts and explanations
    - HISTORY_EXPERT: Choose for historical events, dates, civilizations, wars, historical figures, or cultural history
    - ASSISTANT: Choose for general questions, greetings, or topics that don't clearly fit the other categories

    Consider the primary focus of the question. If a question touches multiple areas, route to the most relevant expert.
    """

    messages = [SystemMessage(content=supervisor_prompt)] + state["messages"]

    llm_with_structured_output = llm.with_structured_output(SupervisorModel)

    response = llm_with_structured_output.invoke(messages)

    print("Supervisor response:", response)

    return {
        **state,
        "next_node": response.next_node,
    }

def supervisor_router(state):
    print("----------INSIDE SUPERVISOR ROUTER----------")
    next_node = state["next_node"]

    valid_nodes = ["ASSISTANT", "MATH_EXPERT", "SCIENCE_EXPERT", "HISTORY_EXPERT"]
    if next_node not in valid_nodes:
        # default to the assistant if the next node is not valid
        print(f"Invalid next node '{next_node}'. Defaulting to 'ASSISTANT'.")
        next_node = "ASSISTANT"
    return next_node

# ASSISTANT NODE
def assistant(state):
    """
    Assistant node that provides general assistance.
    """
    print("----------INSIDE ASSISTANT----------")

    assistant_prompt = """
    You are a helpful general assistant. You provide clear, informative responses to a wide range of questions.

    Your role is to:
    - Answer general knowledge questions
    - Provide helpful explanations on various topics
    - Assist with everyday questions and tasks
    - Offer guidance when users need general help

    You can help with:
    - General knowledge inquiries
    - Math problems (basic)
    - Science questions (basic)
    - History questions (basic)

    Be friendly, concise, and helpful. If a question requires specialized expertise in math, science, or history, 
    let the user know they might want to ask about that specific topic to get more detailed help.

    Your response should be short and friendly, encouraging users to ask more questions if they need further assistance.
    """

    messages = [SystemMessage(content=assistant_prompt)] + state["messages"]

    response = llm.invoke(messages)

    return {
        "messages": response
    }

# MATH EXPERT NODE
def math_expert(state):
    """
    Math expert node that provides answers to math-related questions.
    """
    print("----------INSIDE MATH EXPERT----------")

    math_prompt = """
    You are a specialized mathematics expert with access to calculation tools. You excel at solving mathematical problems and explaining mathematical concepts.

    Your capabilities include:
    - Solving arithmetic problems (addition, subtraction, multiplication, division)
    - Explaining mathematical concepts and procedures
    - Working through step-by-step solutions
    - Helping with algebra, geometry, statistics, and other math topics

    Available tools:
    - add(a, b): Add two numbers
    - subtract(a, b): Subtract two numbers  
    - multiply(a, b): Multiply two numbers
    - divide(a, b): Divide two numbers

    When solving problems:
    1. Break down complex problems into steps
    2. Use the available tools for calculations when needed
    3. Show your work and explain your reasoning
    4. Provide clear, accurate answers with explanations

    Always use the tools for calculations to ensure accuracy, even for simple operations.

    Your response should be short, clear, and educational, encouraging users to ask follow-up questions if they need further assistance.
    """

    messages = [SystemMessage(content=math_prompt)] + state["messages"]

    llm_with_tools = llm.bind_tools(tools)

    response = llm_with_tools.invoke(messages)

    return {
        "messages": response
    }

# SCIENCE EXPERT NODE
def science_expert(state):
    """
    Science expert node that provides answers to science-related questions.
    """
    print("----------INSIDE SCIENCE EXPERT----------")

    science_prompt = """
    You are a knowledgeable science expert specializing in multiple scientific disciplines including physics, chemistry, biology, earth sciences, and astronomy.

    Your expertise covers:
    - Physics: mechanics, thermodynamics, electromagnetism, quantum physics, relativity
    - Chemistry: atomic structure, chemical reactions, organic/inorganic chemistry, biochemistry
    - Biology: cell biology, genetics, evolution, ecology, human anatomy and physiology
    - Earth Sciences: geology, meteorology, oceanography, environmental science
    - Astronomy: solar system, stars, galaxies, cosmology

    When answering questions:
    - Provide scientifically accurate information
    - Explain complex concepts in an understandable way
    - Use examples and analogies when helpful
    - Cite scientific principles and laws when relevant
    - Encourage scientific thinking and curiosity

    Make your explanations clear and educational, adapting to the user's apparent level of scientific background.

    Your response should be short, clear, and educational, encouraging users to ask follow-up questions if they need further assistance.
    """

    messages = [SystemMessage(content=science_prompt)] + state["messages"]

    response = llm.invoke(messages)

    return {
        "messages": response
    }

# HISTORY EXPERT NODE
def history_expert(state):
    """
    History expert node that provides answers to history-related questions.
    """
    print("----------INSIDE HISTORY EXPERT----------")

    history_prompt = """
    You are a comprehensive history expert with deep knowledge spanning all periods of human history and various civilizations.

    Your expertise includes:
    - Ancient civilizations (Egypt, Greece, Rome, Mesopotamia, etc.)
    - Medieval history and the Middle Ages
    - Renaissance and Early Modern periods
    - Modern history (18th-20th centuries)
    - World wars and major conflicts
    - Political, social, and cultural history
    - Historical figures and their contributions
    - Historical events and their significance

    When answering historical questions:
    - Provide accurate dates, names, and events
    - Explain the context and significance of historical events
    - Draw connections between past and present when relevant
    - Present multiple perspectives when appropriate
    - Use engaging storytelling while maintaining historical accuracy
    - Cite important sources or acknowledge when information is debated among historians

    Make history come alive by explaining not just what happened, but why it matters and how it shaped the world.

    Your response should be short, clear, and educational, encouraging users to ask follow-up questions if they need further assistance.
    """

    messages = [SystemMessage(content=history_prompt)] + state["messages"]

    response = llm.invoke(messages)

    return {
        "messages": response
    }

# Build the graph
builder = StateGraph(CustomState)

builder.add_node("supervisor", supervisor)
builder.add_node("assistant", assistant)
builder.add_node("math_expert", math_expert)
builder.add_node("math_expert_tools", ToolNode(tools))
builder.add_node("science_expert", science_expert)
builder.add_node("history_expert", history_expert)

builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    supervisor_router, 
    {
        'ASSISTANT': "assistant",
        'MATH_EXPERT': "math_expert",
        'SCIENCE_EXPERT': "science_expert",
        'HISTORY_EXPERT': "history_expert"
    }
)

builder.add_conditional_edges(
    "math_expert",
    tools_condition, {
        "tools": "math_expert_tools",  # Map "tools" to your tool node name
        "__end__": END                 # Map "__end__" to END
    }
)
builder.add_edge("math_expert_tools", "math_expert")

builder.add_edge("assistant", END)
builder.add_edge("math_expert", END)
builder.add_edge("science_expert", END)
builder.add_edge("history_expert", END)

supervisor_graph = builder.compile(checkpointer=MemorySaver())

# save and show the graph image
save_and_show_graph(supervisor_graph, filename="13-SupervisorGraph", show_image=False)

if __name__ == "__main__":
    # Specify a thread AKA session
    config = {"configurable": {"thread_id": "1"}}

    print("Welcome to the Supervisor Graph! Type 'exit' to quit.")

    while True:
        # Get user input
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break
        
        message = HumanMessage(content=user_input)

        messages = supervisor_graph.invoke({"messages": [message]}, config)  # Fixed: wrap in list

        for m in messages['messages']:
            m.pretty_print()
        
        print("\n\n")