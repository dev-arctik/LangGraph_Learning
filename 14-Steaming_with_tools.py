# 14-SupervisorGraph-Streaming.py
# Streaming version of supervisor graph with tool support
#
# ============================================================================
# ARCHITECTURE OVERVIEW:
# ============================================================================
#
# This script implements a multi-agent supervisor pattern with real-time streaming.
#
# Graph Flow:
# 1. START → supervisor (routes user to appropriate expert)
# 2. supervisor → [assistant | math_expert | science_expert | history_expert]
# 3. math_expert ↔ math_expert_tools (if calculations needed)
# 4. experts → END
#
# Key Features:
# - Async/await for non-blocking operations
# - Real-time token-by-token streaming (ChatGPT-like experience)
# - Tool support for math calculations
# - Memory persistence across conversations (MemorySaver checkpointer)
# - Structured output for routing decisions (Pydantic models)
#
# Streaming Implementation:
# - Uses astream() with stream_mode="messages" for low-latency streaming
# - Filters messages by node name to show only expert responses
# - Filters by message type (AIMessageChunk) to exclude system/tool messages
# - Prevents supervisor's JSON routing decisions from appearing in output
#
# ============================================================================

import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph.message import add_messages, AnyMessage
from typing import TypedDict, Annotated, List, Literal
from langgraph.checkpoint.memory import MemorySaver

from config.secret_keys import OPENAI_API_KEY
from config.config import get_llm

from utils.graph_img_generation import save_and_show_graph

# Define LLM
llm = get_llm()

# Define Custom State
class CustomState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    next_node: str

# Define Model for structured output
class SupervisorModel(BaseModel):
    next_node: Literal['ASSISTANT', 'MATH_EXPERT', 'SCIENCE_EXPERT', 'HISTORY_EXPERT'] = Field(
        ...,
        description="The next node to which the user should be directed. It can be 'ASSISTANT', 'MATH_EXPERT', 'SCIENCE_EXPERT', or 'HISTORY_EXPERT'.",
    )

# Define math tools
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

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

tools = [add, subtract, multiply, divide]

# Define NODES (Async versions)

# SUPERVISOR NODE
async def supervisor(state):
    """
    Supervisor node that directs the user to Math Expert, Science Expert, or History Expert.
    """
    # print("----------INSIDE SUPERVISOR----------")

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

    response = await llm_with_structured_output.ainvoke(messages)

    # Debug output moved to separate line for cleaner console output
    # print(f"Supervisor routing to: {response.next_node}")

    return {
        **state,
        "next_node": response.next_node,
    }

def supervisor_router(state):
    """
    Router function to determine the next node based on supervisor's decision.
    """
    # print("----------INSIDE SUPERVISOR ROUTER----------")
    next_node = state["next_node"]

    valid_nodes = ["ASSISTANT", "MATH_EXPERT", "SCIENCE_EXPERT", "HISTORY_EXPERT"]
    if next_node not in valid_nodes:
        print(f"Invalid next node '{next_node}'. Defaulting to 'ASSISTANT'.")
        next_node = "ASSISTANT"
    return next_node

# ASSISTANT NODE
async def assistant(state):
    """
    Assistant node that provides general assistance.
    """
    # print("----------INSIDE ASSISTANT----------")

    assistant_prompt = """
    You are a helpful general assistant. You provide clear, informative responses to a wide range of questions.

    Your role is to:
    - Answer general knowledge questions
    - Provide helpful explanations on various topics
    - Assist with everyday questions and tasks
    - Offer guidance when users need general help

    Be friendly, concise, and helpful. If a question requires specialized expertise in math, science, or history, 
    let the user know they might want to ask about that specific topic to get more detailed help.

    Your response should be short and friendly, encouraging users to ask more questions if they need further assistance.
    """

    messages = [SystemMessage(content=assistant_prompt)] + state["messages"]

    response = await llm.ainvoke(messages)

    return {
        "messages": response
    }

# MATH EXPERT NODE
async def math_expert(state):
    """
    Math expert node that provides answers to math-related questions.
    """
    # print("----------INSIDE MATH EXPERT----------")

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

    response = await llm_with_tools.ainvoke(messages)

    return {
        "messages": response
    }

# SCIENCE EXPERT NODE
async def science_expert(state):
    """
    Science expert node that provides answers to science-related questions.
    """
    # print("----------INSIDE SCIENCE EXPERT----------")

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

    response = await llm.ainvoke(messages)

    return {
        "messages": response
    }

# HISTORY EXPERT NODE
async def history_expert(state):
    """
    History expert node that provides answers to history-related questions.
    """
    # print("----------INSIDE HISTORY EXPERT----------")

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

    response = await llm.ainvoke(messages)

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
        "tools": "math_expert_tools",
        "__end__": END
    }
)
builder.add_edge("math_expert_tools", "math_expert")

builder.add_edge("assistant", END)
builder.add_edge("science_expert", END)
builder.add_edge("history_expert", END)

supervisor_graph = builder.compile(checkpointer=MemorySaver())

# Save and show the graph image
save_and_show_graph(supervisor_graph, filename="14-SupervisorGraph-Streaming", show_image=False)

async def chat():
    """
    Async chat function with streaming support.
    """
    config = {"configurable": {"thread_id": "1"}}
    
    print("Welcome to the Supervisor Graph with Streaming! Type 'exit' to quit.\n")

    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Ending the conversation. Goodbye!")
            break
        
        message = HumanMessage(content=user_input)

        print("Assistant: ", end="", flush=True)
        
        # ============================================================================
        # STREAMING EXPLANATION:
        # ============================================================================
        # 
        # LangGraph supports multiple streaming modes. Here we use "messages" mode:
        # 
        # Stream Modes:
        # - "values": Streams complete state updates after each node executes
        # - "updates": Streams only the changes (deltas) to state
        # - "messages": Streams individual message chunks as they're generated (best for real-time chat)
        # 
        # Why use stream_mode="messages"?
        # - Provides token-by-token streaming for a smooth, ChatGPT-like experience
        # - Each chunk arrives as soon as the LLM generates it (lowest latency)
        # - User sees text appearing progressively rather than waiting for full response
        # 
        # Event Structure:
        # Each event is a tuple: (message_chunk, metadata)
        # - message_chunk: The actual message object (AIMessageChunk, HumanMessage, ToolMessage, etc.)
        # - metadata: Dictionary containing information like:
        #   * 'langgraph_node': Name of the node that produced this message
        #   * 'langgraph_step': Step number in the execution
        #   * 'langgraph_triggers': What triggered this node
        # 
        # Why We Need Filtering Conditions:
        # 
        # 1. Multiple Nodes Generate Messages:
        #    - supervisor: Generates structured output {"next_node": "MATH_EXPERT"}
        #    - assistant/experts: Generate actual text responses
        #    - math_expert_tools: Generates tool call messages
        #    Without filtering, ALL of these would be printed!
        # 
        # 2. Different Message Types:
        #    - AIMessageChunk: Chunks of AI responses (what we want)
        #    - HumanMessage: User input (already displayed)
        #    - ToolMessage: Tool execution results (internal)
        #    - SystemMessage: System prompts (internal)
        # 
        # Our Filtering Strategy:
        # ============================================================================
        
        async for event in supervisor_graph.astream(
            {"messages": [message]}, 
            config=config, 
            stream_mode="messages"
        ):
            # Unpack the event tuple
            message_chunk, metadata = event
            
            # CONDITION 1: Filter by Node Name
            # Extract which node produced this message
            node_name = metadata.get('langgraph_node', '')
            
            # We only want to display messages from expert nodes, NOT from:
            # - 'supervisor': Would show JSON like {"next_node": "ASSISTANT"}
            # - 'math_expert_tools': Would show tool execution details
            # This ensures we only show the final expert responses to the user
            
            # CONDITION 2: Check Message Type
            # type(message_chunk).__name__ == 'AIMessageChunk' ensures we only print AI responses
            # This filters out HumanMessage, ToolMessage, SystemMessage, etc.
            
            # CONDITION 3: Check for Content
            # hasattr(message_chunk, 'content') - Ensure the chunk has a content attribute
            # message_chunk.content - Ensure content is not empty/None
            # Some chunks might be empty (e.g., initial chunks, tool calls without text)
            
            # COMBINED FILTER: All conditions must be True
            if (hasattr(message_chunk, 'content') and 
                message_chunk.content and 
                type(message_chunk).__name__ == 'AIMessageChunk' and
                node_name in ['assistant', 'math_expert', 'science_expert', 'history_expert']):
                
                # Print the chunk without newline, flush immediately for real-time display
                print(message_chunk.content, end="", flush=True)
                
                # Small delay creates a more natural typing effect
                # Remove this if you want maximum speed
                await asyncio.sleep(0.01)
        
        # Print newline after streaming completes
        print("\n")

if __name__ == "__main__":
    # Run the async chat function
    asyncio.run(chat())