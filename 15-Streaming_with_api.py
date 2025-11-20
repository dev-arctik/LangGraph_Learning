# 15-Streaming_with_api.py
# Flask API version of supervisor graph with streaming support
#
# ============================================================================
# ARCHITECTURE OVERVIEW:
# ============================================================================
#
# This script implements a Flask API for a multi-agent supervisor pattern with real-time streaming.
#
# Graph Flow:
# 1. START → supervisor (routes user to appropriate expert)
# 2. supervisor → [assistant | math_expert | science_expert | history_expert]
# 3. math_expert ↔ math_expert_tools (if calculations needed)
# 4. experts → END
#
# Key Features:
# - Flask REST API with streaming support
# - Server-Sent Events (SSE) for real-time token-by-token streaming
# - Tool support for math calculations
# - Memory persistence across conversations (MemorySaver checkpointer)
# - Structured output for routing decisions (Pydantic models)
# - CORS support for frontend integration
#
# API Endpoints:
# - POST /chat: Send a message and receive streaming response
# - GET /health: Health check endpoint
#
# ============================================================================

import asyncio
import json
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
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

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

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

    return {
        **state,
        "next_node": response.next_node,
    }

def supervisor_router(state):
    """
    Router function to determine the next node based on supervisor's decision.
    """
    next_node = state["next_node"]

    valid_nodes = ["ASSISTANT", "MATH_EXPERT", "SCIENCE_EXPERT", "HISTORY_EXPERT"]
    if next_node not in valid_nodes:
        next_node = "ASSISTANT"
    return next_node

# ASSISTANT NODE
async def assistant(state):
    """
    Assistant node that provides general assistance.
    """
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
save_and_show_graph(supervisor_graph, filename="15-SupervisorGraph-API", show_image=False)

# ============================================================================
# FLASK API ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "LangGraph Streaming API"}), 200

@app.route('/chat', methods=['POST'])
def chat():
    """
    Chat endpoint with streaming support using Server-Sent Events (SSE).
    
    Request body:
    {
        "message": "user message here",
        "thread_id": "optional_thread_id"  # defaults to "default" if not provided
    }
    
    Response: Server-Sent Events stream with chunks in format:
    data: {"content": "chunk text", "done": false}
    data: {"content": "", "done": true}
    """
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message' field in request body"}), 400
        
        user_message = data['message']
        thread_id = data.get('thread_id', 'default')
        
        if not user_message or not user_message.strip():
            return jsonify({"error": "Message cannot be empty"}), 400
        
        # Create message object
        message = HumanMessage(content=user_message.strip())
        
        # Configuration for thread persistence
        config = {"configurable": {"thread_id": thread_id}}
        
        def generate_stream():
            """Generator function for streaming responses"""
            async def stream_messages():
                """Async generator to stream messages from LangGraph"""
                try:
                    async for event in supervisor_graph.astream(
                        {"messages": [message]}, 
                        config=config, 
                        stream_mode="messages"
                    ):
                        # Unpack the event tuple
                        message_chunk, metadata = event
                        
                        # Filter for AI message chunks from expert nodes
                        if (hasattr(message_chunk, 'content') and 
                            message_chunk.content and 
                            type(message_chunk).__name__ == 'AIMessageChunk' and
                            metadata.get('langgraph_node', '') in ['assistant', 'math_expert', 'science_expert', 'history_expert']):
                            
                            # Send chunk as SSE
                            chunk_data = json.dumps({
                                "content": message_chunk.content,
                                "done": False
                            })
                            yield f"data: {chunk_data}\n\n"
                    
                    # Send completion signal
                    final_data = json.dumps({
                        "content": "",
                        "done": True
                    })
                    yield f"data: {final_data}\n\n"
                    
                except Exception as e:
                    # Send error as SSE
                    error_data = json.dumps({
                        "content": "",
                        "done": True,
                        "error": str(e)
                    })
                    yield f"data: {error_data}\n\n"
            
            # Run the async generator in a new event loop
            # This is necessary because Flask doesn't natively support async
            try:
                # Try to get existing event loop, create new one if none exists
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Run the async generator
                async_gen = stream_messages()
                while True:
                    try:
                        chunk = loop.run_until_complete(async_gen.__anext__())
                        yield chunk
                    except StopAsyncIteration:
                        break
            except Exception as e:
                # Send error if event loop handling fails
                error_data = json.dumps({
                    "content": "",
                    "done": True,
                    "error": f"Streaming error: {str(e)}"
                })
                yield f"data: {error_data}\n\n"
        
        # Return streaming response with SSE headers
        return Response(
            generate_stream(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',  # Disable buffering in nginx
                'Connection': 'keep-alive'
            }
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("="*60)
    print("🚀 LangGraph Streaming API Server")
    print("="*60)
    print("📡 Server starting on http://localhost:5000")
    print("📝 Endpoints:")
    print("   - POST /chat - Streaming chat endpoint")
    print("   - GET  /health - Health check")
    print("="*60)
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
