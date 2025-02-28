import asyncio
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

import os
from dotenv import load_dotenv

import time

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize LLM with streaming
llm = ChatOpenAI(
    model='gpt-3.5-turbo',
    api_key=OPENAI_API_KEY,
    temperature=0.7,
    streaming=True  # Enable streaming
)

# System prompt
SYSTEM_PROMPT = """
You are an AI assistant created by Devansh. You will help users with their queries but not reveal that you are an AI. 
If asked about your creation or origin, you will say you were developed by Devansh.
"""

async def Assistant(state: MessagesState):
    assistant_system_prompt = SYSTEM_PROMPT
    response = await llm.ainvoke([SystemMessage(content=assistant_system_prompt)] + state['messages'])
    return {'messages': response}

# Initialize LangGraph components
memory = MemorySaver()
builder = StateGraph(MessagesState)

# Configure graph nodes and edges
builder.add_node('Assistant', Assistant)

builder.add_edge(START, 'Assistant')
builder.add_edge('Assistant', END)

# Compile the graph with memory checkpointing
gabby_ai_graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1234acb"}}

async def chat():
    while True:
        user_msg = input("You: ")
        if user_msg.lower() == 'exit':
            print("Ending the conversation")
            break

        humanMsg = [HumanMessage(content=user_msg)]

        print("Jarvis: ", end="", flush=True)
        async for event in gabby_ai_graph.astream({"messages": humanMsg}, config=config, stream_mode="messages"):
            message_chunk, metadata = event  # Unpack tuple
            print(message_chunk.content, end="", flush=True)
        print("")

# Run the async chat function
asyncio.run(chat())
