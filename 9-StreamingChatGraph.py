import asyncio
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from config.secret_keys import OPENAI_API_KEY
from config.config import get_llm

from utils.graph_img_generation import save_and_show_graph

# Initialize LLM with streaming
llm = get_llm()

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
ai_graph = builder.compile(checkpointer=memory)

# Save and show the graph image
save_and_show_graph(ai_graph, filename="9-StreamingChatGraph", show_image=False)

config = {"configurable": {"thread_id": "1234acb"}}

async def chat():
    while True:
        user_msg = input("You: ")
        if user_msg.lower() == 'exit':
            print("Ending the conversation")
            break

        humanMsg = [HumanMessage(content=user_msg)]

        print("Jarvis: ", end="", flush=True)
        async for event in ai_graph.astream({"messages": humanMsg}, config=config, stream_mode="messages"):
            message_chunk, metadata = event  # Unpack tuple
            print(message_chunk.content, end="", flush=True)
            await asyncio.sleep(0.05)  # Simulate streaming delay
        print("")

# Run the async chat function
asyncio.run(chat())
