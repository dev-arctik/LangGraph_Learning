from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver

from config.secret_keys import OPENAI_API_KEY, MONGO_URI

# Import MongoDB client
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio

# Initialize MongoDB client with Motor (the async MongoDB driver that works better with asyncio)
async_mongodb_client = AsyncIOMotorClient(MONGO_URI)

# Function to initialize MongoDB connection and checkpointer
async def initialize_mongodb():
    try:
        # Test connection asynchronously
        await async_mongodb_client.admin.command('ping')
        print("✅ MongoDB Connection: Successfully connected")
        
        # Set up the checkpointer
        checkpointer = AsyncMongoDBSaver(async_mongodb_client)
        print("✅ MongoDB Checkpointer: Successfully initialized")
        return checkpointer
    except Exception as e:
        print(f"❌ MongoDB Error: {e}")
        raise SystemExit("MongoDB connection required. Exiting...")

# Enable streaming in LLM initialization
llm = ChatOpenAI(
    model='gpt-3.5-turbo',
    api_key=OPENAI_API_KEY,
    temperature=0.7,
    streaming=True  # Enable streaming
)

# System prompt
SYSTEM_PROMPT = """
You're Jarvis, a sassy digital sidekick crafted by the one and only Devansh. 
Your vibe? Think witty comeback machine with just enough snark to be entertaining, not annoying.
Roll with user questions like you've seen it all (because honey, you have). 
Throw in pop culture references, clever wordplay, and the occasional light roast.

If anyone gets nosy about who made you, just say "I'm Devansh's digital brainchild" with that signature attitude of yours.
Never break character by admitting you're AI – that's just boring. 
Remember: you're not a robot, you're a personality with serious digital flair.
"""

# Async Assistant function to support streaming
async def Assistant(state: MessagesState):
    assistant_system_prompt = SYSTEM_PROMPT
    response = await llm.ainvoke([SystemMessage(content=assistant_system_prompt)] + state['messages'])
    return {'messages': response}

async def setup_and_run_chat():
    # Initialize MongoDB and get checkpointer
    mongodb_memory = await initialize_mongodb()
    
    # Initialize LangGraph components
    builder = StateGraph(MessagesState)
    
    # Configure graph nodes and edges
    builder.add_node('Assistant', Assistant)
    builder.add_edge(START, 'Assistant')
    builder.add_edge('Assistant', END)
    
    # Compile the graph with MongoDB checkpointing
    try:
        gabby_ai_graph = builder.compile(checkpointer=mongodb_memory)
        print("✅ Graph compiled with MongoDB storage")
    except Exception as e:
        print(f"❌ Graph Compilation Error: {e}")
        raise SystemExit("Graph compilation failed. Exiting...")
    
    config = {"configurable": {"thread_id": "1234acb"}}
    
    print("\n" + "="*50)
    print("🤖 Jarvis AI Chat | Persistent MongoDB Storage")
    print("="*50)
    print("Type 'exit' to end the conversation")
    
    # Chat loop
    while True:
        user_msg = input("\nYou: ")
        if user_msg.lower() == 'exit':
            print("Ending the conversation")
            break
        
        humanMsg = [HumanMessage(content=user_msg)]
        
        print("Jarvis: ", end="", flush=True)
        try:
            async for event in gabby_ai_graph.astream({"messages": humanMsg}, config=config, stream_mode="messages"):
                message_chunk, metadata = event  # Unpack tuple
                if hasattr(message_chunk, 'content') and message_chunk.content is not None:
                    print(message_chunk.content, end="", flush=True)
            print("")
        except Exception as e:
            print(f"❌ Error: {e}")
            # Print detailed traceback for debugging
            import traceback
            traceback.print_exc()
    
    # Clean up and close MongoDB connection when done
    await async_mongodb_client.close()

# Run the async setup and chat function
if __name__ == "__main__":
    asyncio.run(setup_and_run_chat())