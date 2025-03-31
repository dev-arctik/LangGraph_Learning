# for documentation - https://langchain-ai.github.io/langgraph/how-tos/persistence_mongodb/
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.checkpoint.mongodb import MongoDBSaver

from config.secret_keys import OPENAI_API_KEY, MONGO_URI

# Import MongoDB client
from pymongo import MongoClient

mongodb_client = MongoClient(MONGO_URI)

# Add error handling for MongoDB connection
try:
    # Test connection
    mongodb_client.admin.command('ping')
    print("✅ MongoDB Connection: Successfully connected")
    
    # Set up the checkpointer
    mongodb_memory = MongoDBSaver(mongodb_client)
    print("✅ MongoDB Checkpointer: Successfully initialized")
except Exception as e:
    print(f"❌ MongoDB Error: {e}")
    raise SystemExit("MongoDB connection required. Exiting...")

llm = ChatOpenAI(
    model='gpt-3.5-turbo',
    api_key=OPENAI_API_KEY,
    temperature=0.7
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

def Assistant(state: MessagesState):
    assistant_system_prompt = SYSTEM_PROMPT
    response = llm.invoke([SystemMessage(content=assistant_system_prompt)] + state['messages'])
    return {'messages': response}

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

def chat():
    while True:
        user_msg = input("\nYou: ")
        if user_msg.lower() == 'exit':
            print("Ending the conversation")
            break

        humanMsg = [HumanMessage(content=user_msg)]

        try:
            response = gabby_ai_graph.invoke({"messages": humanMsg}, config=config)
            print("Jarvis:", response['messages'][-1].content)
        except Exception as e:
            print(f"❌ Error: {e}")

# Run the chat function
chat()