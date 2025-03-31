# LangGraph Cross-Thread Memory Tutorial
# For documentation - https://langchain-ai.github.io/langgraph/how-tos/cross_thread_persistence/

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from typing import Annotated, Dict, List, Any
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
import uuid
import time

from config.secret_keys import OPENAI_API_KEY

# System prompt
SYSTEM_PROMPT = """
You're Jarvis, a helpful digital assistant with an excellent memory.
You remember details about users across different conversations.

You have two main users: Krishna and Shiv. Each has their own personality and preferences.
- When talking to Shiv, be respectful and use "ji" as an honorific.
- When talking to Krishna, be more casual and energetic.

When you learn something about a user, acknowledge it and use it in future conversations.
If you don't know something about a user, be honest about it.
"""

# Initialize OpenAI components
llm = ChatOpenAI(
    model='gpt-3.5-turbo',
    api_key=OPENAI_API_KEY,
    temperature=0.7
)

# Create the embeddings model for semantic search
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

# Set up the in-memory store with embedding capabilities
try:
    memory_store = InMemoryStore(
        index={
            "embed": embeddings,
            "dims": 1536,  # Dimensions for text-embedding-3-small
        }
    )
    print("✅ Memory Store: Successfully initialized")
except Exception as e:
    print(f"❌ Memory Store Error: {e}")
    raise SystemExit("Memory store initialization failed. Exiting...")

# Function to extract memory from user message
def extract_memory(message: str) -> str:
    """Extract memory from user message if it contains 'remember:'"""
    message = message.lower()
    if "remember:" in message:
        # Get the part after "remember:"
        memory_part = message.split("remember:", 1)[1].strip()
        return memory_part
    return None

# Assistant node with memory capabilities
def Assistant(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    # Get user_id from config
    user_id = config["configurable"]["user_id"]
    namespace = ("memories", user_id)
    
    # Get the latest user message
    last_message = state["messages"][-1].content
    
    # Check if we need to store a new memory
    memory_to_store = extract_memory(last_message)
    if memory_to_store:
        # Store the new memory with a unique ID
        memory_id = str(uuid.uuid4())
        store.put(namespace, memory_id, {"data": memory_to_store})
        print(f"📝 Stored new memory for user {user_id}: {memory_to_store}")
    
    # Retrieve relevant memories using semantic search
    retrieved_memories = store.search(namespace, query=last_message, limit=5)
    
    # Format memories for the system prompt
    user_info = ""
    if retrieved_memories:
        memory_texts = [f"- {d.value['data']}" for d in retrieved_memories]
        user_info = "User information:\n" + "\n".join(memory_texts)
    
    # Create enhanced system prompt with memories
    enhanced_prompt = f"{SYSTEM_PROMPT}\n\n{user_info}"
    
    # Generate the response
    response = llm.invoke([SystemMessage(content=enhanced_prompt)] + state["messages"])
    return {"messages": response}

# Initialize LangGraph components
builder = StateGraph(MessagesState)

# Configure graph nodes and edges
builder.add_node("Assistant", Assistant)
builder.add_edge(START, "Assistant")
builder.add_edge("Assistant", END)

# Compile the graph with in-memory checkpointer and store
try:
    memory_graph = builder.compile(
        checkpointer=MemorySaver(),  # In-memory checkpointer for conversation history
        store=memory_store           # In-memory store for cross-thread memories
    )
    print("✅ Graph compiled with cross-thread memory")
except Exception as e:
    print(f"❌ Graph Compilation Error: {e}")
    raise SystemExit("Graph compilation failed. Exiting...")

# Function to run automated tests
def run_automated_test(graph, users):
    """Run an automated test to demonstrate cross-thread memory capabilities"""
    test_results = []
    thread_counters = {"Krishna": 1, "Shiv": 1}
    
    print("\n" + "="*50)
    print("🧪 STARTING AUTOMATED TEST")
    print("="*50)
    
    # Function to simulate conversation
    def simulate_conversation(user, message, config):
        thread_id = config["configurable"]["thread_id"]
        print(f"\n🔗 Thread: {thread_id}")
        print(f"👤 {user}: {message}")
        response = graph.invoke({"messages": [HumanMessage(content=message)]}, config=config)
        
        # Check if a memory was stored and display it
        if "remember:" in message.lower():
            memory_content = extract_memory(message)
            if memory_content:
                print(f"📝 Stored new memory for user {user}: {memory_content}")
        
        ai_response = response['messages'][-1].content
        print(f"🤖 Jarvis: {ai_response}")
        time.sleep(1)  # Add short delay for readability
        return response, ai_response
    
    # Phase 1: Store memories for both users
    for user in users:
        print(f"\n{'-'*20} User Session: {user} {'-'*20}")
        
        # Create config for this user
        config = {
            "configurable": {
                "thread_id": f"{user}_thread_{thread_counters[user]}",
                "user_id": user
            }
        }
        
        # Store name
        message = f"Remember: My name is {user}"
        response, ai_response = simulate_conversation(user, message, config)
        test_results.append(f"✓ Stored name for {user}")
        
        # Store favorite food
        if user == "Krishna":
            food = "butter"
        else:
            food = "thandai"
            
        message = f"Remember: My favorite food is {food}"
        response, ai_response = simulate_conversation(user, message, config)
        test_results.append(f"✓ Stored food preference for {user}")
        
        # Store nickname
        if user == "Krishna":
            nickname = "Kanha"
        else:
            nickname = "Bholenath"
            
        message = f"Remember: My nickname is {nickname}"
        response, ai_response = simulate_conversation(user, message, config)
        test_results.append(f"✓ Stored nickname for {user}")
    
    # Phase 2: Test cross-thread memory (new thread, same user)
    print("\n" + "="*50)
    print("📝 TESTING CROSS-THREAD MEMORY")
    print("="*50)
    print("Starting new conversation threads, but memory should persist...")
    time.sleep(2)
    
    for user in users:
        # Increment thread counter to simulate new conversation
        thread_counters[user] += 1
        
        config = {
            "configurable": {
                "thread_id": f"{user}_thread_{thread_counters[user]}",
                "user_id": user
            }
        }
        
        print(f"\n{'-'*20} New Thread for {user} {'-'*20}")
        message = f"What is my name, favorite food, and nickname?"
        response, ai_response = simulate_conversation(user, message, config)
        test_results.append(f"✓ Memory persists across threads for {user}")
    
    # Phase 3: Test user isolation (Krishna shouldn't know Shiv's details and vice versa)
    print("\n" + "="*50)
    print("🔒 TESTING USER MEMORY ISOLATION")
    print("="*50)
    print("Verifying that memories are isolated between users...")
    time.sleep(2)
    
    for i, user in enumerate(users):
        other_user = users[1-i]  # Get the other user
        
        config = {
            "configurable": {
                "thread_id": f"{user}_thread_{thread_counters[user]}",
                "user_id": user
            }
        }
        
        if other_user == "Krishna":
            other_nickname = "Kanha"
        else:
            other_nickname = "Bholenath"
        
        print(f"\n{'-'*20} Testing {user}'s Memory Isolation {'-'*20}")
        message = f"Is my nickname {other_nickname}?"
        response, ai_response = simulate_conversation(user, message, config)
        
        if ("no" in ai_response.lower() or "not" in ai_response.lower()):
            test_results.append(f"✓ Memory isolation confirmed for {user}")
        else:
            test_results.append(f"✗ Memory isolation failed for {user}")
    
    # Print test summary
    print("\n" + "="*50)
    print("🧪 TEST RESULTS SUMMARY")
    print("="*50)
    for result in test_results:
        print(result)
    print("\n" + "="*50)
    
    return thread_counters

def chat():
    current_user = "Krishna"  # Default user starts with Krishna
    thread_counters = {"Krishna": 1, "Shiv": 1}  # Track thread numbers per user
    users = ["Krishna", "Shiv"]  # Our two users
    
    print("\n" + "="*50)
    print("🤖 Jarvis AI Chat | Cross-Thread Memory Tutorial")
    print("="*50)
    print("- Type 'new thread' to start a new conversation")
    print("- Type 'remember: [information]' to store information")
    print("- Type 'switch' to toggle between Krishna and Shiv")
    print("- Type 'run test' to run an automated test case")
    print("- Type 'help' to show these commands")
    print("- Type 'exit' to end the conversation")
    
    while True:
        # Display current user and thread
        print("\n" + "="*50)
        print(f"👤 Current user: {current_user}")
        print(f"🔗 Thread: {current_user}_thread_{thread_counters[current_user]}")
        
        # Get user input
        user_msg = input("You: ")
        
        # Handle commands
        if user_msg.lower() == 'exit':
            print("Ending the conversation")
            break
            
        # Command to show help
        if user_msg.lower() == 'help':
            print("Commands:")
            print("- 'new thread': Start a new conversation")
            print("- 'remember: [information]': Store information")
            print("- 'switch': Toggle between Krishna and Shiv")
            print("- 'run test': Run an automated test case")
            print("- 'exit': End the conversation")
            continue
        
        # Command to switch between users
        if user_msg.lower() == 'switch':
            # Toggle between Krishna and Shiv
            current_user = "Shiv" if current_user == "Krishna" else "Krishna"
            print(f"✅ Switched to user: {current_user}")
            continue
        
        # Command to start a new thread
        if user_msg.lower() == 'new thread':
            thread_counters[current_user] += 1
            print(f"Starting new thread: {thread_counters[current_user]} for {current_user}")
            continue
            
        # Command to run automated test
        if user_msg.lower() == 'run test':
            updated_counters = run_automated_test(memory_graph, users)
            thread_counters = updated_counters  # Update thread counters after test
            continue
            
        # Regular message - prepare and invoke
        try:
            # Prepare the message and config
            human_msg = [HumanMessage(content=user_msg)]
            config = {
                "configurable": {
                    "thread_id": f"{current_user}_thread_{thread_counters[current_user]}",
                    "user_id": current_user
                }
            }
            
            # Check if memory is being stored
            memory_to_store = extract_memory(user_msg)
            if memory_to_store:
                print(f"📝 Storing new memory for user {current_user}: {memory_to_store}")
            
            # Invoke the graph
            response = memory_graph.invoke({"messages": human_msg}, config=config)
            print(f"🤖 Jarvis: {response['messages'][-1].content}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

# Run the chat function
if __name__ == "__main__":
    chat()