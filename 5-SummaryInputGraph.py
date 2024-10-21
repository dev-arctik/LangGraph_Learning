import os
import dotenv

from langchain_openai import ChatOpenAI

from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from utils.graph_img_generation import save_and_show_graph


# loading the env file
dotenv.load_dotenv()

# storing the API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



# defining the LLM
llm = ChatOpenAI(model = "gpt-4o-mini", openai_api_key=OPENAI_API_KEY)


class State(MessagesState):
    summary: str


# Define the logic to call the model
def call_model(state: State):
    
    # Get summary if it exists
    summary = state.get("summary", "")

    # If there is summary, then we add it
    if summary:
        
        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"

        # Append summary to any newer messages
        messages = [SystemMessage(content=system_message)] + state["messages"]
    
    else:
        messages = state["messages"]
    
    response = llm.invoke(messages)
    return {"messages": response}


# Creating a summary of the message history
def summarize_conversation(state: State):
    
    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization prompt 
    if summary:
        
        # A summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
        
    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)
    
    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


# Determine whether to end or summarize the conversation
def should_continue(state: State):
    
    """Return the next node to execute."""
    
    messages = state["messages"]
    
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"
    
    # Otherwise we can just end
    return END


# Define a new graph
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

# Set the entrypoint as conversation
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

# Compile
memory = MemorySaver()
summarize_conversation_graph = workflow.compile(checkpointer=memory)

# Use the utility function to save and optionally show the graph
save_and_show_graph(summarize_conversation_graph, filename="AgentGraph_withMemory_image", show_image=True)



# Specify a thread AKA session
config = {"configurable": {"thread_id": "1"}}

# Start the conversation loop
while True:
    # Take user input
    user_msg = input("Enter your message (or type 'exit' to quit): ")

    # Break the loop if the user types 'exit'
    if user_msg.lower() == 'exit':
        print("Exiting the session...")
        break

    messages = [HumanMessage(content=user_msg)]
    messages = summarize_conversation_graph.invoke({"messages": messages}, config)


    for m in messages['messages'][-1:]:
        m.pretty_print()