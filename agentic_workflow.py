import os
from typing import Annotated, TypedDict, List
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from knowledge_base import get_retriever

# Use the provided Groq API key (could also be loaded from Streamlit secrets)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    patient_context: str
    risk_score: float
    retrieved_guidelines: str

def retrieve_node(state: AgentState):
    """Retrieves medical guidelines based on patient context and current query."""
    retriever = get_retriever()
    
    # We use the patient context and the latest user message to search for relevant guidelines
    last_message = state["messages"][-1].content if state["messages"] else ""
    query = f"Risk Score: {state['risk_score']}. {state['patient_context']} Query: {last_message}"
    
    docs = retriever.invoke(query)
    context_str = "\n".join([doc.page_content for doc in docs])
    
    return {"retrieved_guidelines": context_str}

def reason_and_generate_node(state: AgentState):
    """Uses Groq to generate a response based on patient context and guidelines."""
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        groq_api_key=GROQ_API_KEY,
        temperature=0.3
    )
    
    system_prompt = f"""You are MediRisk AI, an empathetic and professional clinical AI assistant helping doctors and patients. 
You are currently analyzing a patient with the following profile:
{state['patient_context']}
Predicted Mortality Risk Score: {state['risk_score']:.1f}%

Here are the retrieved clinical guidelines mapping to this patient's condition:
{state['retrieved_guidelines']}

# Instructions:
1. Provide a structured, empathetic, and actionable response.
2. Incorporate the retrieved guidelines to justify your recommendations. 
3. Always include a disclaimer that you are an AI and the patient should consult a doctor.
4. If answering a follow-up question, use the chat history to respond appropriately.
5. Speak directly to the person. Begin your response with "Dear Patient" or a warm greeting. You must NEVER use template placeholders like "[Patient's Name]".
"""
    
    # Combine system prompt with conversation history
    messages_for_llm = [SystemMessage(content=system_prompt)] + state["messages"]
    
    response = llm.invoke(messages_for_llm)
    return {"messages": [response]}

def build_graph():
    """Builds and compiles the LangGraph state machine."""
    builder = StateGraph(AgentState)
    
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("reason_and_generate", reason_and_generate_node)
    
    # Simple workflow: Retrieve -> Generate
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "reason_and_generate")
    builder.add_edge("reason_and_generate", END)
    
    return builder.compile()

# Compile the graph globally so it can be imported
agent_app = build_graph()

def chat_with_agent(patient_context: str, risk_score: float, user_message: str, chat_history: list = None):
    """Helper function to interact with the agent from Streamlit."""
    if chat_history is None:
        chat_history = []
        
    initial_state = {
        "messages": chat_history + [HumanMessage(content=user_message)],
        "patient_context": patient_context,
        "risk_score": risk_score,
        "retrieved_guidelines": ""
    }
    
    final_state = agent_app.invoke(initial_state)
    
    # Standardize result, the last message in state is the AI's response
    return final_state["messages"][-1]
