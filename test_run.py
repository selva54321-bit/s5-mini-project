import os
import sys
# Add the current directory to sys.path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.agent.graph import rag_agent
def run_test_query(query: str, session_id: str = "session-1"):
    print(f"\n{'='*50}")
    print(f"Testing Query: {query}")
    print(f"Session ID: {session_id}")
    print(f"{'='*50}\n")
    
    config = {"configurable": {"thread_id": session_id}}
    
    # Initialize the input state
    inputs = {
        "messages": [{"role": "user", "content": query}]
    }
    
    try:
        # Invoke the agent
        response = rag_agent.invoke(inputs, config=config)
        
        # Get the final answer
        final_message = response["messages"][-1]["content"]
        print("\n--- Final Response ---")
        print(final_message)
        
        # You can also inspect other state variables if you modify AgentState to return them
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[ERROR] Failed to run test query: {e}")
if __name__ == "__main__":
    # Ensure you have your .env file configured properly with API keys before running
    
    # Test 1: Direct Conversational (Should route to responder directly)
    # run_test_query("Hello, how are you?")
    
    # Test 2: RAG Query (Should route to retriever, search Qdrant, and then responder)
    run_test_query("""What are the exact key-value configuration examples given for labels, annotations, and selectors in architecture.pptx?""")