from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os

from code_interpreter import CodeInterpreterFunctionTool
from conversation import ConversationManager
from workflow import create_workflow, SYSTEM_MESSAGE

# Load environment variables from .env
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize conversation manager
conversation_manager = ConversationManager()

@app.route("/", methods=["GET"])
def index():
    """Return the main UI page"""
    return render_template('index.html')

@app.route("/run", methods=["POST"])
def run_agent():
    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "Missing 'prompt' field in JSON body"}), 400

    prompt = data["prompt"]
    session_id = data.get("session_id", "default")
    
    # Get memory for this session
    memory = conversation_manager.get_memory(session_id)
    
    # Create a fresh instance of the code interpreter tool for this request
    local_code_interpreter = CodeInterpreterFunctionTool()
    
    try:
        # Create workflow with the local tool instance
        workflow_app = create_workflow(local_code_interpreter)
        
        # Process the request with memory and system message
        result = process_with_memory(workflow_app, memory, prompt)
        
        # Process and enhance outputs for better display
        processed_messages = process_messages(result)
        
        return jsonify({
            "messages": processed_messages,
            "session_id": session_id
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Always close the local sandbox to prevent resource leaks
        local_code_interpreter.close()

@app.route("/clear", methods=["POST"])
def clear_session():
    """Clear a conversation session"""
    data = request.get_json()
    session_id = data.get("session_id", "default")
    
    conversation_manager.clear_memory(session_id)
    return jsonify({"status": "success", "message": f"Session {session_id} cleared"})

def process_with_memory(workflow_app, memory, prompt):
    """Process a prompt using the workflow and memory"""
    from langchain_core.messages import SystemMessage, HumanMessage
    
    # Add system message at the start of the conversation
    system_msg = SystemMessage(content=SYSTEM_MESSAGE)
    
    # Get previous messages from memory
    previous_messages = memory.chat_memory.messages
    
    # Add system message at the beginning
    all_messages = [system_msg]
    
    # Add previous conversation messages
    all_messages.extend(previous_messages)
    
    # Add current human message
    all_messages.append(HumanMessage(content=prompt))
    
    # Invoke the workflow with all messages
    result = workflow_app.invoke(all_messages)
    
    # Save the conversation to memory
    for msg in result:
        if msg.type == "human":
            memory.chat_memory.add_user_message(msg.content)
        elif msg.type == "ai":
            memory.chat_memory.add_ai_message(msg.content)
    
    return result

def process_messages(messages):
    """Process and enhance message outputs for better display"""
    processed_messages = []
    for msg in messages:
        if hasattr(msg, 'raw_output') and msg.raw_output:
            processed_output = process_tool_output(msg.raw_output)
            processed_messages.append({
                "type": msg.type,
                "content": msg.content,
                "enhanced_output": processed_output
            })
        else:
            processed_messages.append({
                "type": msg.type,
                "content": msg.content
            })
    return processed_messages

def process_tool_output(output):
    """Process and enhance tool output for better display"""
    
    # Function to detect image paths or base64 data in output
    def extract_visualization_data(results):
        # Implementation would detect and process visualization data
        # This is a placeholder - real implementation would parse visualization data
        return None
    
    # Extract any visualization data
    viz_data = None
    if "results" in output and output["results"]:
        viz_data = extract_visualization_data(output["results"])
    
    # Enhance the output with formatted text and visualization references
    enhanced_output = {
        "stdout": output.get("stdout", ""),
        "stderr": output.get("stderr", ""),
        "error": output.get("error", ""),
        "visualization": viz_data
    }
    
    return enhanced_output

if __name__ == "__main__":
    # Check for required environment variables
    required_vars = ["E2B_API_KEY", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if var not in os.environ]
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them in your environment or .env file.")
        exit(1)
    
    # Run the Flask app in debug mode on 127.0.0.1:5000
    app.run(debug=True)
