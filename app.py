from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
from werkzeug.utils import secure_filename
import uuid
import asyncio

from code_interpreter import CodeInterpreterFunctionTool
from conversation import ConversationManager
from workflow import create_workflow, SYSTEM_MESSAGE, WorkflowManager
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize conversation manager
conversation_manager = ConversationManager()

# Initialize LLM client for file analysis
llm_client = ChatOpenAI(
    model="gpt-3.5-turbo", 
    temperature=0.1
)

# Configure file upload settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv', 'xlsx', 'xls', 'json', 'py'}

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    conversation = conversation_manager.get_conversation(session_id)
    
    # Create a fresh instance of the code interpreter tool for this request
    local_code_interpreter = CodeInterpreterFunctionTool()
    
    try:
        # Check if there are files to process
        if hasattr(conversation, 'files') and conversation.files:
            # Create workflow manager for file processing
            workflow_manager = WorkflowManager(llm_client)
            # Process with file support
            result = asyncio.run(process_with_files(workflow_manager, conversation, prompt))
        else:
            # Create normal workflow without file support
            workflow_app = create_workflow(local_code_interpreter)
            # Process normally
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

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})
    
    uploaded_files = request.files.getlist('files')
    
    if not uploaded_files or uploaded_files[0].filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})
    
    file_data = []
    
    for file in uploaded_files:
        if file and allowed_file(file.filename):
            # Generate a unique filename to avoid collisions
            original_filename = secure_filename(file.filename)
            file_extension = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else ''
            unique_filename = f"{uuid.uuid4().hex}.{file_extension}" if file_extension else f"{uuid.uuid4().hex}"
            
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Store file metadata
            file_info = {
                'id': unique_filename,
                'name': original_filename,
                'path': file_path,
                'type': file_extension,
                'size': os.path.getsize(file_path)
            }
            file_data.append(file_info)
    
    # Add file info to the conversation context
    if hasattr(request, 'conversation') and request.conversation:
        request.conversation.add_files(file_data)
    
    return jsonify({
        'success': True,
        'files': file_data
    })

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
    
    # Check if messages is a list of dictionaries already
    if isinstance(messages, list) and all(isinstance(msg, dict) for msg in messages):
        # Already in the right format
        return messages
    
    # Process object-style messages
    for msg in messages:
        try:
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
        except Exception as e:
            # If there's an error processing a message, include an error message
            processed_messages.append({
                "type": "ai",
                "content": f"Error processing results: {str(e)}"
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

async def process_with_files(workflow_manager, conversation, prompt):
    """Process a prompt with file support using the workflow manager"""
    # Check if there are any files in the conversation
    files = getattr(conversation, 'files', [])
    
    if files:
        # Process with file support
        return await workflow_manager.process_message_with_files(prompt, conversation)
    else:
        # We don't have workflow_app and memory in scope here, so just return an error message
        return [{
            "type": "ai", 
            "content": "I don't see any uploaded files to analyze. Please upload a file first."
        }]

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
