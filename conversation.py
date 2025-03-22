from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict

class Conversation:
    def __init__(self, conversation_id=None):
        self.id = conversation_id
        self.files = []  # Track uploaded files
        self.messages = []  # Track messages in this conversation
    
    def add_files(self, file_data):
        """Add file metadata to the conversation context"""
        self.files.extend(file_data)
        
        # Add a system message about the uploaded files
        file_names = [file['name'] for file in file_data]
        file_message = f"User has uploaded the following files: {', '.join(file_names)}"
        
        self.add_message({
            "role": "system",
            "content": file_message
        })
    
    def add_message(self, message):
        """Add a message to the conversation"""
        self.messages.append(message)
    
    def get_messages(self):
        """Get all messages in this conversation"""
        return self.messages
    
    def get_files(self):
        """Get all files associated with this conversation"""
        return self.files
    
    def get_conversation_context(self):
        """Get the conversation context including files for the AI"""
        context = {
            "messages": self.messages,
            "conversation_id": self.id
        }
        
        if self.files:
            context["files"] = self.files
        
        return context

class ConversationManager:
    """Manages conversation memory for multiple sessions"""
    
    def __init__(self):
        self.sessions = {}
        self.conversations = {}  # Store Conversation objects by session ID
    
    def get_memory(self, session_id):
        """Get or create memory for a session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True
            )
        return self.sessions[session_id]
    
    def get_conversation(self, session_id):
        """Get or create a Conversation object for a session"""
        if session_id not in self.conversations:
            self.conversations[session_id] = Conversation(conversation_id=session_id)
        return self.conversations[session_id]
    
    def clear_memory(self, session_id):
        """Clear memory and conversation for a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
        if session_id in self.conversations:
            del self.conversations[session_id]
            
    def serialize_memory(self, session_id):
        """Serialize memory to a dict for storage"""
        if session_id in self.sessions:
            return messages_to_dict(self.sessions[session_id].chat_memory.messages)
        return []
    
    def load_memory(self, session_id, messages_dict):
        """Load memory from a serialized dict"""
        memory = self.get_memory(session_id)
        memory.chat_memory.messages = messages_from_dict(messages_dict) 