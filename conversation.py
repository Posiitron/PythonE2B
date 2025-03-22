from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict

class ConversationManager:
    """Manages conversation memory for multiple sessions"""
    
    def __init__(self):
        self.sessions = {}
    
    def get_memory(self, session_id):
        """Get or create memory for a session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True
            )
        return self.sessions[session_id]
    
    def clear_memory(self, session_id):
        """Clear memory for a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            
    def serialize_memory(self, session_id):
        """Serialize memory to a dict for storage"""
        if session_id in self.sessions:
            return messages_to_dict(self.sessions[session_id].chat_memory.messages)
        return []
    
    def load_memory(self, session_id, messages_dict):
        """Load memory from a serialized dict"""
        memory = self.get_memory(session_id)
        memory.chat_memory.messages = messages_from_dict(messages_dict) 