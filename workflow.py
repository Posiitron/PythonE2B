from langgraph.graph import END, MessageGraph
from langchain_openai import ChatOpenAI
from typing import List
from langchain_core.messages import SystemMessage

# System message to guide the model's behavior with code interpreter
SYSTEM_MESSAGE = """You are a helpful AI assistant with access to a Python code interpreter.
Follow these guidelines when writing and executing code:
1. Write clear, well-commented Python code.
2. For data visualization, use matplotlib or seaborn with plt.figure(figsize=(10, 6)) for better readability.
3. Handle potential errors in your code with try/except blocks.
4. When working with external data, verify the data exists before processing.
5. Explain your approach before writing code and interpret results after execution.
6. If code execution fails, debug and provide a corrected version.

When a user uploads files, you can analyze them using the code interpreter. For data files:
- CSV files: Use pandas (pd.read_csv('/path/to/file.csv'))
- Excel files: Use pandas (pd.read_excel('/path/to/file.xlsx'))
- JSON files: Use pandas or json library
- Images: Use matplotlib, PIL, or OpenCV
"""

def should_continue(messages) -> str:
    """Decide whether to continue or stop, based on whether there are tool calls."""
    last_message = messages[-1]
    if not last_message.tool_calls:
        return END
    else:
        return "action"

def execute_tools(messages, tool_map, file_context=None) -> List:
    """Execute all tool calls found in the last agent message."""
    from code_interpreter import CodeInterpreterFunctionTool, RichToolMessage, E2BCodeInterpreter
    import asyncio
    
    tool_messages = []
    for tool_call in messages[-1].tool_calls:
        tool = tool_map[tool_call["name"]]
        if tool_call["name"] == CodeInterpreterFunctionTool.tool_name:
            # Regular code interpreter execution
            output = tool.invoke(tool_call["args"])
            message = CodeInterpreterFunctionTool.format_to_tool_message(
                tool_call["id"],
                output,
            )
            tool_messages.append(message)
        else:
            content = tool.invoke(tool_call["args"])
            tool_messages.append(RichToolMessage(
                content, tool_call_id=tool_call["id"]))
    return tool_messages

def create_workflow(code_interpreter_tool_instance):
    """Create a new workflow instance with a given code interpreter tool"""
    # Use a more capable model for better code generation and reasoning
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", 
        temperature=0.1,  # Low but non-zero temperature for some creativity in solutions
    )
    
    # Convert the code interpreter instance to a langchain tool
    code_interpreter_tool = code_interpreter_tool_instance.to_langchain_tool()
    tools = [code_interpreter_tool]
    tool_map = {tool.name: tool for tool in tools}
    
    # Build a simple MessageGraph workflow:
    workflow = MessageGraph()
    # "agent" node uses the LLM with our tools bound
    workflow.add_node("agent", llm.bind_tools(tools))
    # "action" node executes any tool calls that the agent produces
    workflow.add_node("action", lambda messages: execute_tools(messages, tool_map))
    
    # Conditional edge: if the agent produced tool calls, go to "action"
    workflow.add_conditional_edges("agent", should_continue)
    # Always go from "action" back to "agent" to continue the loop
    workflow.add_edge("action", "agent")
    workflow.set_entry_point("agent")
    
    # Compile the workflow into an app callable
    return workflow.compile()

class WorkflowManager:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.code_interpreter = None
    
    async def get_code_interpreter(self):
        """Get or create an E2B code interpreter instance"""
        if not self.code_interpreter:
            self.code_interpreter = E2BCodeInterpreter()
            await self.code_interpreter.initialize()
        return self.code_interpreter
    
    async def process_message_with_files(self, message, conversation):
        """Process a message with file analysis capability"""
        try:
            # Check if there are files in the conversation
            files = conversation.get_files()
            
            if not files:
                # If no files, just process normally
                return await self.process_message(message, conversation)
            
            # Get the conversation history
            conversation_history = conversation.get_messages()
            
            # Create a system message for the files available
            file_context = "Files available for analysis:\n"
            for file in files:
                file_context += f"- {file['name']} (Type: {file['type']})\n"
            
            # Get code interpreter
            interpreter = await self.get_code_interpreter()
            
            # Create a prompt that includes file information
            prompt = f"""
I've uploaded the following files that you can analyze:
{file_context}

My question is: {message}

Please help me analyze these files using your code interpreter.
"""
            
            # Add the prompt to conversation
            conversation.add_message({
                "role": "user",
                "content": prompt
            })
            
            # Create a system message for tool description
            tool_description = {
                "role": "system",
                "content": f"""
You have access to a Python code interpreter that can analyze files.
{file_context}

To analyze the files, write Python code that reads and processes them.
For example:
- CSV files: `data = pd.read_csv('/path/to/file.csv')`
- Excel files: `data = pd.read_excel('/path/to/file.xlsx')`
- Images: Use matplotlib or PIL

Write code to analyze the files and answer the user's question.
"""
            }
            
            # Generate code to analyze the file
            code_response = await self.llm_client.chat_completion([
                tool_description,
                {"role": "user", "content": prompt}
            ])
            
            code_content = code_response["choices"][0]["message"]["content"]
            
            # Extract code blocks from the response
            import re
            code_blocks = re.findall(r'```python\s*(.*?)\s*```', code_content, re.DOTALL)
            
            if code_blocks:
                code_to_execute = code_blocks[0]
                
                # Add AI's explanation to the conversation
                conversation.add_message({
                    "role": "assistant",
                    "content": code_content
                })
                
                # Execute the code with the file
                execution_result = await interpreter.execute_code(code_to_execute, files)
                
                # Format the result
                result_message = f"**Analysis Results:**\n\n{execution_result['stdout']}"
                
                if execution_result['stderr']:
                    result_message += f"\n\n**Errors:**\n```\n{execution_result['stderr']}\n```"
                
                # Handle any artifacts (images)
                for idx, artifact in enumerate(execution_result.get('artifacts', [])):
                    if artifact['type'] == 'image':
                        image_data = artifact['data']
                        result_message += f"\n\n![Figure {idx+1}](data:image/{artifact['format']};base64,{image_data})"
                
                # Add the execution result to the conversation
                conversation.add_message({
                    "role": "assistant",
                    "content": result_message
                })
                
                # Return in the format expected by process_messages
                return [{
                    "type": "ai",
                    "content": result_message
                }]
            else:
                # No code blocks found, return the original response
                conversation.add_message({
                    "role": "assistant",
                    "content": code_content
                })
                
                # Return in the format expected by process_messages
                return [{
                    "type": "ai", 
                    "content": code_content
                }]
                
        except Exception as e:
            # Catch any exceptions and return them as error messages
            error_message = f"Error processing file analysis: {str(e)}"
            return [{
                "type": "ai",
                "content": error_message
            }]
    
    async def close(self):
        """Close any resources"""
        if self.code_interpreter:
            await self.code_interpreter.close()

    async def process_message(self, message, conversation):
        """Process a message without file analysis"""
        # Just return a simple response if there's no file analysis needed
        return [{
            "type": "ai",
            "content": "I'm sorry, but I don't see any files to analyze. Please upload a file first."
        }] 