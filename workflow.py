from langgraph.graph import END, MessageGraph
from langchain_openai import ChatOpenAI
from typing import List

# System message to guide the model's behavior with code interpreter
SYSTEM_MESSAGE = """You are a helpful AI assistant with access to a Python code interpreter.
Follow these guidelines when writing and executing code:
1. Write clear, well-commented Python code.
2. For data visualization, use matplotlib or seaborn with plt.figure(figsize=(10, 6)) for better readability.
3. Handle potential errors in your code with try/except blocks.
4. When working with external data, verify the data exists before processing.
5. Explain your approach before writing code and interpret results after execution.
6. If code execution fails, debug and provide a corrected version.
"""

def should_continue(messages) -> str:
    """Decide whether to continue or stop, based on whether there are tool calls."""
    last_message = messages[-1]
    if not last_message.tool_calls:
        return END
    else:
        return "action"

def execute_tools(messages, tool_map) -> List:
    """Execute all tool calls found in the last agent message."""
    from code_interpreter import CodeInterpreterFunctionTool, RichToolMessage
    
    tool_messages = []
    for tool_call in messages[-1].tool_calls:
        tool = tool_map[tool_call["name"]]
        if tool_call["name"] == CodeInterpreterFunctionTool.tool_name:
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