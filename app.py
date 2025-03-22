from e2b_code_interpreter import Sandbox
from pydantic.v1 import BaseModel, Field
from langchain_core.messages import ToolMessage
from langchain_core.tools import Tool
from langgraph.graph import END, MessageGraph
from langchain_openai import ChatOpenAI
import os
import json
from typing import Any, List

from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Import LangChain and LangGraph pieces

# Updated import path for the E2B code interpreter


# --------------- Code Interpreter Tool Definition ---------------

class RichToolMessage(ToolMessage):
    raw_output: dict


class LangchainCodeInterpreterToolInput(BaseModel):
    code: str = Field(description="Python code to execute.")


class CodeInterpreterFunctionTool:
    """A tool that executes Python code inside an E2B sandbox."""
    tool_name: str = "code_interpreter"

    def __init__(self):
        if "E2B_API_KEY" not in os.environ:
            raise Exception(
                "E2B_API_KEY is not set. "
                "Please set it in your environment or .env file. "
                "Get your E2B API key at https://e2b.dev/docs"
            )
        self.code_interpreter = Sandbox()

    def close(self):
        self.code_interpreter.kill()

    def call(self, parameters: dict, **kwargs: Any):
        code = parameters.get("code", "")
        print(f"***Code Interpreting...\n{code}\n====")
        execution = self.code_interpreter.run_code(code)
        return {
            "results": execution.results,
            "stdout": execution.logs.stdout,
            "stderr": execution.logs.stderr,
            "error": execution.error,
        }

    # For LangChain, the tool is invoked with just a code string.
    def langchain_call(self, code: str):
        return self.call({"code": code})

    def to_langchain_tool(self) -> Tool:
        tool = Tool(
            name=self.tool_name,
            description="Execute python code in a Jupyter cell and return outputs (e.g. plots, stdout, stderr).",
            func=self.langchain_call,
        )
        tool.args_schema = LangchainCodeInterpreterToolInput
        return tool

    @staticmethod
    def format_to_tool_message(tool_call_id: str, output: dict) -> RichToolMessage:
        # By default, skip the 'results' for brevity in the text output
        content = json.dumps(
            {k: v for k, v in output.items() if k not in ("results")},
            indent=2
        )
        return RichToolMessage(
            content=content,
            raw_output=output,
            tool_call_id=tool_call_id,
        )


# --------------- Workflow Helper Functions ---------------

def should_continue(messages) -> str:
    """Decide whether to continue or stop, based on whether there are tool calls."""
    last_message = messages[-1]
    if not last_message.tool_calls:
        return END
    else:
        return "action"


def execute_tools(messages, tool_map) -> List[RichToolMessage]:
    """Execute all tool calls found in the last agent message."""
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


# --------------- Setup LLM, Tools, Workflow ---------------

# Use GPT-3.5 Turbo with zero temperature
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# Initialize the E2B code interpreter tool
code_interpreter = CodeInterpreterFunctionTool()
code_interpreter_tool = code_interpreter.to_langchain_tool()
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
compiled_app = workflow.compile()


# --------------- Build the Flask App ---------------

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    """
    Simple HTML page with an input box and a button that sends a POST request to /run.
    """
    return """
    <html>
      <head>
        <title>E2B + LangChain Demo</title>
      </head>
      <body>
        <h1>E2B + LangChain Demo</h1>
        <p>Type a prompt and click 'Send' to run the E2B code interpreter workflow.</p>
        
        <input id="promptInput" type="text" placeholder="Enter your prompt here" style="width:300px;" />
        <button id="sendBtn">Send</button>
        
        <h3>Response:</h3>
        <pre id="responseOutput" style="border:1px solid #ccc; padding:8px;"></pre>
        
        <script>
          document.getElementById('sendBtn').addEventListener('click', async () => {
            const prompt = document.getElementById('promptInput').value;
            // Make a POST request to /run with JSON
            const resp = await fetch('/run', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ prompt })
            });
            const data = await resp.json();
            document.getElementById('responseOutput').textContent = JSON.stringify(data, null, 2);
          });
        </script>
      </body>
    </html>
    """


@app.route("/run", methods=["POST"])
def run_agent():
    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "Missing 'prompt' field in JSON body"}), 400

    prompt = data["prompt"]

    # Create a fresh instance of the code interpreter tool for this request
    local_code_interpreter = CodeInterpreterFunctionTool()
    local_tool = local_code_interpreter.to_langchain_tool()
    local_tools = [local_tool]
    local_tool_map = {tool.name: tool for tool in local_tools}

    # Build a fresh MessageGraph workflow using the local tool
    local_workflow = MessageGraph()
    local_workflow.add_node("agent", llm.bind_tools(local_tools))
    local_workflow.add_node(
        "action", lambda messages: execute_tools(messages, local_tool_map))
    local_workflow.add_conditional_edges("agent", should_continue)
    local_workflow.add_edge("action", "agent")
    local_workflow.set_entry_point("agent")
    local_compiled_app = local_workflow.compile()

    # Invoke the workflow with the prompt
    result = local_compiled_app.invoke([("human", prompt)])

    # Close the local sandbox after the run
    local_code_interpreter.close()

    # Collect and return the message contents
    messages = [msg.content for msg in result]
    return jsonify({"messages": messages})


if __name__ == "__main__":
    # Run the Flask app in debug mode on 127.0.0.1:5000
    app.run(debug=True)
