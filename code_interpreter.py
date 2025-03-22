from e2b_code_interpreter import Sandbox
from pydantic.v1 import BaseModel, Field
from langchain_core.messages import ToolMessage
from langchain_core.tools import Tool
import os
import json
from typing import Any

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