from e2b_code_interpreter import Sandbox as CodeInterpreterSandbox
from pydantic.v1 import BaseModel, Field
from langchain_core.messages import ToolMessage
from langchain_core.tools import Tool
import os
import json
from typing import Any
import base64
from e2b import Sandbox

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
        self.code_interpreter = CodeInterpreterSandbox()

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

class E2BCodeInterpreter:
    def __init__(self):
        self.sandbox = None
    
    async def initialize(self):
        """Initialize the E2B sandbox"""
        self.sandbox = Sandbox()
        await self.sandbox.start()
        
        # Install common data analysis packages
        await self.sandbox.process.start_and_wait("pip install pandas matplotlib seaborn numpy openpyxl plotly scikit-learn")
        return self.sandbox
    
    async def upload_file(self, file_path, filename=None):
        """Upload a file to the E2B sandbox"""
        if not self.sandbox:
            await self.initialize()
        
        if not filename:
            filename = os.path.basename(file_path)
        
        # Create destination directory if needed
        await self.sandbox.filesystem.make_dir("/home/user/data", recursive=True)
        
        with open(file_path, "rb") as f:
            content = f.read()
        
        # Write the file to the sandbox
        sandbox_path = f"/home/user/data/{filename}"
        await self.sandbox.filesystem.write(sandbox_path, content)
        
        return sandbox_path
    
    async def execute_code(self, code, files=None):
        """Execute code in the E2B sandbox"""
        try:
            if not self.sandbox:
                await self.initialize()
            
            # Upload any files if provided
            file_paths = []
            if files:
                for file_data in files:
                    sandbox_path = await self.upload_file(file_data['path'], file_data['name'])
                    file_paths.append(sandbox_path)
            
            # If we have files, add some context code for loading them
            if file_paths:
                file_ext_mappings = {
                    'csv': 'pd.read_csv',
                    'xlsx': 'pd.read_excel',
                    'xls': 'pd.read_excel',
                    'json': 'pd.read_json',
                    'txt': 'open',
                    'py': 'open'
                }
                
                file_loader_code = "# File paths available for analysis:\n"
                
                for path in file_paths:
                    filename = os.path.basename(path)
                    ext = filename.split('.')[-1].lower() if '.' in filename else ''
                    var_name = os.path.splitext(filename)[0].replace(' ', '_').replace('-', '_').lower()
                    
                    if ext in file_ext_mappings:
                        if ext in ['txt', 'py']:
                            file_loader_code += f"# {path} - Use with: {var_name} = {file_ext_mappings[ext]}('{path}', 'r')\n"
                        else:
                            file_loader_code += f"# {path} - Use with: {var_name} = {file_ext_mappings[ext]}('{path}')\n"
                    else:
                        file_loader_code += f"# {path}\n"
                
                # Add code to import common libraries
                setup_code = """
# Import common data science libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set matplotlib to save figures instead of displaying them
plt.switch_backend('Agg')
plt.figure(figsize=(10, 6))  # Set a default figure size

# Save figures to the figures directory
import os
os.makedirs('/home/user/figures', exist_ok=True)

def save_current_figure(name='figure'):
    plt.savefig(f'/home/user/figures/{name}.png')
    plt.close()

"""
                code = setup_code + file_loader_code + "\n" + code
            
            # Execute the code using the proper method
            try:
                result = await self.sandbox.process.start_python(code)
            except Exception as e:
                # Handle execution errors by converting them to a string representation
                return {
                    'success': False,
                    'stdout': '',
                    'stderr': f"Execution error: {str(e)}",
                    'artifacts': []
                }
            
            # Ensure we're returning serializable data
            output = {
                'success': result.exit_code == 0 if hasattr(result, 'exit_code') else False,
                'stdout': str(result.stdout) if hasattr(result, 'stdout') else '',
                'stderr': str(result.stderr) if hasattr(result, 'stderr') else '',
                'artifacts': []
            }
            
            # Process any generated figures
            try:
                figures_dir = "/home/user/figures"
                await self.sandbox.filesystem.make_dir(figures_dir, recursive=True)
                
                # Check if the execution created any figures
                ls_result = await self.sandbox.process.start_and_wait(f"ls -la {figures_dir}")
                
                if "No such file or directory" not in ls_result.stdout:
                    # List all figure files
                    fig_files = await self.sandbox.filesystem.list(figures_dir)
                    
                    for file in fig_files:
                        if file.name.endswith(('.png', '.jpg', '.jpeg', '.svg')):
                            file_path = f"{figures_dir}/{file.name}"
                            file_content = await self.sandbox.filesystem.read(file_path)
                            
                            # Add the figure to artifacts
                            output['artifacts'].append({
                                'type': 'image',
                                'format': file.name.split('.')[-1],
                                'data': base64.b64encode(file_content).decode('utf-8')
                            })
            except Exception as e:
                output['stderr'] += f"\nError processing figures: {str(e)}"
            
            return output
        except Exception as e:
            # Catch any other exceptions
            return {
                'success': False,
                'stdout': '',
                'stderr': f"Internal error: {str(e)}",
                'artifacts': []
            }
    
    async def close(self):
        """Close the sandbox"""
        if self.sandbox:
            await self.sandbox.close()
            self.sandbox = None 