# Code Interpreter Web Application

A Flask-based web application that provides a conversational AI interface with code interpretation capabilities. This application allows users to upload files, analyze data, and interact with an AI assistant that can run code to process and visualize data.

## Features

- Interactive chat interface with AI assistant
- Code interpretation and execution
- File upload and analysis support
- Data visualization capabilities
- Memory of conversation context

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- E2B API key (for sandbox code execution)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Posiitron/PythonE2B/
   cd PythonE2B
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   E2B_API_KEY=your_e2b_api_key
   ```

## Usage

1. Start the application:
   ```bash
   python app.py
   ```

2. Open a web browser and navigate to `http://127.0.0.1:5000`

3. Use the interface to:
   - Upload files for analysis
   - Ask questions about your data
   - Run code through the AI assistant

## Environment Variables

- `OPENAI_API_KEY` - Your OpenAI API key
- `E2B_API_KEY` - Your E2B API key for sandboxed code execution

## Supported File Types 

The application supports uploading and analyzing various file formats, including:
- Text files (.txt)
- CSV files (.csv)
- Excel files (.xlsx, .xls)
- Images (.png, .jpg, .jpeg, .gif)
- PDF files (.pdf)
- JSON files (.json)

## Known Issues

**Important Note:** The data analysis functionality is still in development and might occasionally mistake your precious data for confetti or interpret your CSV as an avant-garde poem. If the app starts analyzing your financial spreadsheet as "a beautiful sonnet about economic uncertainty," please be patient with us. We're working on teaching it the difference between Shakespeare and spreadsheets.
