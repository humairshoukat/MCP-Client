# MCP Client

This is a simple MCP client implementation using mcp-use, powered by LLM of your choice (Google's Gemini, OpenAI's GPT, Open Source LLM models using Ollama) that provides interface to connect with multiple MCP servers and use their tools.

## Setup

1. Make sure you have Python 3.12 or higher installed
2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the project root with your custom configurations & API keys (if needed):
   ```
   # LLM Configurations
   LLM_PROVIDER=name_of_llm_provider  # openai, google, ollama
   TEMPERATURE=0.2
   MAX_TOKENS=4096

   # If you're using OpenAI as LLM Provider
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-4o

   # If you're using Google as LLM Provider
   GOOGLE_API_KEY=your_google_api_key_here
   GEMINI_MODEL=gemini-2.5-flash

   # If you're using Ollama as LLM Provider
   OLLAMA_BASE_URL=http://localhost:11434  # The default endpoint when to access ollama model
   OLLAMA_MODEL=llama3.2:latest
   
   # Your MCP servers urls here (add multiple mcp server urls if needed)
   MCP_SERVER_URL=your_mcp_server_url_here
   ANY_MCP_SERVER_URL=your_any_mcp_server_url_here
   ```

## Running the Client

Start the client with:
```bash
python main.py
```

The client will start on `http://localhost:5000` by default.

## Connecting to the MCP Server

To use this client with your custom MCP server:

1. Configure and add your MCP server URL in the code:
   ```
   config = {
            "mcpServers": {
                "mcp-server": {
                    "url": os.getenv("MCP_SERVER_URL")
                }
            }
        }
   ```
2. Use the `http://localhost:5000/chat` endpoint to chat with the AI Agent which has access to the LLM and MCP Tools.

## API Keys Required

- **OpenAI API**: Get from [OpenAI](https://platform.openai.com/api-keys) (if you're using GPT as LLM)
- **Google API**: Get from [Google](https://ai.google.dev/gemini-api/docs/api-key) (if you're using Gemini as LLM)