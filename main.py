from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import json
import logging
import uvicorn
from dotenv import load_dotenv
from json_repair import repair_json
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from mcp_use import MCPAgent, MCPClient

# Load environment variables from .env file (if needed)
load_dotenv()

# Set up logging
logger = logging.getLogger("MCP-Client")

# Create AI Agent with LLM and MCP Server
async def create_agent():
    try:
        # Create MCP servers config
        config = {
            "mcpServers": {
                "mcp-server": {
                    "url": os.getenv("MCP_SERVER_URL")
                }
            }
        }
        
        # Create MCP Client from config
        client = MCPClient.from_dict(config)
        
        # Create LLM
        provider = os.getenv("LLM_PROVIDER", "google").lower()
        logger.info(f"LLM Provider: {provider}")
        if provider == "openai":
            llm = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                temperature=float(os.getenv("TEMPERATURE", "0.2")),
                max_tokens=int(os.getenv("MAX_TOKENS", "4096")),
            )
        elif provider == "google":
            llm = ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                temperature=float(os.getenv("TEMPERATURE", "0.2")),
                max_tokens=int(os.getenv("MAX_TOKENS", "4096")),
            )
        elif provider == "ollama":
            llm = ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "gpt-oss"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=float(os.getenv("TEMPERATURE", "0.2")),
            )
        else:
            raise ValueError(f"Unknown LLM Provider: {provider}")
        
        # Create agent with MCP client and LLM
        agent = MCPAgent(llm=llm, client=client, max_steps=60)

        logger.info("AI Agent initialized successfully")
        return agent

    except Exception as e:
        logger.error(f"Failed to create AI Agent: {e}")
        return None

app = FastAPI(
    title="MCP Client API",
    description="REST API for MCP Client",
    version="1.0.0"
)

# Request model
class RequestModel(BaseModel):
    query: str

# Response model
class ResponseModel(BaseModel):
    success: bool
    tool: Optional[str]
    summary: Optional[str]
    data: Optional[Any]

# API Endpoints
@app.get("/")
async def status():
    return {
        "status": "OK",
        "message": "Welcome to the MCP Client API",
        "docs": "/docs"
    }

@app.post("/chat", response_model=ResponseModel)
async def chat(request: RequestModel):
    query = request.query   
    logger.info(f"Received query: {query}")

    try:
        prompt = f"""
                    You are a helpful AI assistant designed to assist users with their queries using the tools available to you.

                    User Query: "{query}"

                    For the above user query, first decide if a tool call is needed. If so, call the tool and use its output. 
                    But ALWAYS reply to the user conversationally, in a natural and friendly chatbot style, in the `summary` field—do not just echo tool data or say "see the data".

                    Note: Your response must be a JSON object with EXACTLY below fields in this order:

                    {{
                    "success": <true | false>,
                    "tool": <string|null>,            # e.g. "web_search", "select_best_vpn_server" OR null if no tool used
                    "summary": <string>,              # FRIENDLY, conversational and user-facing answer (not just a restatement of the tool or a generic string)
                    "data": <object|array|string|null> # Raw tool output, or null if not applicable (do not append any data from your end)
                    }}

                    **Rules:**
                    1. Always fill the `summary` field with a natural, helpful, complete answer as if chatting with a user.
                    2. NEVER include any commentary, thoughts, or intermediate steps like "Thought", "Action", or "Observation".
                    3. ALWAYS return the JSON object directly—no markdown, no code fences, no additional text.
                    4. If something goes wrong, reply with `success: false`, `tool: null`, brief error description in `summary`, and `data: null`.
                    5. Only if a paid user talks about some mobile app/game/streaming service in the query, then use the mobile_app_download_links tool to find their download links and add them in the summary and response 'data' object.

                    **Examples:**

                    Query: "What is MCP Protocol?"
                    Response:
                    {{
                    "success": true,
                    "tool": "web_search",
                    "summary": "MCP Stands for Model Context Protocol. It is a protocol for AI agents to interact with external tools.",
                    "data": {{ "web_search_response" }}
                    }}

                    Query: "What's the weather in Islamabad?"
                    Response:
                    {{
                    "success": true,
                    "tool": "check_weather",
                    "summary": "The current weather in Islamabad is 25°C with clear skies.",
                    "data": {{ "check_weather_response" }}
                    }}
                    
                    Strictly follow these rules to ensure a smooth user experience. Do not deviate from the JSON format or the conversational style.
                    Validate json format for any errors before returning the response. SO that the response is always a valid JSON object.
                    """

        agent = await create_agent()
        if not agent:
            return {
                "success": False,
                "tool": None,
                "summary": "Failed to initialize AI Agent",
                "data": None
            }
        
        result = await agent.run(prompt)

        # Handle Rate Limit Error
        # if type(result) != dict:  # For OpenAI Response
        if 'error: 429' in str(result.lower()): # For Gemini Response
            return {
                "success": False,
                "tool": None,
                "summary": "Received an empty response from the AI model. Which means API is down, or you have hit the rate limit",
                "data": None
            }
        
        result = result.strip("```").strip("json")
        logger.info(f"Response: {result}")
        fixed_json = repair_json(result)
        result = json.loads(fixed_json)

        # Return the llm response
        return result
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        
        if "expecting value" in str(e).lower():
            summary = "Received an empty response from the AI model. Which means API is down, or you have hit the rate limit."
        elif "timeout" in str(e).lower():
            summary = "The request timed out. Please try again later."
        else:
            summary = "Sorry, I can't help you with this right now, Please try again."

        return {
            "success": False,
            "summary": summary,
            "tool": None,
            "data": None
        }

# Run the app (MCP Client)
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )
