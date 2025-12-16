"""
Main Frame API - FastAPI with FastMCP integration
Provides tools for process generation and agent interaction
"""
from fastmcp import FastMCP
from fastapi import FastAPI, Request
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from pydantic import BaseModel
import asyncio
import logging
import time
import sqlite3

import os
os.environ["OPENAI_API_KEY"] = "your key here"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan_handler(app: FastAPI):
    """Initialize process agent on startup"""
    global process_agent
    logger.info("Starting Main Frame API - Initializing process agent...")
    try:
        process_agent = await init_process_agent()
        agent_ready.set()
        logger.info("Process agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize process agent: {e}")
        logger.warning("Process agent will be initialized on first use")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Main Frame API")

app = FastAPI(title="Main Frame API", lifespan=lifespan_handler)

def get_session_id() -> str:
    """Read session ID from session_id.txt file"""
    try:
        with open("session_id.txt", "r", encoding="utf-8") as f:
            return f.readline().strip()
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning(f"Failed to read session_id.txt: {e}")
        return None

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests"""
    start_time = time.time()
    
    # Log request details
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    logger.info(f"Client: {request.client.host if request.client else 'Unknown'}")
    
    # Process request
    response = await call_next(request)
    
    # Log response time
    process_time = time.time() - start_time
    logger.info(f"Completed in {process_time:.3f}s with status {response.status_code}")
    
    return response

# Initialize MCP client for process agent
process_agent_client = None
process_agent = None

async def init_process_agent():
    """Initialize the process agent with MCP tools"""
    global process_agent_client, process_agent
    
    process_agent_client = MultiServerMCPClient({
        "enervie": {
            "transport": "streamable_http",
            "url": "http://localhost:8000/mcp",
        }
    })
    
    # Get tools from the MCP server
    tools = await process_agent_client.get_tools()
    
    # Create the process agent
    system_prompt = """You are a Process Execution Agent. 
When instructed to execute a process by its process ID, you must use the 'retrieve_process' tool to look up the process description. 
After retrieving the process details, follow every step in the description exactly. 
Additionally, you may use any other tools provided to you as needed to complete the task.

**Handle small exceptions by yourself by analyzing possible tools and checking the SQLITE database to build workarounds.**
**If there are major issues, escalate the issue description to support@company.com using the send_mail tool.**
"""
    
    with open("seed.txt", "r", encoding="utf-8") as f:
        seed_value = int(f.readline().strip())
    
    llm_config = {
        "model": "gpt-5.1",
        #"service_tier": "priority",
        "temperature": 0,
        "seed": seed_value,
        "reasoning_effort": "high",
    }
    
    llm = ChatOpenAI(**llm_config)
    
    process_agent = create_agent(
        system_prompt=system_prompt,
        model=llm,
        tools=tools,
    )
    
    return process_agent

# Create event to signal when agent is ready
agent_ready = asyncio.Event()

class GenerateProcessRuleRequest(BaseModel):
    user_request: str
    process_rule_id: str

class GenerateProcessRuleFromBpmnRequest(BaseModel):
    bpmn_path: str
    process_rule_id: str

class AddProcessRuleRequest(BaseModel):
    process_rule_id: str
    process_rule: str

@app.post("/add_process_rule", operation_id="add_process_rule")
async def add_process_rule(request: AddProcessRuleRequest) -> Dict[str, Any]:
    """
    Directly add a process rule to the database without generation.
    
    Args:
        process_rule_id: ID to store the process rule under.
        process_rule: The process rule text to store.
    
    Returns:
        Dictionary with success status
    """
    try:
        # Write to database
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO process_rules (process_rule_id, process_rule) VALUES (?, ?)",
            (request.process_rule_id, request.process_rule)
        )
        conn.commit()
        conn.close()
        
        logger.info(f"Added process rule '{request.process_rule_id}' to database")
        
        return {
            "success": True,
            "process_rule_id": request.process_rule_id,
            "message": "Process rule added successfully"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to add process rule"
        }

@app.post("/generate_process_rule", operation_id="generate_process_rule")
async def generate_process_rule(request: GenerateProcessRuleRequest) -> Dict[str, Any]:
    """
    Generates a rule based on the user request.
    
    Args:
        user_request: The user request to generate the rule from.
        process_rule_id: ID to store the generated process rule under.
    
    Returns:
        Dictionary with generated rule
    """

    with open("seed.txt", "r", encoding="utf-8") as f:
        seed_value = int(f.readline().strip())
    
    try:
        llm = ChatOpenAI(
            model="gpt-5.1",
            temperature=0,
            seed=seed_value
        )
        
        response = llm.invoke(f"""You are an assistant that converts user requests into process steps an LLM agent will automatically execute to complete the process. 

Convert the user_request into unambiguous and pragmatic step that include important step information, decision rules.

- The steps need to be designed as clear, unambiguous instructions for an LLM agent.
- Use if-else statements to mark decision points.
- Nest every step of an if path within the if statement and every else path step within an else statement.
- Hold the formatting of the steps as simple as possible.

Convert the following user request into steps according to the instructions above:

{request.user_request}""")
        
        process_rule = response.content
        
        # Write to database
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO process_rules (process_rule_id, process_rule) VALUES (?, ?)",
            (request.process_rule_id, process_rule)
        )
        conn.commit()
        conn.close()
        
        logger.info(f"Saved process rule '{request.process_rule_id}' to database")
        
        # Save to session folder if session_id is set
        session_id = get_session_id()
        if session_id:
            session_dir = os.path.join("sessions", session_id)
            os.makedirs(session_dir, exist_ok=True)
            
            response_file = os.path.join(session_dir, "main_agent.txt")
            with open(response_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Action: generate_process_rule\n")
                f.write(f"Process Rule ID: {request.process_rule_id}\n")
                f.write(f"User Request: {request.user_request}\n")
                f.write(f"Generated Rule:\n{process_rule}\n")
            
            logger.info(f"Saved main agent response to {response_file}")
        
        return {
            "success": True,
            "process_rule_id": request.process_rule_id,
            "process_rule": process_rule
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to generate rule"
        }

@app.post("/generate_process_rule_from_bpmn", operation_id="generate_process_rule_from_bpmn")
async def generate_process_rule_from_bpmn(request: GenerateProcessRuleFromBpmnRequest) -> Dict[str, Any]:
    """
    Reads a BPMN XML file and converts it into LLM-executable process steps.
    
    Args:
        bpmn_path: Path to the BPMN XML file.
        process_rule_id: ID to store the generated process rule under.
    
    Returns:
        Dictionary with generated process rule text
    """
    try:
        # Read BPMN file
        with open(request.bpmn_path, "r") as f:
            process_bpmn = f.read()

        with open("seed.txt", "r", encoding="utf-8") as f:
            seed_value = int(f.readline().strip())

        # Create LangChain LLM
        llm = ChatOpenAI(
            model="gpt-5.1",
            service_tier="priority",
            temperature=0,
            seed=seed_value
        )

        # Run LLM transformation
        response = llm.invoke(f"""
You are an assistant that converts business process models defined with the business process model notation in XML into process steps an LLM agent will automatically run to complete the process. 

Convert the BPMN XML into unambiguous and pragmatic step that include important step information, decision rules, and the additional text annotations.

- The steps need to be designed as clear, unambiguous instructions for an LLM agent.
- Convert XOR statements to if-else statements based on the XOR gateway decision rules.
- Nest every step of an if path within the if statement and every else path step within an else statement.
- Linearize parallelization gateways into sequential steps.
- Include all additional text annotations from the BPMN XML in the respective step instruction.
- Only refer to the multi-agent system lane. Do not include steps from other lanes.
- Hold the formatting of the steps as simple as possible.
- Make this clear and precise steps. The need to be correctly in sequential order.
- Hold text short but include **all** necessary information.

Convert the following XML BPMN modeling according to the instructions above:

{process_bpmn}
""")

        process_rule = response.content
        
        # Write to database
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO process_rules (process_rule_id, process_rule) VALUES (?, ?)",
            (request.process_rule_id, process_rule)
        )
        conn.commit()
        conn.close()
        
        logger.info(f"Saved process rule '{request.process_rule_id}' to database")
        
        # Save to session folder if session_id is set
        session_id = get_session_id()
        if session_id:
            session_dir = os.path.join("sessions", session_id)
            os.makedirs(session_dir, exist_ok=True)
            
            response_file = os.path.join(session_dir, "main_agent.txt")
            with open(response_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Action: generate_process_rule_from_bpmn\n")
                f.write(f"Process Rule ID: {request.process_rule_id}\n")
                f.write(f"BPMN Path: {request.bpmn_path}\n")
                f.write(f"Generated Rule:\n{process_rule}\n")
            
            logger.info(f"Saved main agent response to {response_file}")

        return {
            "success": True,
            "process_rule_id": request.process_rule_id,
            "process_rule": process_rule
        }
        
    except FileNotFoundError:
        return {
            "success": False,
            "error": "File not found",
            "message": f"BPMN file not found: {request.bpmn_path}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to generate process rule from BPMN"
        }

@app.post("/call_process_agent", operation_id="call_process_agent")
async def call_process_agent(query: str) -> Dict[str, Any]:
    """
    Agent that executes processes based on their ID.
    This endpoint forwards queries to the process agent.
    
    Args:
        query: The query to send to the process agent
    
    Returns:
        Dictionary with agent response
    """
    global process_agent
    
    try:
        # Initialize agent if not already done
        if process_agent is None:
            await agent_ready.wait()
            if process_agent is None:
                process_agent = await init_process_agent()
        
        # Invoke the process agent
        response = await process_agent.ainvoke({
            "messages": [{"role": "user", "content": query}]
        })
        
        # Extract the final message content
        final_message = response["messages"][-1].content if response["messages"] else "No response from agent."
        
        # Save response to session folder if session_id is set
        session_id = get_session_id()
        if session_id:
            session_dir = os.path.join("sessions", session_id)
            os.makedirs(session_dir, exist_ok=True)
            
            response_file = os.path.join(session_dir, "process_agent.txt")
            with open(response_file, "w", encoding="utf-8") as f:
                f.write("\n".join([x.pretty_repr() for x in response["messages"]]))
            
            logger.info(f"Saved process agent response to {response_file}")
        
        return {
            "success": True,
            "response": final_message,
            "message": "Process agent executed successfully"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to call process agent"
        }

# Generate MCP server from the API
mcp = FastMCP.from_fastapi(app=app, name="Main Frame API")

# Create the MCP's ASGI app
mcp_app = mcp.http_app(path='/mcp')

# Combine lifespan contexts
@asynccontextmanager
async def combined_lifespan(app: FastAPI):
    """Combined lifespan for both app and mcp_app"""
    # Start main app lifespan
    async with lifespan_handler(app):
        # Start mcp app lifespan
        async with mcp_app.lifespan(app):
            yield

# Create a new FastAPI app that combines both sets of routes
combined_app = FastAPI(
    title="Main Frame API with MCP",
    routes=[
        *mcp_app.routes,  # MCP routes
        *app.routes,      # Original API routes
    ],
    lifespan=combined_lifespan,
)

# Run with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(combined_app, host="0.0.0.0", port=8001)
