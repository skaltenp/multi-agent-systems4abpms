"""
Unified frame server - replaces main_frame*.py
Usage: python -m servers.frame_server --experiment-type base --port 8001
"""
import argparse
import asyncio
import logging
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastmcp import FastMCP

load_dotenv()
load_dotenv(".env.ollama")
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel

from config.model_config import ModelConfig
from config.server_config import ExperimentType, ProcessAgentPromptConfig
from langgraph.errors import GraphRecursionError

from experiment.message_tracker import save_messages_json

PROCESS_AGENT_RECURSION_LIMIT = 100
DB_PATH = "database.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_session_id() -> str | None:
    try:
        with open("session_id.txt", "r", encoding="utf-8") as f:
            return f.readline().strip()
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning(f"Failed to read session_id.txt: {e}")
        return None


def get_sessions_folder() -> str:
    try:
        with open("sessions_folder.txt", "r", encoding="utf-8") as f:
            return f.readline().strip()
    except FileNotFoundError:
        return "sessions"


def create_app(experiment_type: ExperimentType, model_config: ModelConfig, process_server_port: int = 8000) -> FastAPI:
    process_agent_client = None
    process_agent_tools = None
    process_agent_prompt = None
    client_ready = asyncio.Event()

    async def init_mcp_client():
        nonlocal process_agent_client, process_agent_tools, process_agent_prompt

        process_agent_client = MultiServerMCPClient(
            {
                "enervie": {
                    "transport": "streamable_http",
                    "url": f"http://localhost:{process_server_port}/mcp",
                    "timeout": timedelta(minutes=30),
                    "sse_read_timeout": timedelta(minutes=30),
                }
            }
        )
        process_agent_tools = await process_agent_client.get_tools()
        process_agent_prompt = ProcessAgentPromptConfig.get_prompt(experiment_type)

    def create_process_agent(seed_value: int):
        llm = model_config.create_llm(seed=seed_value, reasoning_effort="high")
        return create_agent(
            system_prompt=process_agent_prompt,
            model=llm,
            tools=process_agent_tools,
        )

    @asynccontextmanager
    async def lifespan_handler(app: FastAPI):
        logger.info("Starting Main Frame API - Initializing MCP client...")
        try:
            await init_mcp_client()
            logger.info("MCP client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")
            logger.warning("MCP client will be initialized on first use")
        finally:
            client_ready.set()
        yield
        logger.info("Shutting down Main Frame API")

    app = FastAPI(title="Main Frame API", lifespan=lifespan_handler)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        logger.info(f"Incoming request: {request.method} {request.url.path}")
        logger.info(f"Client: {request.client.host if request.client else 'Unknown'}")
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Completed in {process_time:.3f}s with status {response.status_code}")
        return response

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
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO process_rules (process_rule_id, process_rule) VALUES (?, ?)",
                    (request.process_rule_id, request.process_rule),
                )

            logger.info(f"Added process rule '{request.process_rule_id}' to database")
            return {
                "success": True,
                "process_rule_id": request.process_rule_id,
                "message": "Process rule added successfully",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "message": "Failed to add process rule"}

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

        RULE_GENERATION_TIMEOUT = 900  # 15 minutes

        try:
            llm = model_config.create_llm(seed=seed_value, reasoning_effort="high")

            response = await asyncio.wait_for(
                llm.ainvoke(
                    f"""You are an assistant that converts user requests into process steps an LLM agent will automatically execute to complete the process.

Convert the user_request into unambiguous and pragmatic step that include important step information, decision rules.

- The steps need to be designed as clear, unambiguous instructions for an LLM agent.
- Use if-else statements to mark decision points.
- Nest every step of an if path within the if statement and every else path step within an else statement.
- Hold the formatting of the steps as simple as possible.

Convert the following user request into steps according to the instructions above:

{request.user_request}"""
                ),
                timeout=RULE_GENERATION_TIMEOUT,
            )

            process_rule = response.content

            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO process_rules (process_rule_id, process_rule) VALUES (?, ?)",
                    (request.process_rule_id, process_rule),
                )

            logger.info(f"Saved process rule '{request.process_rule_id}' to database")

            return {
                "success": True,
                "process_rule_id": request.process_rule_id,
                "process_rule": process_rule,
            }
        except asyncio.TimeoutError:
            logger.warning(f"Rule generation timed out after {RULE_GENERATION_TIMEOUT}s")
            return {"success": False, "error": "Timeout", "message": f"Rule generation timed out after {RULE_GENERATION_TIMEOUT}s"}
        except Exception as e:
            return {"success": False, "error": str(e), "message": "Failed to generate rule"}

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
        RULE_GENERATION_TIMEOUT = 900  # 15 minutes

        try:
            with open(request.bpmn_path, "r") as f:
                process_bpmn = f.read()

            with open("seed.txt", "r", encoding="utf-8") as f:
                seed_value = int(f.readline().strip())

            llm = model_config.create_llm(seed=seed_value, reasoning_effort="high")

            response = await asyncio.wait_for(llm.ainvoke(
                f"""
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
"""
            ), timeout=RULE_GENERATION_TIMEOUT)

            process_rule = response.content

            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO process_rules (process_rule_id, process_rule) VALUES (?, ?)",
                    (request.process_rule_id, process_rule),
                )

            logger.info(f"Saved process rule '{request.process_rule_id}' to database")

            return {
                "success": True,
                "process_rule_id": request.process_rule_id,
                "process_rule": process_rule,
            }
        except asyncio.TimeoutError:
            logger.warning(f"BPMN rule generation timed out after {RULE_GENERATION_TIMEOUT}s")
            return {"success": False, "error": "Timeout", "message": f"BPMN rule generation timed out after {RULE_GENERATION_TIMEOUT}s"}
        except FileNotFoundError:
            return {
                "success": False,
                "error": "File not found",
                "message": f"BPMN file not found: {request.bpmn_path}",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "message": "Failed to generate process rule from BPMN"}

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
        try:
            if process_agent_tools is None:
                await client_ready.wait()
                if process_agent_tools is None:
                    await init_mcp_client()

            with open("seed.txt", "r", encoding="utf-8") as f:
                seed_value = int(f.readline().strip())
            process_agent = create_process_agent(seed_value)
            logger.info(f"Created process agent with seed={seed_value}")

            PROCESS_AGENT_TIMEOUT = 1800

            st = time.time()
            hit_recursion_limit = False
            hit_timeout = False
            last_state = None

            async def _run_process_agent():
                nonlocal last_state
                async for state in process_agent.astream(
                    {"messages": [{"role": "user", "content": query}]},
                    {"recursion_limit": PROCESS_AGENT_RECURSION_LIMIT},
                    stream_mode="values",
                ):
                    last_state = state

            try:
                await asyncio.wait_for(_run_process_agent(), timeout=PROCESS_AGENT_TIMEOUT)
            except asyncio.TimeoutError:
                hit_timeout = True
                logger.warning(
                    f"Process agent timed out after {PROCESS_AGENT_TIMEOUT}s"
                )
            except GraphRecursionError:
                hit_recursion_limit = True
                logger.warning(
                    f"Process agent hit recursion limit ({PROCESS_AGENT_RECURSION_LIMIT})"
                )

            process_agent_time = time.time() - st
            messages = last_state.get("messages", []) if last_state else []
            final_message = messages[-1].content if messages else "No response from agent."

            session_id = get_session_id()
            if session_id:
                session_dir = os.path.join(get_sessions_folder(), session_id)
                os.makedirs(session_dir, exist_ok=True)
                response_file = os.path.join(session_dir, "process_agent.txt")
                with open(response_file, "w", encoding="utf-8") as f:
                    f.write("\n".join([x.pretty_repr() for x in messages]))
                save_messages_json(messages, session_dir, "process_agent.json")
                timings_path = os.path.join(session_dir, "timings.csv")
                with open(timings_path, "a", encoding="utf-8") as f:
                    f.write(f"process,{process_agent_time}\n")
                logger.info(f"Saved process agent response to {response_file}")
                logger.info(f"Process agent completed in {process_agent_time:.1f}s")
                if hit_recursion_limit:
                    logger.warning(f"Session {session_id}: saved partial messages after recursion limit")
                if hit_timeout:
                    logger.warning(f"Session {session_id}: saved partial messages after timeout")

            if hit_timeout:
                status_msg = f"Process agent timed out after {PROCESS_AGENT_TIMEOUT}s"
            elif hit_recursion_limit:
                status_msg = "Process agent hit recursion limit"
            else:
                status_msg = "Process agent executed successfully"

            return {
                "success": not hit_timeout and not hit_recursion_limit,
                "response": final_message,
                "message": status_msg,
                "hit_recursion_limit": hit_recursion_limit,
                "hit_timeout": hit_timeout,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "message": "Failed to call process agent"}

    return app


def main():
    parser = argparse.ArgumentParser(description="Main Frame Server")
    parser.add_argument(
        "--experiment-type",
        type=str,
        default="base",
        choices=["base", "exception_handling", "exception_handling_db_error"],
    )
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--process-server-port", type=int, default=8000)
    parser.add_argument("--model", type=str, default=None, help="Model name (or set LLM_MODEL env var). Non-gpt/claude models auto-use Ollama.")
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI-compatible base URL (or set LLM_BASE_URL env var)")
    args = parser.parse_args()

    experiment_type = ExperimentType(args.experiment_type)
    model_kwargs = {"temperature": 0}
    if args.model:
        model_kwargs["model"] = args.model
    if args.base_url:
        model_kwargs["base_url"] = args.base_url
    model_cfg = ModelConfig(**model_kwargs)
    model_cfg.validate_model()

    app = create_app(experiment_type, model_cfg, process_server_port=args.process_server_port)

    mcp = FastMCP.from_fastapi(app=app, name="Main Frame API")
    mcp_app = mcp.http_app(path="/mcp", stateless_http=True)

    @asynccontextmanager
    async def combined_lifespan(combined: FastAPI):
        async with app.router.lifespan_context(app):
            async with mcp_app.lifespan(combined):
                yield

    combined_app = FastAPI(
        title="Main Frame API with MCP",
        routes=[*mcp_app.routes, *app.routes],
        lifespan=combined_lifespan,
    )

    import uvicorn

    uvicorn.run(combined_app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
