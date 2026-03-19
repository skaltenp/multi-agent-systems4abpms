import asyncio
import json
import os
import random
import time
from datetime import timedelta
from itertools import product

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.errors import GraphRecursionError

from config.experiment_config import ExperimentConfig
from config.process_adaptations import BPMN_RULES, CLASSIC_RULES, TEXT_RULES
from config.server_config import ClassicAgentPromptConfig
from experiment.message_tracker import save_messages_json
from experiment.session import SessionManager


class ExperimentRunner:
    FRAME_AGENT_SYSTEM_PROMPT = """You are a process-frame agent with three main capabilities.
First, you can take a BPMN file path and generate a structured process rule from it, automatically storing it in the rule database along with an ID.
Second, you can take a natural language description from a human and also convert that into a structured process rule, automatically writing it to the database with an ID.
Third, if a user requests that a specific process ID be run, you can call a process-execution agent to handle that execution.
Always follow these steps and ensure each rule is stored and each process is run as requested."""

    FRAME_TIMEOUT = 900  # 15 minutes
    PROCESS_TIMEOUT = 1800  # 30 minutes
    RECURSION_LIMIT = 100

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.session_mgr = SessionManager(
            db_path=config.db_path,
            sessions_folder=config.sessions_folder,
            experiment_type=config.experiment_type,
        )

    @staticmethod
    def _write_limits(session_dir: str, limits: dict) -> None:
        with open(os.path.join(session_dir, "limits_errors.json"), "w", encoding="utf-8") as f:
            json.dump(limits, f, indent=2)

    def generate_runs(self) -> list[tuple]:
        all_runs = list(
            product(
                self.config.process_adaptations,
                self.config.rule_adaptation_methods,
                self.config.tours,
                self.config.seeds,
            )
        )
        if self.config.sample_size is not None and self.config.sample_size < len(all_runs):
            random.seed(42)
            all_runs = random.sample(all_runs, self.config.sample_size)
        return all_runs

    def _build_rule_prompt(self, process_adaptation: str, method: str, tour: str) -> str:
        if method == "add":
            template = TEXT_RULES[process_adaptation]
            filled = template.format(tour=tour)
            return f"""**Add** this process rule with id '{tour}' to the rule database:\n\n{filled}"""
        elif method == "generate_bpmn":
            return BPMN_RULES[process_adaptation][tour]
        elif method == "classic":
            template = CLASSIC_RULES[process_adaptation]
            return template.format(tour=tour)
        else:
            raise ValueError(f"Unknown method: {method}")

    async def run_single(self, process_adaptation: str, method: str, tour: str, seed: int) -> None:
        session_id = self.session_mgr.build_session_id(
            process_adaptation, method, tour, seed, self.config.experiment_type
        )

        if self.session_mgr.session_exists(session_id):
            print(f"Session {session_id} already exists. Skipping...")
            return

        self.session_mgr.write_seed(seed)
        self.session_mgr.reset_database()

        client = MultiServerMCPClient(
            {
                "fastapi": {
                    "transport": "streamable_http",
                    "url": f"http://localhost:{self.config.frame_server_port}/mcp",
                    "timeout": timedelta(minutes=30),
                    "sse_read_timeout": timedelta(minutes=30),
                }
            }
        )
        tools = await client.get_tools()

        model = self.config.model_config.create_llm(seed=seed, reasoning_effort="high")

        frame_agent = create_agent(
            model,
            tools=tools,
            system_prompt=self.FRAME_AGENT_SYSTEM_PROMPT,
        )

        # Phase 1: Add/generate process rule
        prompt = self._build_rule_prompt(process_adaptation, method, tour)

        self.session_mgr.write_session_id(session_id)

        session_dir = os.path.join(self.config.sessions_folder, session_id)
        os.makedirs(session_dir, exist_ok=True)

        limits = {
            "recursion_limit_frame": False, "recursion_limit_process": False,
            "timeout_frame": False, "timeout_process": False,
            "error_frame": False, "error_process": False,
        }

        print(f"Adding process rule for session {session_id}...")
        st = time.time()

        try:
            response = await asyncio.wait_for(
                frame_agent.ainvoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    {"recursion_limit": self.RECURSION_LIMIT},
                ),
                timeout=self.FRAME_TIMEOUT,
            )
        except asyncio.TimeoutError:
            limits["timeout_frame"] = True
            print(f"Frame agent timed out during rule addition after {self.FRAME_TIMEOUT}s")
            response = None
        except GraphRecursionError:
            limits["recursion_limit_frame"] = True
            print(f"Frame agent hit recursion limit ({self.RECURSION_LIMIT}) during rule addition")
            response = None
        except Exception as e:
            limits["error_frame"] = True
            print(f"Frame agent error during rule addition: {e}")
            response = None

        if response:
            with open(os.path.join(session_dir, "frame_agent_process_addition.txt"), "w", encoding="utf-8") as f:
                for m in response["messages"]:
                    f.write(m.pretty_repr() + "\n")
            save_messages_json(response["messages"], session_dir, "frame_agent_process_addition.json")

        rule_et = time.time() - st
        print(f"Rule addition time: {rule_et} seconds")

        rule_added = self.session_mgr.verify_rule_added(tour)

        with open(os.path.join(session_dir, "frame_metadata.csv"), "w", encoding="utf-8") as f:
            f.write("rule_added\n")
            f.write(f"{rule_added}\n")

        if not rule_added:
            print(f"Rule generation failed for session {session_id}. Skipping process execution.")
            with open(os.path.join(session_dir, "timings.csv"), "w", encoding="utf-8") as f:
                f.write("agent,time\n")
                f.write(f"frame,{rule_et}\n")
            self._write_limits(session_dir, limits)
            return

        # Phase 2: Run process rule
        print(f"Running process rule for session {session_id}...")
        process_prompt = f"Run process_rule with process_rule_id '{tour}'."

        st = time.time()

        try:
            response = await asyncio.wait_for(
                frame_agent.ainvoke(
                    {"messages": [{"role": "user", "content": process_prompt}]},
                    {"recursion_limit": self.RECURSION_LIMIT},
                ),
                timeout=self.PROCESS_TIMEOUT,
            )
        except asyncio.TimeoutError:
            limits["timeout_process"] = True
            print(f"Frame agent timed out during process execution after {self.PROCESS_TIMEOUT}s")
            response = None
        except GraphRecursionError:
            limits["recursion_limit_process"] = True
            print(f"Frame agent hit recursion limit ({self.RECURSION_LIMIT}) during process execution")
            response = None
        except Exception as e:
            limits["error_process"] = True
            print(f"Frame agent error during process execution: {e}")
            response = None

        if response:
            with open(os.path.join(session_dir, "frame_agent_process_execution.txt"), "w", encoding="utf-8") as f:
                for m in response["messages"]:
                    f.write(m.pretty_repr() + "\n")
            save_messages_json(response["messages"], session_dir, "frame_agent_process_execution.json")

        process_et = time.time() - st
        print(f"Process execution time: {process_et} seconds")

        with open(os.path.join(session_dir, "timings.csv"), "w", encoding="utf-8") as f:
            f.write("agent,time\n")
            f.write(f"frame,{rule_et}\n")
            f.write(f"process,{process_et}\n")

        self._write_limits(session_dir, limits)

    async def run_single_classic(self, process_adaptation: str, method: str, tour: str, seed: int) -> None:
        """Run a classic agent baseline: single agent with prose prompt, no frame agent, no rule storage."""
        session_id = self.session_mgr.build_session_id(
            process_adaptation, method, tour, seed, self.config.experiment_type
        )

        if self.session_mgr.session_exists(session_id):
            print(f"Session {session_id} already exists. Skipping...")
            return

        self.session_mgr.write_seed(seed)
        self.session_mgr.reset_database()

        client = MultiServerMCPClient(
            {
                "enervie": {
                    "transport": "streamable_http",
                    "url": f"http://localhost:{self.config.classic_server_port}/mcp",
                    "timeout": timedelta(minutes=30),
                    "sse_read_timeout": timedelta(minutes=30),
                }
            }
        )
        tools = await client.get_tools()

        system_prompt = ClassicAgentPromptConfig.get_prompt(self.config.experiment_type)
        model = self.config.model_config.create_llm(seed=seed, reasoning_effort="high")

        classic_agent = create_agent(
            model,
            tools=tools,
            system_prompt=system_prompt,
        )

        prompt = self._build_rule_prompt(process_adaptation, method, tour)

        self.session_mgr.write_session_id(session_id)

        session_dir = os.path.join(self.config.sessions_folder, session_id)
        os.makedirs(session_dir, exist_ok=True)

        CLASSIC_AGENT_RECURSION_LIMIT = 100
        CLASSIC_AGENT_TIMEOUT = 2700

        print(f"Running classic agent for session {session_id}...")
        st = time.time()

        hit_recursion_limit = False
        hit_timeout = False
        last_state = None

        async def _run_classic_agent():
            nonlocal last_state
            async for state in classic_agent.astream(
                {"messages": [{"role": "user", "content": prompt}]},
                {"recursion_limit": CLASSIC_AGENT_RECURSION_LIMIT},
                stream_mode="values",
            ):
                last_state = state

        hit_error = False

        try:
            await asyncio.wait_for(_run_classic_agent(), timeout=CLASSIC_AGENT_TIMEOUT)
        except asyncio.TimeoutError:
            hit_timeout = True
            print(f"Classic agent timed out after {CLASSIC_AGENT_TIMEOUT}s")
        except GraphRecursionError:
            hit_recursion_limit = True
            print(f"Classic agent hit recursion limit ({CLASSIC_AGENT_RECURSION_LIMIT})")
        except Exception as e:
            hit_error = True
            print(f"Classic agent error: {e}")

        messages = last_state.get("messages", []) if last_state else []

        with open(os.path.join(session_dir, "classic_agent.txt"), "w", encoding="utf-8") as f:
            for m in messages:
                f.write(m.pretty_repr() + "\n")
        save_messages_json(messages, session_dir, "classic_agent.json")
        # Also write as process_agent.txt so evaluator can find send_bulk_mail calls
        with open(os.path.join(session_dir, "process_agent.txt"), "w", encoding="utf-8") as f:
            for m in messages:
                f.write(m.pretty_repr() + "\n")

        et = time.time() - st
        print(f"Total time: {et} seconds")

        with open(os.path.join(session_dir, "timings.csv"), "w", encoding="utf-8") as f:
            f.write("agent,time\n")
            f.write(f"classic,{et}\n")

        self._write_limits(session_dir, {
            "recursion_limit_frame": False,
            "recursion_limit_process": hit_recursion_limit,
            "timeout_frame": False,
            "timeout_process": hit_timeout,
            "error_frame": False,
            "error_process": hit_error,
        })

    async def run_all(self) -> None:
        self.session_mgr.write_sessions_folder()
        runs = self.generate_runs()
        print(f"Generated {len(runs)} experiment runs")
        failed = []
        for run in runs:
            process_adaptation, method, tour, seed = run
            try:
                if method == "classic":
                    await self.run_single_classic(process_adaptation, method, tour, seed)
                else:
                    await self.run_single(process_adaptation, method, tour, seed)
            except Exception as e:
                session_id = self.session_mgr.build_session_id(
                    process_adaptation, method, tour, seed, self.config.experiment_type
                )
                print(f"ERROR: Run {session_id} failed: {e}")
                failed.append(session_id)
                # Ensure session dir + limits_errors.json exist so the run is skipped on retry
                session_dir = os.path.join(self.config.sessions_folder, session_id)
                os.makedirs(session_dir, exist_ok=True)
                limits_path = os.path.join(session_dir, "limits_errors.json")
                if not os.path.exists(limits_path):
                    self._write_limits(session_dir, {
                        "recursion_limit_frame": False,
                        "recursion_limit_process": False,
                        "timeout_frame": False,
                        "timeout_process": False,
                        "error_frame": method != "classic",
                        "error_process": method == "classic",
                    })
        if failed:
            print(f"\n{len(failed)} run(s) failed:")
            for s in failed:
                print(f"  - {s}")
            print("Delete their session folders to retry.")
