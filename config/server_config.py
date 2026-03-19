from dataclasses import dataclass
from enum import Enum


class ExperimentType(Enum):
    BASE = "base"
    EXCEPTION_HANDLING = "exception_handling"
    EXCEPTION_HANDLING_DB_ERROR = "exception_handling_db_error"

    @property
    def session_suffix(self) -> str:
        if self == ExperimentType.BASE:
            return ""
        elif self == ExperimentType.EXCEPTION_HANDLING:
            return "_exception_handling"
        elif self == ExperimentType.EXCEPTION_HANDLING_DB_ERROR:
            return "_exception_handling_db_error"


class ProcessAgentPromptConfig:
    BASE_PROMPT = """You are a Process Execution Agent.
When instructed to execute a process by its process ID, you must use the 'retrieve_process' tool to look up the process description.
After retrieving the process details, follow every step in the description exactly.
Additionally, you may use any other tools provided to you as needed to complete the task."""

    EXCEPTION_HANDLING_ADDENDUM = """
**Handle small exceptions by yourself by analyzing possible tools and checking the SQLITE database to build workarounds.**
**If there are major issues, escalate the issue description to support@company.com using the send_mail tool.**
"""

    @classmethod
    def get_prompt(cls, experiment_type: ExperimentType) -> str:
        if experiment_type == ExperimentType.BASE:
            return cls.BASE_PROMPT
        return cls.BASE_PROMPT + cls.EXCEPTION_HANDLING_ADDENDUM


class ClassicAgentPromptConfig:
    BASE_PROMPT = """You are a Process Execution Agent.
You will receive instructions to execute a process from the user.
After receiving the process details, follow every step in the description exactly.
Additionally, you may use any other tools provided to you as needed to complete the task."""

    EXCEPTION_HANDLING_ADDENDUM = """
**Handle small exceptions by yourself by analyzing possible tools and checking the SQLITE database to build workarounds.**
**If there are major issues, escalate the issue description to support@company.com using the send_mail tool.**
"""

    @classmethod
    def get_prompt(cls, experiment_type: ExperimentType) -> str:
        if experiment_type == ExperimentType.BASE:
            return cls.BASE_PROMPT
        return cls.BASE_PROMPT + cls.EXCEPTION_HANDLING_ADDENDUM


@dataclass
class ToolBehaviorConfig:
    simulate_db_error: bool = False
    use_schema_typo: bool = False

    @classmethod
    def from_experiment_type(cls, experiment_type: ExperimentType) -> "ToolBehaviorConfig":
        if experiment_type == ExperimentType.BASE:
            return cls(simulate_db_error=False, use_schema_typo=False)
        elif experiment_type == ExperimentType.EXCEPTION_HANDLING:
            return cls(simulate_db_error=False, use_schema_typo=True)
        elif experiment_type == ExperimentType.EXCEPTION_HANDLING_DB_ERROR:
            return cls(simulate_db_error=True, use_schema_typo=True)
