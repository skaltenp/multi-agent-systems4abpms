import os
import shutil
import sqlite3

from config.server_config import ExperimentType


class SessionManager:
    CLEAN_DB_PATH = "database_clean.db"
    CLEAN_TYPO_DB_PATH = "database_clean_typo.db"

    def __init__(self, db_path: str = "database.db", sessions_folder: str = "sessions",
                 experiment_type: ExperimentType = ExperimentType.BASE):
        self.db_path = db_path
        self.sessions_folder = sessions_folder
        self.experiment_type = experiment_type

    def build_session_id(
        self,
        process_adaptation: str,
        rule_adaptation_method: str,
        tour: str,
        seed: int,
        experiment_type: ExperimentType,
    ) -> str:
        base = f"processadaptation_{process_adaptation}_processadaptationmethod_{rule_adaptation_method}_tour_{tour}_seed_{seed}"
        return base + experiment_type.session_suffix

    def session_exists(self, session_id: str) -> bool:
        return os.path.exists(os.path.join(self.sessions_folder, session_id))

    def write_seed(self, seed: int) -> None:
        with open("seed.txt", "w", encoding="utf-8") as f:
            f.write(str(seed))

    def write_session_id(self, session_id: str) -> None:
        with open("session_id.txt", "w", encoding="utf-8") as f:
            f.write(session_id)

    def write_sessions_folder(self) -> None:
        with open("sessions_folder.txt", "w", encoding="utf-8") as f:
            f.write(self.sessions_folder)

    def _clean_db_source(self) -> str:
        if self.experiment_type in (ExperimentType.EXCEPTION_HANDLING, ExperimentType.EXCEPTION_HANDLING_DB_ERROR):
            return self.CLEAN_TYPO_DB_PATH
        return self.CLEAN_DB_PATH

    def reset_database(self) -> None:
        source = self._clean_db_source()
        if os.path.exists(source):
            shutil.copy2(source, self.db_path)
        else:
            raise FileNotFoundError(
                f"Clean database '{source}' not found. Run: python run_experiment.py --setup-all-dbs"
            )

    def verify_rule_added(self, tour: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM process_rules WHERE process_rule_id = ?", (tour,))
            row = cursor.fetchone()
        return row is not None
