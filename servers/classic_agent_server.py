"""
Classic agent server - identical to process_server but used for the classic agent baseline.
Usage: python -m servers.classic_agent_server --experiment-type base --port 8002
"""
import argparse
import csv
import logging
import os
import sqlite3
import time
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastmcp import FastMCP

from config.server_config import ExperimentType, ToolBehaviorConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DB_PATH = "database.db"


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


def _db_schema_text(use_typo: bool) -> str:
    table_name = "decision_talbe" if use_typo else "decision_table"
    return f"""# Table: net_master
## column_name (type) description
EQUNR (TEXT) Equipment Number
TITLE (TEXT) Title (Mr., Mrs., Dr., etc.)
FORENAME (TEXT) First name
SURNAME (TEXT) Last name/surname
STREET (TEXT) Street name
STREETNO (TEXT) Street number
CITY1 (TEXT) City name
CITY2 (TEXT) District/additional city information
POST_CODE1 (INTEGER) Postal code
EMAIL (TEXT) Email address

# Table: sales_master
## column_name (type) description
EQUNR (TEXT) Equipment Number
TITLE (TEXT) Title (Mr., Mrs., Dr., etc.)
FORENAME (TEXT) First name
SURNAME (TEXT) Last name/surname
STREET (TEXT) Street name
STREETNO (TEXT) Street number
CITY1 (TEXT) City name
CITY2 (TEXT) District/additional city information
POST_CODE1 (INTEGER) Postal code
EMAIL (TEXT) Email address

# Table: {table_name}
## column_name (type) description
HAUS (TEXT) House identifier
ANLAGE (TEXT) Installation/facility identifier
ABLEINH (TEXT) Reading unit identifier
TOUR (TEXT) Tour/route identifier
ME_MA_ID (TEXT) Meter/Measuring point ID
CITY1 (TEXT) City name
CITY2 (TEXT) District/additional city information
HOUSE_NUM1 (INTEGER) House/building number
POST_CODE1 (INTEGER) Postal code
EQUNR (TEXT) Equipment Number

# Table: eabl
## column_name (type) description
Meter Reading (MR) Doc. No. (TEXT) Meter Reading Document Number
EQUNR (TEXT) Equipment Number
Geplante Ableseart (INTEGER) Planned reading type
MR type (TEXT) Meter reading type code
MR TYPE TEXT (TEXT) Meter Reading type description
MR category (TEXT) Meter reading category
ISTABLART_TXT (TEXT) Actual reading type description (SAP)
LOGIKZW (TEXT) Logic indicator
KENNZIFF (TEXT) Identification number
Record created on (TEXT) Record creation date
Object changed on (TEXT) Object modification date

# Table: eablg
## column_name (type) description
Meter Reading (MR) Doc. No. (TEXT) Meter Reading Document Number
Installation (TEXT) Installation identifier
Meter Reading reason (INTEGER) Meter reading reason code
MR Reason - Text (TEXT) Meter reading reason description
Scheduled MR Date (TEXT) Scheduled meter reading date
Meter Reading unit (TEXT) Meter reading unit
Meter Reading (MR) Doc. No..1 (INTEGER) Secondary/duplicate Meter Reading Document Number

# Table: eanl
## column_name (type) description
Installation (TEXT) Installation identifier
Installation type (TEXT) Type of installation
Record created on (TEXT) Record creation date
Object changed on (TEXT) Object modification date
SPARTE_TEXT (TEXT) Division/sector description
AKLASSE (TEXT) Customer class code
AKLASSE_TEXT (TEXT) Customer class description
TARIFTYP (TEXT) Tariff type code
TARIFTYP_TEXT (TEXT) Tariff type description
BRANCHE (TEXT) Industry/sector code
BRANCHE_TEXT (TEXT) Industry/sector description
ABLEINH (TEXT) Reading unit identifier
ME_MA_ID (TEXT) Meter/Measuring point ID
HAUS (TEXT) House identifier

# Table: process_rules
## column_name (type) description
process_rule_id (TEXT) PRIMARY KEY - Process rule identifier
process_rule (TEXT) NOT NULL - Process rule text/description"""


DB_ERROR_RESPONSE = {
    "success": False,
    "error": "Error no database connection",
    "message": "Error no database connection",
}


def create_app(experiment_type: ExperimentType) -> FastAPI:
    tool_config = ToolBehaviorConfig.from_experiment_type(experiment_type)
    app = FastAPI(title="Enervie API")

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        logger.info(f"Incoming request: {request.method} {request.url.path}")
        logger.info(f"Client: {request.client.host if request.client else 'Unknown'}")
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Completed in {process_time:.3f}s with status {response.status_code}")
        return response

    @app.post("/run_sqlite_query", operation_id="run_sqlite_query")
    async def run_sqlite_query(sqlite_command: str) -> Dict[str, Any]:
        """
        Execute SQLlite commands on the decision_table in the SQLite database.

        Args:
            sqlite_command: SQLite query to execute (e.g., "SELECT * FROM decision_table LIMIT 10")

        Returns:
            Dictionary with results (for SELECT) or affected rows count (for INSERT/UPDATE/DELETE)
        """
        if tool_config.simulate_db_error:
            return DB_ERROR_RESPONSE

        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(sqlite_command)

                if sqlite_command.strip().upper().startswith("SELECT"):
                    rows = cursor.fetchall()
                    results = [dict(row) for row in rows]
                else:
                    affected_rows = cursor.rowcount

            time.sleep(2)

            if sqlite_command.strip().upper().startswith("SELECT"):
                if len(results) > 50 and "qwen3.5_35b" in get_sessions_folder():
                    return {
                        "success": False,
                        "row_count": len(results),
                        "error": f"Query returned {len(results)} rows. Result set too large to display.",
                    }
                return {"success": True, "row_count": len(results), "data": results}
            else:
                return {
                    "success": True,
                    "affected_rows": affected_rows,
                    "message": f"Query executed successfully. {affected_rows} row(s) affected.",
                }
        except sqlite3.Error as e:
            return {"success": False, "error": str(e), "message": "SQL execution failed"}
        except Exception as e:
            return {"success": False, "error": str(e), "message": "Unexpected error occurred"}

    @app.post("/prepare_csv", operation_id="prepare_csv")
    async def prepare_csv(sql_query: str, filename: str = None) -> Dict[str, Any]:
        """
        Execute SQL query and export results to CSV file in sessions/<session_id> folder.

        Args:
            sql_query: SQL query to execute
            filename: Optional filename for the CSV (without extension). If not provided, timestamp will be used.

        Returns:
            Dictionary with success status and filename
        """
        if tool_config.simulate_db_error:
            return DB_ERROR_RESPONSE

        try:
            session_id = get_session_id()
            output_dir = os.path.join(get_sessions_folder(), session_id) if session_id else "temp"
            os.makedirs(output_dir, exist_ok=True)

            if not filename:
                return {"success": False, "error": "No filename provided", "message": "A filename must be provided for the CSV export."}
            if not filename.endswith(".csv"):
                filename = f"{filename}.csv"

            filepath = os.path.join(output_dir, filename)

            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(sql_query)

                rows = cursor.fetchall()
                if not rows:
                    return {"success": False, "message": "Query returned no results"}

                column_names = rows[0].keys()

            with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=column_names)
                writer.writeheader()
                for row in rows:
                    writer.writerow(dict(row))
            return {
                "success": True,
                "filename": filename,
                "filepath": filepath,
                "csv_path": filepath,
                "row_count": len(rows),
                "message": f"Successfully exported {len(rows)} rows to {filepath}",
            }
        except sqlite3.Error as e:
            print(str(e))
            return {"success": False, "error": str(e), "message": "SQL execution failed"}
        except Exception as e:
            print(str(e))
            return {"success": False, "error": str(e), "message": "CSV export failed"}

    @app.post("/send_mail", operation_id="send_mail")
    async def send_mail(
        file_path: str = "" if tool_config.simulate_db_error else ...,
        email: str = "" if tool_config.simulate_db_error else ...,
        email_address: str = "" if tool_config.simulate_db_error else ...,
    ) -> Dict[str, Any]:
        """
        Send email with attachment from a file path.

        Args:
            file_path: Path to the CSV file to attach
            email: Email content to send
            email_address: Email address to send to

        Returns:
            Dictionary with success status
        """
        try:
            if not tool_config.simulate_db_error and file_path and not os.path.exists(file_path):
                return {"success": False, "message": f"File not found: {file_path}"}

            if tool_config.simulate_db_error:
                directory = os.path.join(get_sessions_folder(), get_session_id())
            else:
                directory = os.path.dirname(file_path)

            email_csv_path = os.path.join(directory, "process_metadata.csv")
            file_exists = os.path.exists(email_csv_path)

            with open(email_csv_path, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["email_address", "file_path", "email"])
                if not file_exists:
                    writer.writeheader()
                writer.writerow(
                    {"email_address": email_address, "file_path": file_path, "email": email}
                )

            logger.info(f"Stored email metadata at {email_csv_path}")
            return {
                "success": True,
                "email_csv_path": email_csv_path,
                "message": f"Email metadata stored successfully at {email_csv_path}",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "message": "Failed to store email"}

    @app.post("/send_bulk_mail", operation_id="send_bulk_mail")
    async def send_bulk_mail(csv_file_path: str) -> str:
        """
        Send bulk emails from a CSV file path.

        Args:
            csv_file_path: Path to the CSV file.

        Returns:
            String with message about number of emails sent
        """
        if not os.path.exists(csv_file_path):
            raise ValueError(f"CSV file not found: {csv_file_path}")

        with open(csv_file_path, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            line_count = sum(1 for _ in reader)

        logger.info(f"Mock: Sending {line_count} emails from {csv_file_path}")
        return f"sent {line_count} emails"

    @app.get("/get_db_schema", operation_id="get_db_schema")
    async def get_db_schema() -> Dict[str, Any]:
        """
        Get documentation of all SQL tables in the database.

        Returns:
            Dictionary with schema information as formatted string
        """
        if tool_config.simulate_db_error:
            return DB_ERROR_RESPONSE

        return {"success": True, "schema": _db_schema_text(tool_config.use_schema_typo)}

    return app


def main():
    parser = argparse.ArgumentParser(description="Enervie Process Server")
    parser.add_argument(
        "--experiment-type",
        type=str,
        default="base",
        choices=["base", "exception_handling", "exception_handling_db_error"],
    )
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    experiment_type = ExperimentType(args.experiment_type)
    app = create_app(experiment_type)

    mcp = FastMCP.from_fastapi(app=app, name="Enervie API")
    mcp_app = mcp.http_app(path="/mcp", stateless_http=True)

    combined_app = FastAPI(
        title="Enervie API with MCP",
        routes=[*mcp_app.routes, *app.routes],
        lifespan=mcp_app.lifespan,
    )

    import uvicorn

    uvicorn.run(combined_app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
