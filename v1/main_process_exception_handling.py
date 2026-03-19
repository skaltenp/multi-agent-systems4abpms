# 1) Your standard FastAPI app
# Assumes the FastAPI app from above is already defined
from fastmcp import FastMCP
from fastapi import FastAPI, Request
import sqlite3
from typing import List, Dict, Any
import logging
import time
import csv
import os
from datetime import datetime
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = "database.db"

app = FastAPI(title="Enervie API")

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


@app.post("/run_sqlite_query", operation_id="run_sqlite_query")
async def run_sqlite_query(sqlite_command: str) -> Dict[str, Any]:
    """
    Execute SQLlite commands on the decision_table in the SQLite database.
    
    Args:
        sqlite_command: SQLite query to execute (e.g., "SELECT * FROM decision_table LIMIT 10")
    
    Returns:
        Dictionary with results (for SELECT) or affected rows count (for INSERT/UPDATE/DELETE)
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        #conn.row_factory = sqlite3.Row  # Enable column access by name
        cursor = conn.cursor()
        
        # Execute the SQL command
        cursor.execute(sqlite_command)
        
        # Check if it's a SELECT query
        if sqlite_command.strip().upper().startswith("SELECT"):
            # Fetch results
            rows = cursor.fetchall()
            # Convert to list of dictionaries
            results = [dict(row) for row in rows]
            conn.close()
            time.sleep(2) # prevent multi-query overload
            return {
                "success": True,
                "row_count": len(results),
                "data": results
            }
        else:
            # For INSERT, UPDATE, DELETE, etc.
            conn.commit()
            affected_rows = cursor.rowcount
            conn.close()
            time.sleep(2) # prevent multi-query overload
            return {
                "success": True,
                "affected_rows": affected_rows,
                "message": f"Query executed successfully. {affected_rows} row(s) affected."
            }
        
    
    except sqlite3.Error as e:
        return {
            "success": False,
            "error": str(e),
            "message": "SQL execution failed"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Unexpected error occurred"
        }

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
    try:
        # Determine output directory based on session_id
        session_id = get_session_id()
        if session_id:
            output_dir = os.path.join("sessions", session_id)
        else:
            output_dir = "temp"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"export_{timestamp}"
        
        # Ensure .csv extension
        if not filename.endswith('.csv'):
            filename = f"{filename}.csv"
        
        filepath = os.path.join(output_dir, filename)
        
        # Execute query
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql_query)
        
        # Get results and column names
        rows = cursor.fetchall()
        if not rows:
            conn.close()
            return {
                "success": False,
                "message": "Query returned no results"
            }
        
        column_names = rows[0].keys()
        
        # Write to CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=column_names)
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
        
        conn.close()
        
        return {
            "success": True,
            "filename": filename,
            "filepath": filepath,
            "csv_path": filepath,
            "row_count": len(rows),
            "message": f"Successfully exported {len(rows)} rows to {filepath}"
        }
        
    except sqlite3.Error as e:
        print(str(e))
        return {
            "success": False,
            "error": str(e),
            "message": "SQL execution failed"
        }
    except Exception as e:
        print(str(e))
        return {
            "success": False,
            "error": str(e),
            "message": "CSV export failed"
        }

@app.post("/send_mail", operation_id="send_mail")
async def send_mail(file_path: str, email: str, email_address: str) -> Dict[str, Any]:
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
        # Check if file exists
        if not os.path.exists(file_path):
            return {
                "success": False,
                "message": f"File not found: {file_path}"
            }
        
        # Get directory and filename
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        
        # Remove .csv extension and create email filename
        if filename.endswith('.csv'):
            base_name = filename[:-4]
        else:
            base_name = filename
        
        # Create CSV with email metadata
        email_csv_filename = f"process_metadata.csv"
        email_csv_path = os.path.join(directory, email_csv_filename)
        
        with open(email_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['email_address', 'file_path', 'email'])
            writer.writeheader()
            writer.writerow({
                'email_address': email_address,
                'file_path': file_path,
                'email': email
            })
        
        logger.info(f"Stored email metadata at {email_csv_path}")
        
        return {
            "success": True,
            "email_csv_path": email_csv_path,
            "message": f"Email metadata stored successfully at {email_csv_path}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to store email"
        }

@app.post("/send_bulk_mail", operation_id="send_bulk_mail")
async def send_bulk_mail(csv_file_path: str) -> str:
    """
    Send bulk emails from a CSV file path.
    
    Args:
        csv_file_path: Path to the CSV file.
    
    Returns:
        String with message about number of emails sent
    """
    # Check if CSV file exists
    if not os.path.exists(csv_file_path):
        raise ValueError(f"CSV file not found: {csv_file_path}")
    
    # Count lines in the file (excluding header)
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        line_count = sum(1 for _ in reader)
    
    logger.info(f"Mock: Sending {line_count} emails from {csv_file_path}")
    
    return f"sent {line_count} emails"

@app.get("/retrieve_process", operation_id="retrieve_process")
async def retrieve_process(process_rule_id: str) -> Dict[str, Any]:
    """
    Retrieve process information by process ID.
    
    Args:
        process_rule_id: The unique identifier of the process to retrieve
    
    Returns:
        Dictionary with process information including process_rule
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Query process_rules table
        cursor.execute(
            "SELECT process_rule_id, process_rule FROM process_rules WHERE process_rule_id = ?",
            (process_rule_id,)
        )
        result = cursor.fetchone()
        conn.close()
        logger.info(f"Retrieved process_rule for ID {process_rule_id}: {result}")
        if result:
            return {
                "success": True,
                "process_rule_id": result[0],
                "process_rule": result[1]
            }
        else:
            return {
                "success": False,
                "message": f"Process Rule ID '{process_rule_id}' not found"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to retrieve process_rule"
        }

# @app.post("/apply_base_rule", operation_id="apply_base_rule")
# async def apply_base_rule(table_name: str) -> Dict[str, Any]:
#     """
#     Apply base rule: Houses with 3+ electricity equipment ids get ENTSCHEIDUNG "EVU" and all others to "KSA"
#     Updates ENTSCHEIDUNG in the specified table.
    
#     Args:
#         table_name: Name of the table to apply the rule to
    
#     Returns:
#         Dictionary with success status and number of updated rows
#     """
#     try:
#         conn = sqlite3.connect(DB_PATH)
#         cursor = conn.cursor()
        
#         # First set all to KSA
#         query_ksa = f"""
#         UPDATE {table_name}
#         SET ENTSCHEIDUNG = 'KSA'
#         """
#         cursor.execute(query_ksa)
#         ksa_rows = cursor.rowcount
        
#         # Then set houses with 3+ equipment to EVU
#         query_evu = f"""
#         UPDATE {table_name}
#         SET ENTSCHEIDUNG = 'EVU'
#         WHERE HAUS IN (
#             SELECT HAUS
#             FROM {table_name}
#             GROUP BY HAUS
#             HAVING COUNT(EQUNR) >= 3
#         )
#         """
#         cursor.execute(query_evu)
#         evu_rows = cursor.rowcount
        
#         conn.commit()
#         conn.close()
        
#         return {
#             "success": True,
#             "ksa_rows": ksa_rows,
#             "evu_rows": evu_rows,
#             "message": f"Base rule applied: {ksa_rows} rows set to KSA, then {evu_rows} rows with 3+ electricity meters set to EVU"
#         }
        
#     except sqlite3.Error as e:
#         return {
#             "success": False,
#             "error": str(e),
#             "message": "Failed to apply base rule"
#         }
#     except Exception as e:
#         return {
#             "success": False,
#             "error": str(e),
#             "message": "Unexpected error occurred"
#         }

@app.get("/get_db_schema", operation_id="get_db_schema")
async def get_db_schema() -> Dict[str, Any]:
    """
    Get documentation of all SQL tables in the database.
    
    Returns:
        Dictionary with schema information as formatted string
    """
    schema = """# Table: net_master
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

# Table: decision_talbe
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
    
    return {
        "success": True,
        "schema": schema
    }

# 1. Generate MCP server from your API
mcp = FastMCP.from_fastapi(app=app, name="Enervie API")

# 2. Create the MCP's ASGI app
mcp_app = mcp.http_app(path='/mcp')

# 3. Create a new FastAPI app that combines both sets of routes
combined_app = FastAPI(
    title="Enervie API with MCP",
    routes=[
        *mcp_app.routes,  # MCP routes
        *app.routes,      # Original API routes
    ],
    lifespan=mcp_app.lifespan,
)

# Now you have:
# - Regular API: http://localhost:8000/products
# - LLM-friendly MCP: http://localhost:8000/mcp
# Both served from the same FastAPI application!

# 5) Run with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(combined_app, host="0.0.0.0", port=8000)