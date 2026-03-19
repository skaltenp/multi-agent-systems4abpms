import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def generate_master_csvs(data_folder: str = "data_prep") -> None:
    """Generate NET_MASTER.csv and SALES_MASTER.csv from DECISION_TABLE.csv."""
    data_path = Path(data_folder)
    np.random.seed(42)

    decision_df = pd.read_csv(data_path / "DECISION_TABLE.csv")

    titles = ["Herr", "Frau", "Dr.", "Prof.", "Dipl.-Ing."]
    forenames = ["Anna", "Hans", "Maria", "Peter", "Klaus", "Sabine", "Thomas", "Julia", "Michael", "Petra"]
    surnames = ["Müller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer", "Wagner", "Becker", "Schulz", "Hoffmann"]
    streets = ["Hauptstraße", "Bahnhofstraße", "Kirchweg", "Gartenstraße", "Schulstraße", "Bergstraße", "Waldweg", "Dorfstraße", "Lindenallee", "Marktplatz"]
    streetnos = ["1", "2", "5", "10", "15", "20", "25", "30", "42", "50", "1a", "3b", "12c"]

    num_rows = len(decision_df)
    decision_df["TITLE"] = np.random.choice(titles, size=num_rows)
    decision_df["FORENAME"] = np.random.choice(forenames, size=num_rows)
    decision_df["SURNAME"] = np.random.choice(surnames, size=num_rows)
    decision_df["STREET"] = np.random.choice(streets, size=num_rows)
    decision_df["STREETNO"] = np.random.choice(streetnos, size=num_rows)
    decision_df["EMAIL"] = (
        decision_df["FORENAME"].str.lower() + "." + decision_df["SURNAME"].str.lower() + "@example.com"
    )

    decision_df = decision_df[
        ["EQUNR", "TITLE", "FORENAME", "SURNAME", "STREET", "STREETNO", "CITY1", "CITY2", "POST_CODE1", "EMAIL"]
    ].copy()
    decision_df = decision_df.drop_duplicates(subset=["EQUNR"]).reset_index(drop=True)

    # NET_MASTER
    decision_df.to_csv(data_path / "NET_MASTER.csv", index=False)
    print(f"Created NET_MASTER.csv with {len(decision_df)} rows")

    # SALES_MASTER (90% of NET_MASTER)
    num_empty = int(len(decision_df) * 0.1)
    empty_indices = np.random.choice(decision_df.index, size=num_empty, replace=False)
    sales_df = decision_df.drop(index=empty_indices)
    sales_df.to_csv(data_path / "SALES_MASTER.csv", index=False)
    print(f"Created SALES_MASTER.csv with {len(sales_df)} rows")


def create_database(db_path: str = "database.db", data_folder: str = "data_prep", use_schema_typo: bool = False) -> None:
    """Load all CSVs from data_folder into SQLite database.

    Args:
        db_path: Path to the SQLite database file.
        data_folder: Path to the folder containing CSV files.
        use_schema_typo: If True, name the decision table 'decision_talbe'
            (intentional typo) to match the exception_handling experiment setup.
    """
    db_file = Path(db_path)
    if db_file.exists():
        db_file.unlink()

    data_path = Path(data_folder)
    csv_files = list(data_path.glob("*.csv"))
    # Filter out results CSVs
    csv_files = [f for f in csv_files if not f.name.startswith("process_adaptation_results")]  # safety filter (results now in experiment_results/)

    print(f"Found {len(csv_files)} CSV files in {data_folder}")

    with sqlite3.connect(db_path) as conn:
        for csv_file in csv_files:
            table_name = csv_file.stem.lower()
            print(f"  Loading {csv_file.name} -> table '{table_name}'...")
            df = pd.read_csv(csv_file).fillna("")
            if table_name.startswith("decision_table"):
                df["HOUSE_NUM1"] = df["HOUSE_NUM1"].apply(lambda x: "1111" if x == "" else x).astype(int)
                df = df.drop(columns=["ENTSCHEIDUNG", "REGEL"], errors="ignore").copy()
                if use_schema_typo:
                    table_name = "decision_talbe"
            df.to_sql(table_name, conn, if_exists="replace", index=False)

        # Create process_rules table
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS process_rules")
        cursor.execute("""
            CREATE TABLE process_rules (
                process_rule_id TEXT PRIMARY KEY,
                process_rule TEXT NOT NULL
            )
        """)

    print(f"Database created at {db_path}")


def create_all_clean_databases(data_folder: str = "data_prep") -> None:
    """Create both clean template databases: one without and one with the schema typo."""
    generate_master_csvs(data_folder)
    create_database(db_path="database_clean.db", data_folder=data_folder, use_schema_typo=False)
    create_database(db_path="database_clean_typo.db", data_folder=data_folder, use_schema_typo=True)
    print("Clean template databases created: database_clean.db, database_clean_typo.db")


if __name__ == "__main__":
    generate_master_csvs()
    create_database()
