import glob
import os

import pandas as pd


class ResultsAggregator:
    ADAPTATION_LABELS = {
        "base_rule": "Base rule",
        "0_values": "Special cases: area 0",
        "500_values": "Special cases: area 500",
        "900_values": "Special cases: area 900",
        "city_values": "Excluded cities",
        "extension_estimates": "Extension: mandatory reading ",
        "extension_mail": "Extension: direct mail",
        "Typing Error Handling": "Typing Error Handling",
        "Database Error Handling": "Database Error Handling",
    }

    def __init__(self, data_folder: str = "experiment_results"):
        self.data_folder = data_folder

    def load_and_merge(self) -> pd.DataFrame:
        frames = []

        pattern = os.path.join(self.data_folder, "process_adaptation_results_*.csv")
        files = sorted(glob.glob(pattern))

        for path in files:
            fname = os.path.basename(path)
            df = pd.read_csv(path)
            if "_exception_handling_db_error" in fname:
                df["process_adaptation"] = "Database Error Handling"
            elif "_exception_handling" in fname:
                df["process_adaptation"] = "Typing Error Handling"
            frames.append(df)

        if not frames:
            raise FileNotFoundError(f"No process_adaptation_results_*.csv files found in {self.data_folder}")

        df = pd.concat(frames, ignore_index=True).reset_index(drop=True)

        # Clean up columns
        df = df.drop(
            columns=["session_id", "seed", "correct_matches", "correct_columns", "file_extracted", "mails_sent"],
            errors="ignore",
        )

        df["tour"] = df["tour"].apply(lambda x: x.replace("J09", " ").title())
        df["rule_adaptation_method"] = df["rule_adaptation_method"].apply(
            lambda x: "BPMN model" if x == "generate_bpmn" else "Classic agent" if x == "classic" else "Human"
        )
        df["process_adaptation"] = df["process_adaptation"].apply(
            lambda x: self.ADAPTATION_LABELS.get(x, x)
        )

        df = df.rename(
            columns={
                "process_adaptation": "Process adaptation",
                "rule_adaptation_method": "Frame source",
                "tour": "Tour",
                "all_correct": "Successful process execution",
            }
        )

        return df

    def compute_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(["Frame source", "Process adaptation"], as_index=False)[
            "Successful process execution"
        ].mean()

    def generate_latex_table(self, df: pd.DataFrame) -> str:
        df_plot = df.groupby(["Frame source"], as_index=False)["Successful process execution"].sum()
        total_per_source = df.groupby("Frame source").size()
        for idx, row in df_plot.iterrows():
            source = row["Frame source"]
            df_plot.loc[idx, "Successful process execution"] = (
                round(row["Successful process execution"] / total_per_source[source], 4) * 100
            )
        return df_plot.to_latex(index=False, float_format="%.2f")

