"""Telemetry Data Processing and Statistical Analytics Module.

This module processes interaction telemetry from `data/session_logs.csv` and post-session
survey responses from `data/affective_surveys.csv` to calculate learning gains, efficiency metrics,
action selection distributions, and affective scores for comparative A/B evaluation.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class SessionAnalyzer:
    """Telemetry data analyzer for POMDP-BKT Intelligent Tutoring System A/B testing.

    Attributes:
        session_log_path (str): Path to interaction telemetry CSV log file.
        survey_log_path (str): Path to post-session affective survey CSV log file.
    """

    def __init__(
        self,
        session_log_path: str = "data/session_logs.csv",
        survey_log_path: str = "data/affective_surveys.csv",
    ) -> None:
        """Initializes SessionAnalyzer with target CSV log paths.

        Args:
            session_log_path: Path to session interaction logs.
            survey_log_path: Path to post-session affective surveys.
        """
        self.session_log_path = session_log_path
        self.survey_log_path = survey_log_path

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads interaction logs and survey data into Pandas DataFrames.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (session_df, survey_df).
        """
        session_df = pd.DataFrame()
        survey_df = pd.DataFrame()

        if os.path.exists(self.session_log_path):
            try:
                session_df = pd.read_csv(self.session_log_path)
            except Exception:
                session_df = pd.DataFrame()

        if os.path.exists(self.survey_log_path):
            try:
                survey_df = pd.read_csv(self.survey_log_path)
            except Exception:
                survey_df = pd.DataFrame()

        return session_df, survey_df

    def normalize_mode_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalizes tutor mode names into 'RL Mode' and 'Control Mode'.

        Args:
            df: DataFrame containing a 'tutor_mode' column.

        Returns:
            pd.DataFrame: Processed DataFrame with standardized 'mode_label' column.
        """
        if df.empty or "tutor_mode" not in df.columns:
            return df

        df = df.copy()

        def _map_mode(val: str) -> str:
            val_str = str(val)
            if "Experimental" in val_str or "RL" in val_str:
                return "RL Mode"
            elif "Control" in val_str:
                return "Control Mode"
            return "Other Mode"

        df["mode_label"] = df["tutor_mode"].apply(_map_mode)
        return df

    def get_learning_gains(self) -> pd.DataFrame:
        """Calculates Mean Normalized Learning Gain (NLG) per session grouped by mode.

        NLG = (b_final - b_initial) / (1.0 - b_initial)

        Returns:
            pd.DataFrame: Summary table with columns ['Mode', 'Mean_NLG', 'Std_NLG', 'Session_Count'].
        """
        session_df, _ = self.load_data()
        if session_df.empty or "student_id" not in session_df.columns:
            return pd.DataFrame(columns=["Mode", "Mean_NLG", "Std_NLG", "Session_Count"])

        session_df = self.normalize_mode_names(session_df)

        gains: List[Dict[str, Any]] = []

        # Group by student_id and mode_label
        grouped = session_df.groupby(["student_id", "mode_label", "tutor_mode"])

        for (stu_id, mode_label, orig_mode), group in grouped:
            group_sorted = group.sort_values("step")
            if group_sorted.empty:
                continue

            init_b = float(group_sorted.iloc[0]["mean_belief"])
            final_b = float(group_sorted.iloc[-1]["mean_belief"])

            denom = max(1.0 - init_b, 1e-4)
            nlg = (final_b - init_b) / denom

            gains.append({
                "student_id": stu_id,
                "Mode": mode_label,
                "nlg": max(0.0, min(1.0, float(nlg))),
                "initial_belief": init_b,
                "final_belief": final_b,
            })

        gain_df = pd.DataFrame(gains)
        if gain_df.empty:
            return pd.DataFrame(columns=["Mode", "Mean_NLG", "Std_NLG", "Session_Count"])

        summary = (
            gain_df.groupby("Mode")["nlg"]
            .agg(Mean_NLG="mean", Std_NLG="std", Session_Count="count")
            .reset_index()
        )
        summary["Std_NLG"] = summary["Std_NLG"].fillna(0.0)

        return summary

    def get_steps_to_mastery(self) -> pd.DataFrame:
        """Calculates Average Steps to Mastery (Time-on-Task) grouped by mode.

        Returns:
            pd.DataFrame: Per-session step count breakdown with columns ['student_id', 'Mode', 'steps'].
        """
        session_df, _ = self.load_data()
        if session_df.empty or "student_id" not in session_df.columns:
            return pd.DataFrame(columns=["student_id", "Mode", "steps"])

        session_df = self.normalize_mode_names(session_df)
        records: List[Dict[str, Any]] = []

        grouped = session_df.groupby(["student_id", "mode_label", "tutor_mode"])
        for (stu_id, mode_label, orig_mode), group in grouped:
            max_step = int(group["step"].max())
            records.append({
                "student_id": stu_id,
                "Mode": mode_label,
                "steps": max_step,
            })

        return pd.DataFrame(records)

    def get_affective_averages(self) -> pd.DataFrame:
        """Calculates average post-session affective survey scores grouped by mode.

        Returns:
            pd.DataFrame: Summary table with columns ['Mode', 'Mean_Engagement', 'Mean_Frustration', 'Survey_Count'].
        """
        _, survey_df = self.load_data()
        if survey_df.empty or "tutor_mode" not in survey_df.columns:
            return pd.DataFrame(columns=["Mode", "Mean_Engagement", "Mean_Frustration", "Survey_Count"])

        survey_df = self.normalize_mode_names(survey_df)

        summary = (
            survey_df.groupby("mode_label")
            .agg(
                Mean_Engagement=("engagement_score", "mean"),
                Mean_Frustration=("frustration_score", "mean"),
                Survey_Count=("engagement_score", "count"),
            )
            .reset_index()
            .rename(columns={"mode_label": "Mode"})
        )

        return summary

    def get_action_distribution(self) -> pd.DataFrame:
        """Calculates action selection counts and percentages for RL Mode vs Control Mode.

        Returns:
            pd.DataFrame: Breakdown table with columns ['Mode', 'action_name', 'count', 'percentage'].
        """
        session_df, _ = self.load_data()
        if session_df.empty or "action_name" not in session_df.columns:
            return pd.DataFrame(columns=["Mode", "action_name", "count", "percentage"])

        session_df = self.normalize_mode_names(session_df)

        counts = (
            session_df.groupby(["mode_label", "action_name"])
            .size()
            .reset_index(name="count")
            .rename(columns={"mode_label": "Mode"})
        )

        total_per_mode = counts.groupby("Mode")["count"].transform("sum")
        counts["percentage"] = (counts["count"] / total_per_mode) * 100.0

        return counts
