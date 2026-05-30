from datetime import datetime
from pathlib import Path
import uuid

import pandas as pd

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger("silver.quarantine")


class QuarantineManager:

    def __init__(self, config: dict | None = None):

        if config is None:
            config = load_config()

        self.config = config

        self.base_dir = resolve_path(
            config["paths"]["silver"]["rejected_dir"]
        )

        self.manifest_path = resolve_path(
            config["paths"]["silver"]["rejection_manifest"]
        )

        self.metrics_path = resolve_path(
            config["paths"]["silver"]["rejection_metrics"]
        )

        self.rejections: list[pd.DataFrame] = []

        self._initialize_directories()


# INITIALIZATION

    def _initialize_directories(self):

        for severity in ["CRITICAL", "WARNING", "INFO"]:

            (
                self.base_dir / severity.lower()
            ).mkdir(
                parents=True,
                exist_ok=True
            )

    # ADD REJECTIONS

    def add_rejections(
        self,
        rejection_df: pd.DataFrame
    ) -> None:

        if rejection_df.empty:
            return

        rejection_df = rejection_df.copy()

        rejection_df["rejection_id"] = [
            self._generate_rejection_id()
            for _ in range(len(rejection_df))
        ]

        rejection_df["quarantine_timestamp"] = (
            datetime.now().isoformat()
        )

        self.rejections.append(rejection_df)

        logger.info(
            f"Collected {len(rejection_df):,} "
            f"rejected records"
        )

    
    # GENERATE REJECTION ID
    

    @staticmethod
    def _generate_rejection_id() -> str:

        return f"REJ_{uuid.uuid4().hex[:12]}"

    # GET ALL REJECTIONS

    def get_all_rejections(self) -> pd.DataFrame:

        if not self.rejections:
            return pd.DataFrame()

        combined = pd.concat(
            self.rejections,
            ignore_index=True
        )

        combined = combined.drop_duplicates()

        return combined

    # BUILD METRICS

    def build_metrics(
        self,
        rejection_df: pd.DataFrame
    ) -> pd.DataFrame:

        if rejection_df.empty:
            return pd.DataFrame()

        metrics = (
            rejection_df
            .groupby(
                ["dataset", "rule_id", "severity"]
            )
            .size()
            .reset_index(name="violation_count")
        )

        metrics["generated_at"] = (
            datetime.now().isoformat()
        )

        return metrics

    # FLUSH TO STORAGE

    def flush(self) -> None:

        all_rejections = self.get_all_rejections()

        if all_rejections.empty:

            logger.info(
                "No quarantined records to flush."
            )

            return

        # PARTITION BY SEVERITY + DATASET

        grouped = all_rejections.groupby(
            ["severity", "dataset"]
        )

        for (
            severity,
            dataset_name
        ), group_df in grouped:

            output_path = (
                self.base_dir
                / severity.lower()
                / f"{dataset_name}_rejected.parquet"
            )

            group_df.to_parquet(
                output_path,
                index=False
            )

            logger.info(
                f"Wrote {len(group_df):,} "
                f"{severity} records -> "
                f"{output_path.name}"
            )

        
        # WRITE MANIFEST
        
        all_rejections.to_csv(
            self.manifest_path,
            index=False
        )

        logger.info(
            f"Manifest updated -> "
            f"{self.manifest_path.name}"
        )

        
        # WRITE METRICS
       
        metrics_df = self.build_metrics(
            all_rejections
        )

        metrics_df.to_csv(
            self.metrics_path,
            index=False
        )

        logger.info(
            f"Metrics updated -> "
            f"{self.metrics_path.name}"
        )

    
    # SUMMARY
    
    @property
    def total_rejections(self) -> int:

        return sum(
            len(df)
            for df in self.rejections
        )

    @property
    def total_batches(self) -> int:

        return len(self.rejections)