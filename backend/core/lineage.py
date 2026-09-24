"""
Structured Data Lineage & Session State Tracking.
Provides operation IDs, input/output shape audits, parameter snapshots,
and dataset versioning.
"""

import time
import uuid
from typing import Dict, Any, List, Optional
import pandas as pd


class LineageRecord:
    """Detailed record of a single dataset transformation operation."""
    def __init__(
        self,
        operation_name: str,
        parameters: Dict[str, Any],
        input_shape: tuple,
        output_shape: tuple,
        description: str,
        columns_affected: Optional[List[str]] = None,
        dropped_null_count: int = 0
    ):
        self.operation_id = str(uuid.uuid4())[:8]
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.operation_name = operation_name
        self.parameters = parameters
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.description = description
        self.columns_affected = columns_affected or []
        self.dropped_null_count = dropped_null_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "timestamp": self.timestamp,
            "operation": self.operation_name,
            "description": self.description,
            "input_shape": f"{self.input_shape[0]}x{self.input_shape[1]}",
            "output_shape": f"{self.output_shape[0]}x{self.output_shape[1]}",
            "rows_diff": self.output_shape[0] - self.input_shape[0],
            "columns_affected": ", ".join(self.columns_affected) if self.columns_affected else "None",
            "dropped_null_count": self.dropped_null_count
        }


class DatasetSessionManager:
    """
    Manages dataset state with immutable original dataset, versioned current dataset,
    and structured lineage audit history.
    """
    def __init__(self):
        self._original_df: Optional[pd.DataFrame] = None
        self._current_df: Optional[pd.DataFrame] = None
        self._history: List[pd.DataFrame] = []
        self._lineage_records: List[LineageRecord] = []
        self.dataset_name: str = "Untitled Dataset"
        self.version: int = 0

    def load_dataset(self, df: pd.DataFrame, dataset_name: str = "Dataset") -> None:
        """Loads a new baseline dataset, resetting history and locking immutable copy."""
        self._original_df = df.copy(deep=True)
        self._current_df = df.copy(deep=True)
        self._history = [df.copy(deep=True)]
        self._lineage_records = []
        self.dataset_name = dataset_name
        self.version = 1
        self.record_lineage(
            operation_name="Dataset Ingestion",
            parameters={"dataset_name": dataset_name},
            input_shape=(0, 0),
            output_shape=df.shape,
            description=f"Ingested baseline dataset '{dataset_name}' with {len(df)} rows and {len(df.columns)} columns."
        )

    @property
    def original_df(self) -> Optional[pd.DataFrame]:
        """Immutable baseline dataset."""
        return self._original_df.copy(deep=True) if self._original_df is not None else None

    @property
    def current_df(self) -> Optional[pd.DataFrame]:
        """Active working dataset."""
        return self._current_df

    def update_current_df(
        self,
        new_df: pd.DataFrame,
        operation_name: str,
        parameters: Dict[str, Any],
        description: str,
        columns_affected: Optional[List[str]] = None,
        dropped_null_count: int = 0
    ) -> None:
        """Applies a transformation, advancing version and logging structured lineage."""
        prev_shape = self._current_df.shape if self._current_df is not None else (0, 0)
        self._history.append(new_df.copy(deep=True))
        if len(self._history) > 10:  # Bound undo history to conserve memory
            self._history.pop(0)

        self._current_df = new_df
        self.version += 1

        self.record_lineage(
            operation_name=operation_name,
            parameters=parameters,
            input_shape=prev_shape,
            output_shape=new_df.shape,
            description=description,
            columns_affected=columns_affected,
            dropped_null_count=dropped_null_count
        )

    def record_lineage(
        self,
        operation_name: str,
        parameters: Dict[str, Any],
        input_shape: tuple,
        output_shape: tuple,
        description: str,
        columns_affected: Optional[List[str]] = None,
        dropped_null_count: int = 0
    ) -> None:
        record = LineageRecord(
            operation_name=operation_name,
            parameters=parameters,
            input_shape=input_shape,
            output_shape=output_shape,
            description=description,
            columns_affected=columns_affected,
            dropped_null_count=dropped_null_count
        )
        self._lineage_records.append(record)

    def reset_to_original(self) -> None:
        """Reverts active working state to original baseline."""
        if self._original_df is not None:
            self._current_df = self._original_df.copy(deep=True)
            self._history = [self._original_df.copy(deep=True)]
            self.version += 1
            self.record_lineage(
                operation_name="Reset to Baseline",
                parameters={},
                input_shape=self._current_df.shape,
                output_shape=self._original_df.shape,
                description="Reverted all modifications back to immutable original dataset."
            )

    def get_lineage_dataframe(self) -> pd.DataFrame:
        """Exports lineage records as a pandas DataFrame for UI rendering."""
        if not self._lineage_records:
            return pd.DataFrame(columns=[
                "operation_id", "timestamp", "operation", "description",
                "input_shape", "output_shape", "rows_diff", "columns_affected", "dropped_null_count"
            ])
        return pd.DataFrame([r.to_dict() for r in self._lineage_records])

    def get_lineage_dicts(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._lineage_records]
