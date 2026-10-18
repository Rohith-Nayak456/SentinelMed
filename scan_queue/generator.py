import json
import logging
from pathlib import Path
from typing import Any, Iterable

from scan_queue.schema import validate_record
from scan_queue.priority import rank_records
from scan_queue.explanations import generate_explanation


logger = logging.getLogger(__name__)


class QueueGenerator:
    """Generate a review queue from replay or upstream pipeline records."""

    def __init__(self) -> None:
        self.errors: list[dict[str, str]] = []

    def _prepare_records(
        self,
        records: Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        valid_records = []
        seen_ids = set()

        self.errors = []

        for index, record in enumerate(records):
            try:
                validate_record(record)

                scan_id = record["scan_id"]

                if scan_id in seen_ids:
                    raise ValueError(
                        f"Duplicate scan identifier: {scan_id}"
                    )

                seen_ids.add(scan_id)

                prepared = dict(record)
                prepared["explanation"] = generate_explanation(prepared)

                valid_records.append(prepared)

            except (ValueError, TypeError, KeyError) as exc:
                error = {
                    "record_index": str(index),
                    "error": str(exc),
                }

                self.errors.append(error)
                logger.error("Rejected record: %s", error)

        return rank_records(valid_records)

    def generate_replay(
        self,
        replay_file: str | Path,
    ) -> list[dict[str, Any]]:
        """Read a JSON replay dataset and generate a test queue."""

        path = Path(replay_file)

        with path.open("r", encoding="utf-8") as file:
            records = json.load(file)

        if not isinstance(records, list):
            raise ValueError(
                "Replay dataset must contain a JSON list of records."
            )

        return self._prepare_records(records)

    def generate_live(
        self,
        upstream_records: Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Generate a queue from actual upstream records.

        The caller must supply records returned by the integrated
        upstream pipeline. This method does not generate predictions
        or recovery outcomes.
        """

        if upstream_records is None:
            raise ValueError(
                "Live mode requires actual upstream pipeline records."
            )

        return self._prepare_records(upstream_records)