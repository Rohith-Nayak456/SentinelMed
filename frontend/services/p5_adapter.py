"""
P5 Queue Adapter
Loads and manages P5 scan review queue data, parses evaluation benchmarks,
and maintains the integration boundary for future P5 generator connection.

Strictly preserves P5 field names and values without fabricating data.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json


class P5LiveNotAvailableError(Exception):
    """Raised when live P5 queue generator execution is attempted before modules are available."""
    pass


class P5Adapter:
    """Adapter for P5 scan review queue integration."""

    DEFAULT_SAMPLE_PATH = Path(__file__).resolve().parent.parent.parent / "sample_p5_queue.json"
    DEFAULT_EVALUATION_PATH = Path(__file__).resolve().parent.parent.parent / "p5_evaluation.txt"

    @classmethod
    def is_live_available(cls) -> bool:
        """Checks whether the actual P5 implementation modules are importable."""
        try:
            import scan_queue.generator  # noqa: F401
            import scan_queue.schema  # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def load_sample_queue(
        cls, file_path: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Loads the sample queue from sample_p5_queue.json.
        Preserves original P5 values and record ordering.
        Reports malformed records or file issues without crashing.

        Returns:
            (records, errors)
        """
        target_path = Path(file_path) if file_path else cls.DEFAULT_SAMPLE_PATH
        errors: List[str] = []

        if not target_path.exists():
            return [], [f"Sample queue file not found at: {target_path}"]

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return [], [f"Malformed JSON in sample queue file: {e}"]
        except Exception as e:
            return [], [f"Error reading sample queue file: {e}"]

        if not isinstance(data, list):
            return [], ["Sample queue payload must be a JSON array of record objects."]

        # Validate each record and detect duplicates
        seen_ids = set()
        validated_records: List[Dict[str, Any]] = []

        for idx, rec in enumerate(data):
            if not isinstance(rec, dict):
                errors.append(f"Record at index {idx} is not a JSON object.")
                continue

            scan_id = rec.get("scan_id")
            if not scan_id:
                errors.append(f"Record at index {idx} is missing required 'scan_id'.")
            elif scan_id in seen_ids:
                errors.append(f"Duplicate scan_id '{scan_id}' detected at index {idx}.")
            else:
                seen_ids.add(scan_id)

            # Preserve exact record
            validated_records.append(rec)

        return validated_records, errors

    @classmethod
    def load_evaluation_data(cls, file_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Parses p5_evaluation.txt into structured benchmark data.
        Returns None if file is absent or malformed.
        """
        target_path = Path(file_path) if file_path else cls.DEFAULT_EVALUATION_PATH
        if not target_path.exists():
            return None

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                content = f.read()

            lines = [line.strip() for line in content.splitlines() if line.strip()]
            eval_data: Dict[str, Any] = {
                "raw_text": content,
                "dataset": "4 synthetic replay records",
                "review_budget": 2,
                "targets": "synthetic demonstration labels only",
                "strategies": {},
            }

            current_strat = None
            for line in lines:
                if line.startswith("Dataset:"):
                    eval_data["dataset"] = line.replace("Dataset:", "").strip()
                elif line.startswith("Review budget:"):
                    eval_data["review_budget"] = line.replace("Review budget:", "").strip()
                elif line.startswith("Targets:"):
                    eval_data["targets"] = line.replace("Targets:", "").strip()
                elif line.startswith("Strategy:"):
                    current_strat = line.replace("Strategy:", "").strip()
                    eval_data["strategies"][current_strat] = {}
                elif current_strat and ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip()
                    try:
                        val_num = float(val) if "." in val else int(val)
                        eval_data["strategies"][current_strat][key] = val_num
                    except ValueError:
                        eval_data["strategies"][current_strat][key] = val

            return eval_data
        except Exception:
            return None

    @classmethod
    def generate_live_queue(cls, replay_file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Calls live P5 queue generator when available.
        Raises P5LiveNotAvailableError if scan_queue modules are not yet implemented.
        """
        if not cls.is_live_available():
            raise P5LiveNotAvailableError(
                "Live P5 queue generator is not yet available in the repository.\n\n"
                "Missing P5 implementation modules:\n"
                "- scan_queue/generator.py\n"
                "- scan_queue/schema.py\n"
                "- scan_queue/priority.py\n"
                "- scan_queue/explanations.py\n"
                "- scan_queue/replay_data.json\n\n"
                "Please use 'Synthetic Demo (sample_p5_queue.json)' to inspect the verified queue contract."
            )

        # Teammate future integration point once scan_queue/ is pushed
        from scan_queue.generator import QueueGenerator  # type: ignore
        generator = QueueGenerator()
        if replay_file_path:
            return generator.generate_replay(replay_file_path)
        return generator.generate_live([])
