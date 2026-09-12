import json
from datetime import datetime, timezone
from pathlib import Path


SCAN_LOG_PATH = Path(__file__).resolve().parent / "scan_events.jsonl"


class ScanEventLogger:
    def __init__(self, log_path=SCAN_LOG_PATH):
        self.log_path = Path(log_path)

    def log(self, user_name, barcode_result, outcome):
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "user_name": user_name,
            "barcode": barcode_result["value"],
            "barcode_type": barcode_result.get("type", ""),
            "outcome": outcome,
        }
        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(event) + "\n")
        return event
