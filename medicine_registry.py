from dataclasses import dataclass
import json
from pathlib import Path


REGISTRY_PATH = Path(__file__).resolve().parent / "medicine_records.json"


@dataclass
class MedicineRecord:
    barcode: str
    user_name: str
    medicine_name: str
    description: str
    uses: str
    directions: str


class MedicineRegistry:
    def __init__(self, registry_path=REGISTRY_PATH):
        self.registry_path = Path(registry_path)
        self.records = []
        self._last_loaded_mtime_ns = None
        self.reload()

    def reload(self):
        if not self.registry_path.exists():
            self.records = []
            self._last_loaded_mtime_ns = None
            return

        current_mtime_ns = self.registry_path.stat().st_mtime_ns
        if current_mtime_ns == self._last_loaded_mtime_ns:
            return

        data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        self.records = [
            MedicineRecord(
                barcode=str(record["barcode"]).strip(),
                user_name=record["user_name"].strip(),
                medicine_name=record["medicine_name"].strip(),
                description=record["description"].strip(),
                uses=record["uses"].strip(),
                directions=record["directions"].strip(),
            )
            for record in data.get("records", [])
            if str(record.get("barcode", "")).strip()
        ]
        self._last_loaded_mtime_ns = current_mtime_ns

    def find_for_user(self, barcode, user_name):
        normalized_barcode = barcode.strip()
        normalized_user_name = self._normalize_name(user_name)

        for record in self.records:
            if (
                record.barcode == normalized_barcode
                and self._normalize_name(record.user_name) == normalized_user_name
            ):
                return record

        return None

    def find_owner_for_barcode(self, barcode):
        normalized_barcode = barcode.strip()

        for record in self.records:
            if record.barcode == normalized_barcode:
                return record.user_name

        return None

    def format_record_summary(self, record):
        return (
            f"Medicine: {record.medicine_name}\n"
            f"For: {record.user_name}\n"
            f"Description: {record.description}\n"
            f"Uses: {record.uses}\n"
            f"Directions: {record.directions}"
        )

    def _normalize_name(self, user_name):
        return " ".join(user_name.lower().split())
