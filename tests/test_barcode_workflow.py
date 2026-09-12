import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2

from barcode_scanner import BarcodeScanner
from face_alignment import get_target_box, polygon_inside_box
from medicine_registry import MedicineRegistry
from scan_logger import ScanEventLogger
from test_barcodes.generate_test_barcode import (
    TEST_CASES,
    calculate_check_digit,
    encode_upc_a,
    generate_barcode,
)


class ScanAreaTests(unittest.TestCase):
    def test_barcode_fully_inside_target_is_accepted(self):
        target = get_target_box(1000, 800)
        left, top, width, height = target
        points = [
            (left, top),
            (left + width, top),
            (left + width, top + height),
            (left, top + height),
        ]
        self.assertTrue(polygon_inside_box(target, points))

    def test_barcode_crossing_target_edge_is_rejected(self):
        target = get_target_box(1000, 800)
        left, top, _, _ = target
        points = [(left - 1, top), (left + 20, top), (left + 20, top + 20)]
        self.assertFalse(polygon_inside_box(target, points))


class MedicineRegistryTests(unittest.TestCase):
    def test_known_unknown_and_wrong_user_outcomes(self):
        registry = MedicineRegistry()
        green_record = registry.find_for_user("036000291452", "Jose")
        yellow_record = registry.find_for_user("987654321098", "Jose")
        red_record = registry.find_for_user("314159265358", "Jose")
        self.assertEqual(green_record.demo_status, "green")
        self.assertEqual(yellow_record.demo_status, "yellow")
        self.assertEqual(red_record.demo_status, "red")
        self.assertIsNone(registry.find_for_user("036000291452", "Another User"))
        self.assertEqual(registry.find_owner_for_barcode("036000291452"), "Jose")
        self.assertEqual(registry.find_owner_for_barcode("135791357917"), "Alex")
        self.assertIsNone(registry.find_owner_for_barcode("246802468024"))


class ScanLoggerTests(unittest.TestCase):
    def test_scan_event_contains_expected_fields(self):
        with TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "events.jsonl"
            logger = ScanEventLogger(log_path)
            logger.log(
                "Jose",
                {"value": "036000291452", "type": "UPC_A"},
                "matched",
            )
            event = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual(event["user_name"], "Jose")
            self.assertEqual(event["barcode"], "036000291452")
            self.assertEqual(event["barcode_type"], "UPC_A")
            self.assertEqual(event["outcome"], "matched")
            self.assertIn("timestamp_utc", event)


class TestBarcodeGeneratorTests(unittest.TestCase):
    def test_sample_upc_is_valid_and_has_expected_module_count(self):
        self.assertEqual(calculate_check_digit("03600029145"), "2")
        self.assertEqual(len(encode_upc_a("036000291452")), 95)

    def test_generated_barcode_decodes_with_application_scanner(self):
        with TemporaryDirectory() as temp_dir:
            for filename, barcode_value in TEST_CASES.items():
                output_path = Path(temp_dir) / filename
                generate_barcode(barcode_value, output_path)
                image = cv2.imread(str(output_path))
                results = BarcodeScanner().scan(image)
                self.assertEqual(results[0]["value"], barcode_value)
                self.assertEqual(results[0]["type"], "UPC_A")


if __name__ == "__main__":
    unittest.main()
