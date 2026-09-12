from pathlib import Path

import cv2
import numpy as np


TEST_BARCODE = "036000291452"
OUTPUT_PATH = Path(__file__).resolve().parent / "acetaminophen_036000291452.png"
TEST_CASES = {
    "01_green_known_medicine.png": "036000291452",
    "02_yellow_wrong_user.png": "135791357917",
    "03_red_unknown_barcode.png": "246802468024",
    "04_yellow_simulated_symptoms.png": "987654321098",
    "05_red_simulated_allergy.png": "314159265358",
}

LEFT_PATTERNS = {
    "0": "0001101",
    "1": "0011001",
    "2": "0010011",
    "3": "0111101",
    "4": "0100011",
    "5": "0110001",
    "6": "0101111",
    "7": "0111011",
    "8": "0110111",
    "9": "0001011",
}


def calculate_check_digit(first_eleven_digits):
    weighted_sum = sum(int(digit) for digit in first_eleven_digits[::2]) * 3
    weighted_sum += sum(int(digit) for digit in first_eleven_digits[1::2])
    return str((10 - weighted_sum % 10) % 10)


def encode_upc_a(value):
    if len(value) != 12 or not value.isdigit():
        raise ValueError("UPC-A values must contain exactly 12 digits.")
    if calculate_check_digit(value[:11]) != value[-1]:
        raise ValueError("UPC-A check digit is invalid.")

    left = "".join(LEFT_PATTERNS[digit] for digit in value[:6])
    right = "".join(
        "".join("1" if bit == "0" else "0" for bit in LEFT_PATTERNS[digit])
        for digit in value[6:]
    )
    return "101" + left + "01010" + right + "101"


def generate_barcode(value=TEST_BARCODE, output_path=OUTPUT_PATH):
    bits = encode_upc_a(value)
    module_width = 5
    quiet_zone_modules = 12
    bar_height = 260
    label_height = 55
    width = (len(bits) + quiet_zone_modules * 2) * module_width
    image = np.full((bar_height + label_height, width), 255, dtype=np.uint8)

    left_offset = quiet_zone_modules * module_width
    for index, bit in enumerate(bits):
        if bit == "1":
            left = left_offset + index * module_width
            cv2.rectangle(image, (left, 0), (left + module_width - 1, bar_height), 0, -1)

    cv2.putText(
        image,
        value,
        (left_offset + 45, bar_height + 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        0,
        2,
        cv2.LINE_AA,
    )
    # A small blur approximates a camera image and makes the synthetic bars
    # decodable by the same OpenCV detector used by the application.
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not write barcode image to {output_path}")
    return output_path


def generate_all_test_barcodes(output_dir=None):
    output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent
    return [
        generate_barcode(value, output_dir / filename)
        for filename, value in TEST_CASES.items()
    ]


if __name__ == "__main__":
    for generated_path in generate_all_test_barcodes():
        print(generated_path)
