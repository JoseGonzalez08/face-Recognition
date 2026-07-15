import cv2


class BarcodeScanner:
    def __init__(self):
        self.detector = cv2.barcode_BarcodeDetector()

    def scan(self, bgr_frame):
        found, decoded_values, decoded_types, points = self.detector.detectAndDecodeWithType(
            bgr_frame
        )
        if not found:
            return []

        results = []
        point_sets = self._normalize_points(points)

        for index, (value, barcode_type) in enumerate(zip(decoded_values, decoded_types)):
            clean_value = value.strip()
            if not clean_value:
                continue

            results.append(
                {
                    "value": clean_value,
                    "type": barcode_type,
                    "points": point_sets[index] if index < len(point_sets) else [],
                }
            )

        return results

    def _normalize_points(self, points):
        if points is None:
            return []

        if len(points.shape) == 2:
            points = [points]

        normalized_points = []
        for point_group in points:
            normalized_points.append(
                [(int(point[0]), int(point[1])) for point in point_group]
            )

        return normalized_points
