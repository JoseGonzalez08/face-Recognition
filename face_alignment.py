import cv2


TARGET_BOX_WIDTH_RATIO = 0.58
TARGET_BOX_HEIGHT_RATIO = 0.72
BOX_COLOR_OK = (0, 180, 0)
BOX_COLOR_OUTSIDE = (220, 0, 0)
BOX_COLOR_READY = (0, 0, 255)
BOX_THICKNESS = 2


_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def annotate_face_alignment(frame, user_recognized=False, barcode_points_list=None):
    height, width, _ = frame.shape
    target_box = _get_target_box(width, height)
    face_box = _detect_largest_face(frame)

    is_aligned = face_box is not None and _box_contains(target_box, face_box)
    if user_recognized:
        color = BOX_COLOR_READY
    else:
        color = BOX_COLOR_OK if is_aligned else BOX_COLOR_OUTSIDE

    _draw_box(frame, target_box, color)

    if face_box is not None:
        _draw_box(frame, face_box, color)

    if barcode_points_list:
        for barcode_points in barcode_points_list:
            barcode_color = BOX_COLOR_OUTSIDE
            if user_recognized and _polygon_inside_box(target_box, barcode_points):
                barcode_color = BOX_COLOR_READY
            _draw_polygon(frame, barcode_points, barcode_color)

    return frame


def _get_target_box(frame_width, frame_height):
    box_width = int(frame_width * TARGET_BOX_WIDTH_RATIO)
    box_height = int(frame_height * TARGET_BOX_HEIGHT_RATIO)
    left = (frame_width - box_width) // 2
    top = (frame_height - box_height) // 2
    return left, top, box_width, box_height


def _detect_largest_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )

    if len(faces) == 0:
        return None

    return max(faces, key=lambda face: face[2] * face[3])


def _box_contains(outer_box, inner_box):
    outer_left, outer_top, outer_width, outer_height = outer_box
    inner_left, inner_top, inner_width, inner_height = inner_box

    outer_right = outer_left + outer_width
    outer_bottom = outer_top + outer_height
    inner_right = inner_left + inner_width
    inner_bottom = inner_top + inner_height

    return (
        inner_left >= outer_left
        and inner_top >= outer_top
        and inner_right <= outer_right
        and inner_bottom <= outer_bottom
    )


def _draw_box(frame, box, color):
    left, top, width, height = box
    right = left + width
    bottom = top + height
    cv2.rectangle(frame, (left, top), (right, bottom), color, BOX_THICKNESS)


def _polygon_inside_box(box, points):
    left, top, width, height = box
    right = left + width
    bottom = top + height

    for point_x, point_y in points:
        if point_x < left or point_x > right or point_y < top or point_y > bottom:
            return False

    return True


def _draw_polygon(frame, points, color):
    polygon_points = [(int(point_x), int(point_y)) for point_x, point_y in points]
    for index, start_point in enumerate(polygon_points):
        end_point = polygon_points[(index + 1) % len(polygon_points)]
        cv2.line(frame, start_point, end_point, color, BOX_THICKNESS)
