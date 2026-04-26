import cv2


TARGET_BOX_WIDTH_RATIO = 0.50
TARGET_BOX_HEIGHT_RATIO = 0.65
BOX_COLOR_OK = (0, 180, 0)
BOX_COLOR_OUTSIDE = (220, 0, 0)
BOX_THICKNESS = 2


_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def annotate_face_alignment(frame):
    height, width, _ = frame.shape
    target_box = _get_target_box(width, height)
    face_box = _detect_largest_face(frame)

    is_aligned = face_box is not None and _box_contains(target_box, face_box)
    color = BOX_COLOR_OK if is_aligned else BOX_COLOR_OUTSIDE

    _draw_box(frame, target_box, color)

    if face_box is not None:
        _draw_box(frame, face_box, color)

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
