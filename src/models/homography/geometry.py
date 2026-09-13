import cv2
import numpy as np

PITCH_LENGTH = 12000
PITCH_WIDTH = 7000
HALF_LENGTH = PITCH_LENGTH // 2
PENALTY_BOX_LENGTH = 2015
PENALTY_BOX_WIDTH = 4100
GOAL_BOX_LENGTH = 550
GOAL_BOX_WIDTH = 1832
CENTER_CIRCLE_RADIUS = 915
KEYPOINT_CONFIDENCE = 0.25
POSITION_SMOOTHING = 0.3
HALF_SWITCH_MARGIN = 1000

PITCH_POINTS = np.array(
    [
        (0, 0),
        (0, 1450),
        (0, 2584),
        (0, 4416),
        (0, 5550),
        (0, 7000),
        (550, 2584),
        (550, 4416),
        (1100, 3500),
        (2015, 1450),
        (2015, 2584),
        (2015, 4416),
        (2015, 5550),
        (6000, 0),
        (6000, 2585),
        (6000, 4415),
        (6000, 7000),
        (9985, 1450),
        (9985, 2584),
        (9985, 4416),
        (9985, 5550),
        (10900, 3500),
        (11450, 2584),
        (11450, 4416),
        (12000, 0),
        (12000, 1450),
        (12000, 2584),
        (12000, 4416),
        (12000, 5550),
        (12000, 7000),
        (5085, 3500),
        (6915, 3500),
    ],
    dtype=np.float32,
)

EXTRA_POINTS = np.array(
    [(2932, 3500), (6000, 3500), (9069, 3500)],
    dtype=np.float32,
)

KEYPOINT_TO_PITCH = np.array(
    [
        0,
        1,
        9,
        4,
        12,
        2,
        6,
        3,
        7,
        5,
        32,
        13,
        16,
        14,
        15,
        33,
        24,
        25,
        17,
        28,
        20,
        26,
        22,
        27,
        23,
        29,
        34,
        30,
        31,
    ]
)

PITCH_VERTICES = np.vstack([PITCH_POINTS, EXTRA_POINTS])[KEYPOINT_TO_PITCH]


def find_transformation(points, confidence):
    if points is None or len(points) != len(PITCH_VERTICES):
        return None, None

    valid = confidence >= KEYPOINT_CONFIDENCE
    source = points[valid]
    target = PITCH_VERTICES[valid]

    if len(source) < 4:
        return None, np.zeros(len(points), dtype=bool)

    matrix, _ = cv2.findHomography(source, target, 0)

    if matrix is None:
        return None, np.zeros(len(points), dtype=bool)

    return matrix / matrix[2, 2], valid


def project_players(matrix, boxes, ids):
    if matrix is None or not boxes:
        return []

    feet = np.array(
        [[(x1 + x2) / 2, y2] for x1, y1, x2, y2 in boxes],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    positions = cv2.perspectiveTransform(feet, matrix).reshape(-1, 2)

    return [(position, track_id) for position, track_id in zip(positions, ids, strict=False)]


def smooth_positions(positions, previous):
    smoothed = []

    for point, track_id in positions:
        if track_id in previous:
            point = previous[track_id] * (1 - POSITION_SMOOTHING) + point * POSITION_SMOOTHING

        previous[track_id] = point
        smoothed.append((point, track_id))

    return smoothed


def select_half(positions, previous):
    valid = [point[0] for point, _ in positions if 0 <= point[0] <= PITCH_LENGTH]

    if not valid:
        return previous

    middle = np.median(valid)

    if previous is None:
        return middle > HALF_LENGTH

    if previous and middle < HALF_LENGTH - HALF_SWITCH_MARGIN:
        return False

    if not previous and middle > HALF_LENGTH + HALF_SWITCH_MARGIN:
        return True

    return previous


def draw_keypoints(frame, points, confidence, inliers):
    if points is None:
        return

    for index, (point, score) in enumerate(zip(points, confidence, strict=False)):
        if score < KEYPOINT_CONFIDENCE:
            continue

        color = (0, 255, 0) if inliers is not None and inliers[index] else (0, 165, 255)
        position = tuple(np.round(point).astype(int))
        cv2.circle(frame, position, 6, color, -1)
        cv2.putText(
            frame,
            str(index),
            (position[0] + 7, position[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
        )


def draw_homography(frame, matrix):
    if matrix is None:
        return

    inverse = np.linalg.inv(matrix)
    lines = [
        [(0, 0), (PITCH_LENGTH, 0), (PITCH_LENGTH, PITCH_WIDTH), (0, PITCH_WIDTH), (0, 0)],
        [(HALF_LENGTH, 0), (HALF_LENGTH, PITCH_WIDTH)],
        [
            (0, (PITCH_WIDTH - PENALTY_BOX_WIDTH) / 2),
            (PENALTY_BOX_LENGTH, (PITCH_WIDTH - PENALTY_BOX_WIDTH) / 2),
            (PENALTY_BOX_LENGTH, (PITCH_WIDTH + PENALTY_BOX_WIDTH) / 2),
            (0, (PITCH_WIDTH + PENALTY_BOX_WIDTH) / 2),
        ],
        [
            (PITCH_LENGTH, (PITCH_WIDTH - PENALTY_BOX_WIDTH) / 2),
            (PITCH_LENGTH - PENALTY_BOX_LENGTH, (PITCH_WIDTH - PENALTY_BOX_WIDTH) / 2),
            (PITCH_LENGTH - PENALTY_BOX_LENGTH, (PITCH_WIDTH + PENALTY_BOX_WIDTH) / 2),
            (PITCH_LENGTH, (PITCH_WIDTH + PENALTY_BOX_WIDTH) / 2),
        ],
    ]

    circle = [
        (
            HALF_LENGTH + CENTER_CIRCLE_RADIUS * np.cos(angle),
            PITCH_WIDTH / 2 + CENTER_CIRCLE_RADIUS * np.sin(angle),
        )
        for angle in np.linspace(0, 2 * np.pi, 48)
    ]
    lines.append(circle)

    for line in lines:
        points = np.array(line, dtype=np.float32).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(points, inverse).reshape(-1, 2)

        if np.isfinite(projected).all() and np.abs(projected).max() < 100000:
            cv2.polylines(
                frame,
                [np.round(projected).astype(np.int32)],
                False,
                (0, 255, 255),
                2,
            )


def view_size(height):
    return round(height * HALF_LENGTH / PITCH_WIDTH), height


def field_point(point, width, height, margin):
    x, y = point
    field_width = width - 2 * margin
    field_height = height - 2 * margin
    return (
        round(margin + x / HALF_LENGTH * field_width),
        round(margin + y / PITCH_WIDTH * field_height),
    )


def draw_top_view(height, positions, color_for_id, mirror=False):
    width, height = view_size(height)
    canvas = np.full((height, width, 3), (47, 125, 55), dtype=np.uint8)
    margin = max(20, round(height * 0.03))

    cv2.rectangle(
        canvas,
        (margin, margin),
        (width - margin, height - margin),
        (255, 255, 255),
        2,
    )
    middle_top = field_point((HALF_LENGTH, 0), width, height, margin)
    middle_bottom = field_point((HALF_LENGTH, PITCH_WIDTH), width, height, margin)
    cv2.line(canvas, middle_top, middle_bottom, (255, 255, 255), 2)

    penalty_top = field_point((0, (PITCH_WIDTH - PENALTY_BOX_WIDTH) / 2), width, height, margin)
    penalty_bottom = field_point(
        (PENALTY_BOX_LENGTH, (PITCH_WIDTH + PENALTY_BOX_WIDTH) / 2),
        width,
        height,
        margin,
    )
    cv2.rectangle(canvas, penalty_top, penalty_bottom, (255, 255, 255), 2)

    goal_top = field_point((0, (PITCH_WIDTH - GOAL_BOX_WIDTH) / 2), width, height, margin)
    goal_bottom = field_point(
        (GOAL_BOX_LENGTH, (PITCH_WIDTH + GOAL_BOX_WIDTH) / 2),
        width,
        height,
        margin,
    )
    cv2.rectangle(canvas, goal_top, goal_bottom, (255, 255, 255), 2)

    center = field_point((HALF_LENGTH, PITCH_WIDTH / 2), width, height, margin)
    radius = round(CENTER_CIRCLE_RADIUS / PITCH_WIDTH * (height - 2 * margin))
    cv2.circle(canvas, center, radius, (255, 255, 255), 2)

    if positions:
        for point, track_id in positions:
            x, y = point
            if mirror:
                x = PITCH_LENGTH - x

            if 0 <= x <= HALF_LENGTH and 0 <= y <= PITCH_WIDTH:
                position = field_point((x, y), width, height, margin)
                cv2.circle(canvas, position, 8, color_for_id(track_id), -1)
                cv2.circle(canvas, position, 8, (255, 255, 255), 1)
    else:
        cv2.putText(
            canvas,
            "Homografia indisponivel",
            (margin + 10, margin + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

    return canvas
