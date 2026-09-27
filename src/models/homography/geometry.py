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
HOMOGRAPHY_RANSAC_THRESHOLD = 150.0
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

# Cada sequência representa uma reta real do campo, em ordem ao longo da tinta.
FIELD_LINES = (
    (0, 11, 16),  # lateral superior
    (9, 12, 25),  # lateral inferior
    (0, 1, 5, 7, 3, 9),  # linha de gol esquerda
    (16, 17, 21, 23, 19, 25),  # linha de gol direita
    (11, 13, 15, 14, 12),  # meio-campo
    (1, 2),
    (2, 4),
    (4, 3),  # área esquerda
    (18, 17),
    (18, 20),
    (20, 19),  # área direita
    (5, 6),
    (6, 8),
    (8, 7),  # pequena área esquerda
    (22, 21),
    (22, 24),
    (24, 23),  # pequena área direita
)


def find_transformation(points, confidence):
    if points is None or len(points) != len(PITCH_VERTICES):
        return None, None

    valid = confidence >= KEYPOINT_CONFIDENCE
    source = points[valid]
    target = PITCH_VERTICES[valid]

    if len(source) < 4:
        return None, np.zeros(len(points), dtype=bool)

    matrix, source_inliers = cv2.findHomography(
        source,
        target,
        cv2.RANSAC,
        HOMOGRAPHY_RANSAC_THRESHOLD,
    )

    if matrix is None or source_inliers is None:
        return None, np.zeros(len(points), dtype=bool)

    inliers = np.zeros(len(points), dtype=bool)
    inliers[np.flatnonzero(valid)] = source_inliers.ravel().astype(bool)
    return matrix / matrix[2, 2], inliers


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


def _parallel_sideline(points, confidence, y, frame_shape):
    """Lateral pela quina e pelo ponto de fuga das bordas paralelas da área."""
    if points is None or confidence is None or len(points) != len(PITCH_VERTICES):
        return None

    if y == 0:
        sides = ((16, 0, ((17, 18), (21, 22))), (0, 16, ((1, 2), (5, 6))))
    else:
        sides = ((25, 9, ((19, 20), (23, 24))), (9, 25, ((3, 4), (7, 8))))

    def visible(index):
        return confidence[index] >= KEYPOINT_CONFIDENCE and np.isfinite(points[index]).all()

    for corner, opposite, references in sides:
        if not visible(corner) or visible(opposite):
            continue
        segments = []
        for outer, inner in references:
            if visible(outer) and visible(inner):
                direction = np.asarray(points[inner] - points[outer], dtype=np.float64)
                if np.linalg.norm(direction) >= 20:
                    segments.append((outer, inner, direction))
        if not segments:
            continue
        direction = segments[0][2]
        if len(segments) == 2:
            a, b = segments
            line_a = np.cross((*points[a[0]], 1), (*points[a[1]], 1))
            line_b = np.cross((*points[b[0]], 1), (*points[b[1]], 1))
            vanishing = np.cross(line_a, line_b)
            if abs(vanishing[2]) > 1e-8:
                candidate = vanishing[:2] / vanishing[2] - points[corner]
                if np.isfinite(candidate).all() and np.linalg.norm(candidate) > 20:
                    if np.dot(candidate, direction) < 0:
                        candidate = -candidate
                    direction = candidate
        length = np.linalg.norm(direction)
        start = np.asarray(points[corner], dtype=np.float64)
        end = start + direction / length * (2 * np.hypot(*frame_shape[:2]))
        valid, clipped_start, clipped_end = cv2.clipLine(
            (0, 0, frame_shape[1], frame_shape[0]),
            tuple(np.round(start).astype(int)),
            tuple(np.round(end).astype(int)),
        )
        if valid:
            return clipped_start, clipped_end
    return None


def _image_line(first, second):
    line = np.cross((*first, 1.0), (*second, 1.0))
    norm = np.linalg.norm(line[:2])
    return line / norm if norm > 1e-8 else None


def _line_models(keypoints, visible):
    """Retas observadas e retas com uma âncora guiadas pelas paralelas."""
    models = [None] * len(FIELD_LINES)
    for number, chain in enumerate(FIELD_LINES):
        detected = [index for index in chain if visible[index]]
        if len(detected) >= 2:
            models[number] = _image_line(keypoints[detected[0]], keypoints[detected[-1]])

    direct_models = models.copy()
    for number, chain in enumerate(FIELD_LINES):
        detected = [index for index in chain if visible[index]]
        if len(detected) != 1 or models[number] is not None:
            continue
        horizontal = PITCH_VERTICES[chain[0], 1] == PITCH_VERTICES[chain[-1], 1]
        offset = float(PITCH_VERTICES[chain[0], 1 if horizontal else 0])
        references = []
        for other_number, other in enumerate(FIELD_LINES):
            model = direct_models[other_number]
            if model is None or other_number == number:
                continue
            other_horizontal = PITCH_VERTICES[other[0], 1] == PITCH_VERTICES[other[-1], 1]
            if horizontal != other_horizontal:
                continue
            other_offset = float(PITCH_VERTICES[other[0], 1 if horizontal else 0])
            references.append((abs(other_offset - offset), model))
        if not references:
            continue
        references.sort(key=lambda item: item[0])
        direction = np.array([references[0][1][1], -references[0][1][0]])
        if len(references) > 1:
            vanishing = np.cross(references[0][1], references[1][1])
            if abs(vanishing[2]) > 1e-8:
                candidate = vanishing[:2] / vanishing[2] - keypoints[detected[0]]
                if np.isfinite(candidate).all() and np.linalg.norm(candidate) > 20:
                    direction = candidate
        anchor = np.asarray(keypoints[detected[0]], dtype=np.float64)
        models[number] = _image_line(anchor, anchor + direction)
    return models


def _inferred_intersections(models, visible, frame_shape):
    inferred = {}
    limit = 2 * np.hypot(*frame_shape[:2])
    for index in range(len(PITCH_VERTICES)):
        if visible[index]:
            continue
        crossing = []
        for number, chain in enumerate(FIELD_LINES):
            if index in chain and models[number] is not None:
                crossing.append(models[number])
        if len(crossing) != 2:
            continue
        homogeneous = np.cross(*crossing)
        if abs(homogeneous[2]) < 1e-5:
            continue
        image_point = homogeneous[:2] / homogeneous[2]
        if np.isfinite(image_point).all() and np.abs(image_point).max() < limit:
            inferred[index] = image_point
    return inferred


def _draw_field_template(frame, matrix, keypoints, confidence):
    visible = (confidence >= KEYPOINT_CONFIDENCE) & np.isfinite(keypoints).all(axis=1)
    if len(visible) != len(PITCH_VERTICES):
        return
    inverse = None
    projected = None
    if matrix is not None:
        inverse = np.linalg.inv(matrix)
        projected = cv2.perspectiveTransform(PITCH_VERTICES.reshape(-1, 1, 2), inverse)
        projected = projected.reshape(-1, 2)
    models = _line_models(keypoints, visible)
    inferred = _inferred_intersections(models, visible, frame.shape)

    for number, chain in enumerate(FIELD_LINES):
        resolved = []
        for index in chain:
            if visible[index]:
                point = keypoints[index]
            elif index in inferred:
                point = inferred[index]
            elif projected is not None:
                point = projected[index]
                model = models[number]
                if model is not None and np.isfinite(point).all():
                    point = (
                        point - (model[0] * point[0] + model[1] * point[1] + model[2]) * model[:2]
                    )
            else:
                continue
            if np.isfinite(point).all() and np.abs(point).max() < 100000:
                resolved.append(point)
        if len(resolved) >= 2:
            pixels = np.round(resolved).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pixels], False, (0, 255, 255), 2)
        elif number in (0, 1):
            parallel = _parallel_sideline(
                keypoints, confidence, PITCH_VERTICES[chain[0], 1], frame.shape
            )
            if parallel is not None:
                cv2.line(frame, *parallel, (0, 255, 255), 2)

    if inverse is None:
        return
    circle = np.float32(
        [
            (
                HALF_LENGTH + CENTER_CIRCLE_RADIUS * np.cos(angle),
                PITCH_WIDTH / 2 + CENTER_CIRCLE_RADIUS * np.sin(angle),
            )
            for angle in np.linspace(0, 2 * np.pi, 96)
        ]
    ).reshape(-1, 1, 2)
    curve = cv2.perspectiveTransform(circle, inverse)
    circle_indices = [index for index in (13, 14, 27, 28) if visible[index]]
    if len(circle_indices) >= 3:
        anchors = projected[circle_indices[:3]].astype(np.float32)
        correction = cv2.getAffineTransform(
            anchors, keypoints[circle_indices[:3]].astype(np.float32)
        )
        curve = cv2.transform(curve, correction)
    if np.isfinite(curve).all() and np.abs(curve).max() < 100000:
        cv2.polylines(frame, [np.round(curve).astype(np.int32)], True, (0, 255, 255), 2)


def draw_homography(frame, matrix, keypoints=None, confidence=None):
    if keypoints is not None and confidence is not None:
        _draw_field_template(frame, matrix, keypoints, confidence)
        return

    if matrix is None:
        return

    inverse = np.linalg.inv(matrix)
    lines = [
        [(0, 0), (0, PITCH_WIDTH)],
        [(PITCH_LENGTH, 0), (PITCH_LENGTH, PITCH_WIDTH)],
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

    for y in (0, PITCH_WIDTH):
        projected = cv2.perspectiveTransform(
            np.float32([[[0, y]], [[PITCH_LENGTH, y]]]), inverse
        ).reshape(-1, 2)
        if np.isfinite(projected).all() and np.abs(projected).max() < 100000:
            start, end = np.round(projected).astype(int)
            cv2.line(frame, tuple(start), tuple(end), (0, 255, 255), 2)


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
