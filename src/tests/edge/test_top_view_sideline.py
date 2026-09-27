# pyright: basic

import cv2
import numpy as np

from models.homography.geometry import (
    PITCH_VERTICES,
    _inferred_intersections,
    _line_models,
    _parallel_sideline,
    draw_homography,
    find_transformation,
)


def test_sideline_follows_penalty_edge_with_one_visible_corner():
    points = np.zeros_like(PITCH_VERTICES)
    confidence = np.zeros(len(points))
    points[[16, 17, 18, 21]] = [(580, 50), (500, 120), (300, 140), (580, 200)]
    confidence[[16, 17, 18, 21]] = 0.9
    segment = _parallel_sideline(points, confidence, 0, (450, 650, 3))
    assert segment is not None
    start, end = segment
    assert start == (580, 50)
    assert abs((end[1] - start[1]) / (end[0] - start[0]) + 0.1) < 0.01

    pitch_to_image = np.array([[0.04, 0, 100], [0, 0.05, 50], [0, 0, 1]])
    frame = np.zeros((450, 650, 3), dtype=np.uint8)
    draw_homography(frame, np.linalg.inv(pitch_to_image), points, confidence)
    assert frame[70, 380, 1] == 255  # lateral corrigida
    assert frame[50, 580, 1] == 255  # canto detectado
    assert frame[130, 400, 1] == 255  # borda da área passa nos dois pontos


def test_no_reference_uses_existing_field_template():
    points = np.zeros_like(PITCH_VERTICES)
    confidence = np.zeros(len(points))
    points[16] = (580, 50)
    confidence[16] = 0.9
    assert _parallel_sideline(points, confidence, 0, (450, 650, 3)) is None
    pitch_to_image = np.array([[0.04, 0, 100], [0, 0.05, 50], [0, 0, 1]])
    frame = np.zeros((450, 650, 3), dtype=np.uint8)
    draw_homography(frame, np.linalg.inv(pitch_to_image), points, confidence)
    assert frame[50, 380, 1] == 255


def test_two_visible_corners_use_original_projection():
    points = np.zeros_like(PITCH_VERTICES)
    confidence = np.zeros(len(points))
    points[[16, 0, 17, 18]] = [(580, 50), (100, 50), (500, 120), (300, 140)]
    confidence[[16, 0, 17, 18]] = 0.9
    assert _parallel_sideline(points, confidence, 0, (450, 650, 3)) is None


def test_bottom_sideline_uses_its_nearest_penalty_edge():
    points = np.zeros_like(PITCH_VERTICES)
    confidence = np.zeros(len(points))
    points[[25, 19, 20]] = [(590, 390), (510, 320), (310, 300)]
    confidence[[25, 19, 20]] = 0.9
    segment = _parallel_sideline(points, confidence, 7000, (450, 650, 3))
    assert segment is not None
    start, end = segment
    assert start == (590, 390)
    assert end[0] < start[0]
    assert end[1] < start[1]


def test_direct_line_connects_every_detected_point_even_with_bad_homography():
    points = np.zeros_like(PITCH_VERTICES)
    confidence = np.zeros(len(points))
    points[[16, 17, 21, 23]] = [(500, 30), (540, 100), (570, 150), (620, 240)]
    confidence[[16, 17, 21, 23]] = 0.9
    frame = np.zeros((300, 650, 3), dtype=np.uint8)
    draw_homography(frame, np.eye(3), points, confidence)
    for x, y in points[[16, 17, 21, 23]].astype(int):
        assert frame[y, x, 1] == 255
    assert frame[125, 555, 1] == 255


def test_detected_lines_draw_without_homography():
    points = np.zeros_like(PITCH_VERTICES)
    confidence = np.zeros(len(points))
    points[[18, 17]] = [(100, 70), (260, 50)]
    confidence[[18, 17]] = 0.9
    frame = np.zeros((150, 320, 3), dtype=np.uint8)
    draw_homography(frame, None, points, confidence)
    assert frame[60, 180, 1] == 255


def test_two_parallel_references_use_their_vanishing_point():
    points = np.zeros_like(PITCH_VERTICES)
    confidence = np.zeros(len(points))
    points[[16, 17, 18, 21, 22]] = [(1048, 211), (1266, 279), (714, 322), (1477, 342), (1286, 361)]
    confidence[[16, 17, 18, 21, 22]] = 0.9
    segment = _parallel_sideline(points, confidence, 0, (1080, 1920, 3))
    assert segment is not None
    start, end = segment
    slope = (end[1] - start[1]) / (end[0] - start[0])
    assert -0.06 < slope < -0.05


def test_missing_penalty_corner_is_parallel_intersection_with_goal_line():
    pitch_to_image = np.array([[0.04, 0.01, 100], [0, 0.05, 50], [0, 0, 1]])
    points = cv2.perspectiveTransform(PITCH_VERTICES.reshape(-1, 1, 2), pitch_to_image).reshape(
        -1, 2
    )
    confidence = np.zeros(len(points))
    confidence[[16, 17, 18, 20, 21, 22, 23]] = 0.9
    visible = confidence >= 0.25
    models = _line_models(points, visible)
    inferred = _inferred_intersections(models, visible, (600, 800, 3))
    np.testing.assert_allclose(inferred[19], points[19], atol=1)


def test_goal_line_extends_straight_when_final_corner_is_missing():
    true_projection = np.array([[0.02, 0.005, 50], [0, 0.04, 50], [0, 0, 1]])
    points = cv2.perspectiveTransform(PITCH_VERTICES.reshape(-1, 1, 2), true_projection).reshape(
        -1, 2
    )
    confidence = np.zeros(len(points))
    confidence[[16, 17, 21, 23]] = 0.9
    wrong_projection = true_projection.copy()
    wrong_projection[0, 2] += 40
    frame = np.zeros((450, 650, 3), dtype=np.uint8)
    draw_homography(frame, np.linalg.inv(wrong_projection), points, confidence)
    # Trecho além do último ponto observado segue a mesma reta de gol.
    assert frame[300, 321, 1] == 255


def test_missing_intermediate_points_complete_the_field_line():
    true_projection = np.array([[0.02, 0.005, 50], [0, 0.04, 50], [0, 0, 1]])
    points = cv2.perspectiveTransform(PITCH_VERTICES.reshape(-1, 1, 2), true_projection).reshape(
        -1, 2
    )
    confidence = np.zeros(len(points))
    confidence[[17, 23]] = 0.9

    # A homografia antiga desloca a linha; os pontos detectados devem ancorar
    # também os trechos intermediários que não foram vistos neste quadro.
    wrong_projection = true_projection.copy()
    wrong_projection[0, 2] += 40
    frame = np.zeros((450, 650, 3), dtype=np.uint8)
    draw_homography(frame, np.linalg.inv(wrong_projection), points, confidence)

    x, y = np.round(points[21]).astype(int)
    assert frame[y, x, 1] == 255


def test_homography_rejects_displaced_keypoint():
    pitch_to_image = np.array(
        [[0.08, 0.015, 120], [0.005, 0.07, 60], [0.00001, 0.00002, 1]],
        dtype=np.float64,
    )
    points = cv2.perspectiveTransform(PITCH_VERTICES.reshape(-1, 1, 2), pitch_to_image).reshape(
        -1, 2
    )
    confidence = np.zeros(len(points))
    detected = np.array([0, 1, 9, 11, 12, 16, 17, 25, 28])
    confidence[detected] = 0.9
    points[12] += (350, -250)

    matrix, inliers = find_transformation(points, confidence)

    assert matrix is not None
    assert inliers is not None
    assert not inliers[12]
    assert inliers[detected].sum() >= 8
    recovered = cv2.perspectiveTransform(points[[0, 16, 28]].reshape(-1, 1, 2), matrix).reshape(
        -1, 2
    )
    np.testing.assert_allclose(recovered, PITCH_VERTICES[[0, 16, 28]], atol=2)
