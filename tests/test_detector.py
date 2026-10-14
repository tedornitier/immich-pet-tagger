"""Tests for dedupe_boxes: cross-class and body-part duplicate animal boxes.

Box tuples are (conf, x1, y1, x2, y2), normalized, highest confidence first, as
returned by detect_animals(). The coordinates below come from real photos where
YOLO boxed one cat two or three times.
"""
import detector
from detector import dedupe_boxes


def test_single_box_is_kept():
    boxes = [(0.9, 0.1, 0.1, 0.5, 0.5)]
    assert dedupe_boxes(boxes) == boxes


def test_near_identical_boxes_from_two_classes_collapse_to_the_most_confident():
    # Same cat reported as "cat" 0.9 and as "dog" 0.3: per-class NMS keeps both.
    boxes = [(0.9, 0.0, 0.27, 0.535, 0.547), (0.3, 0.0, 0.269, 0.537, 0.55)]
    assert dedupe_boxes(boxes) == [boxes[0]]


def test_body_part_inside_the_whole_animal_is_dropped():
    # Whole cat (with tail) and a box around the body only, ~98% inside it. Their IoU
    # is ~0.55, under the default IOU_THRESHOLD, so NMS alone keeps both.
    whole = (0.9, 0.407, 0.46, 0.647, 0.639)
    part = (0.4, 0.463, 0.46, 0.65, 0.587)
    assert dedupe_boxes([whole, part]) == [whole]


def test_whole_animal_replaces_a_more_confident_part():
    part = (0.8, 0.46, 0.46, 0.65, 0.59)
    whole = (0.6, 0.40, 0.45, 0.65, 0.64)
    assert dedupe_boxes([part, whole]) == [whole]


def test_large_low_confidence_box_does_not_swallow_a_confident_detection():
    cat = (0.9, 0.4, 0.4, 0.6, 0.6)
    spurious = (0.2, 0.0, 0.0, 1.0, 1.0)
    assert dedupe_boxes([cat, spurious]) == [cat]


def test_two_overlapping_pets_are_both_kept():
    left = (0.9, 0.1, 0.3, 0.45, 0.8)
    right = (0.85, 0.4, 0.35, 0.8, 0.85)
    assert dedupe_boxes([left, right]) == [left, right]


def test_containment_rule_can_be_disabled(monkeypatch):
    monkeypatch.setattr(detector, "CONTAINMENT_THRESHOLD", 1.0)
    whole = (0.9, 0.407, 0.46, 0.647, 0.639)
    part = (0.4, 0.463, 0.46, 0.65, 0.587)
    assert dedupe_boxes([whole, part]) == [whole, part]


def test_result_is_sorted_by_confidence():
    a = (0.5, 0.0, 0.0, 0.2, 0.2)
    b = (0.9, 0.5, 0.5, 0.7, 0.7)
    assert dedupe_boxes([b, a]) == [b, a]
