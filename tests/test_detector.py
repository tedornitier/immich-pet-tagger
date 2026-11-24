"""Tests for dedupe_boxes: cross-class and body-part duplicate animal boxes.

Box tuples are (conf, x1, y1, x2, y2), normalized, highest confidence first, as
returned by detect_animals(). The coordinates below come from real photos where
YOLO boxed one cat two or three times.
"""
import pytest

import detector
from detector import dedupe_boxes


@pytest.fixture
def containment(monkeypatch):
    """Turn on the opt-in containment rules (off by default, CONTAINMENT_THRESHOLD=1.0)."""
    monkeypatch.setattr(detector, "CONTAINMENT_THRESHOLD", 0.8)


@pytest.fixture
def no_containment(monkeypatch):
    """Pin the default, so the tests do not depend on the environment."""
    monkeypatch.setattr(detector, "CONTAINMENT_THRESHOLD", 1.0)


def test_single_box_is_kept():
    boxes = [(0.9, 0.1, 0.1, 0.5, 0.5)]
    assert dedupe_boxes(boxes) == boxes


def test_near_identical_boxes_from_two_classes_collapse_to_the_most_confident(no_containment):
    # Same cat reported as "cat" 0.9 and as "dog" 0.3: per-class NMS keeps both.
    # The IoU rule merges them even with the containment rules off.
    boxes = [(0.9, 0.0, 0.27, 0.535, 0.547), (0.3, 0.0, 0.269, 0.537, 0.55)]
    assert dedupe_boxes(boxes) == [boxes[0]]


def test_body_part_inside_the_whole_animal_is_dropped(containment):
    # Whole cat (with tail) and a box around the body only, ~98% inside it. Their IoU
    # is ~0.55, under the default IOU_THRESHOLD, so NMS alone keeps both.
    whole = (0.9, 0.407, 0.46, 0.647, 0.639)
    part = (0.4, 0.463, 0.46, 0.65, 0.587)
    assert dedupe_boxes([whole, part]) == [whole]


def test_whole_animal_replaces_a_more_confident_part(containment):
    part = (0.8, 0.46, 0.46, 0.65, 0.59)
    whole = (0.6, 0.40, 0.45, 0.65, 0.64)
    assert dedupe_boxes([part, whole]) == [whole]


def test_large_low_confidence_box_does_not_swallow_a_confident_detection(containment):
    cat = (0.9, 0.4, 0.4, 0.6, 0.6)
    spurious = (0.2, 0.0, 0.0, 1.0, 1.0)
    assert dedupe_boxes([cat, spurious]) == [cat]


def test_two_overlapping_pets_are_both_kept(containment):
    left = (0.9, 0.1, 0.3, 0.45, 0.8)
    right = (0.85, 0.4, 0.35, 0.8, 0.85)
    assert dedupe_boxes([left, right]) == [left, right]


def test_box_around_two_cuddling_pets_does_not_merge_them(containment):
    # Real photo: two cats lying together. YOLO returned one box per cat plus one
    # around both. The group box must go and both cats stay.
    cat_a = (0.3122, 0.1558, 0.2526, 0.4998, 0.5308)
    cat_b = (0.276, 0.1237, 0.3731, 0.9908, 0.998)
    group = (0.218, 0.031, 0.0726, 0.9978, 0.9837)
    assert dedupe_boxes([cat_a, cat_b, group]) == [cat_a, cat_b]


def test_group_box_is_dropped_even_when_more_confident(containment):
    cat_a = (0.4, 0.05, 0.3, 0.45, 0.9)
    cat_b = (0.35, 0.5, 0.3, 0.95, 0.9)
    group = (0.6, 0.0, 0.2, 1.0, 1.0)
    assert dedupe_boxes([group, cat_a, cat_b]) == [cat_a, cat_b]


def test_two_boxes_inside_each_other_keep_the_more_confident(containment, monkeypatch):
    monkeypatch.setattr(detector, "IOU_THRESHOLD", 1.0)  # force the containment path
    a = (0.7, 0.10, 0.10, 0.50, 0.50)
    b = (0.4, 0.11, 0.09, 0.51, 0.52)
    assert dedupe_boxes([a, b]) == [a]


def test_containment_rules_are_off_by_default(no_containment):
    # A small pet inside a larger pet's box (or a body part inside the whole animal)
    # is kept unless CONTAINMENT_THRESHOLD is lowered below 1.0.
    big_pet = (0.9, 0.407, 0.46, 0.647, 0.639)
    small_pet = (0.4, 0.463, 0.46, 0.65, 0.587)
    assert dedupe_boxes([big_pet, small_pet]) == [big_pet, small_pet]


def test_group_box_is_kept_when_containment_is_off(no_containment):
    cat_a = (0.4, 0.05, 0.3, 0.45, 0.9)
    cat_b = (0.35, 0.5, 0.3, 0.95, 0.9)
    group = (0.6, 0.0, 0.2, 1.0, 1.0)
    assert dedupe_boxes([group, cat_a, cat_b]) == [group, cat_a, cat_b]


def test_result_is_sorted_by_confidence():
    a = (0.5, 0.0, 0.0, 0.2, 0.2)
    b = (0.9, 0.5, 0.5, 0.7, 0.7)
    assert dedupe_boxes([b, a]) == [b, a]
