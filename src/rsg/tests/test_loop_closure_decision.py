"""Unit tests for the pure loop-closure decision helpers (no ROS).

These lock down the "turning the feature on changes nothing until a real
correction arrives" contract: ``loop_closure_delta`` returns ``None`` for an
identity / sub-threshold ``map -> odom`` reading, so phase 1 never calls
``reanchor_all``.
"""

from __future__ import annotations

import math

import numpy as np

from nodes.support.phase1.loop_closure import loop_closure_delta, quat_to_rot


def _rz(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_quat_to_rot_identity_and_z90():
    assert np.allclose(quat_to_rot(0.0, 0.0, 0.0, 1.0), np.eye(3))
    q = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    assert np.allclose(quat_to_rot(*q) @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


def test_delta_identity_is_none():
    d = loop_closure_delta(
        np.eye(3), np.zeros(3), np.eye(3), np.zeros(3),
        min_translation_m=0.05, min_rotation_deg=0.5,
    )
    assert d is None


def test_delta_below_translation_threshold_is_none():
    d = loop_closure_delta(
        np.eye(3), np.zeros(3), np.eye(3), np.array([0.01, 0.0, 0.0]),
        min_translation_m=0.05, min_rotation_deg=0.5,
    )
    assert d is None


def test_delta_below_rotation_threshold_is_none():
    d = loop_closure_delta(
        np.eye(3), np.zeros(3), _rz(0.2), np.zeros(3),
        min_translation_m=0.05, min_rotation_deg=0.5,
    )
    assert d is None


def test_delta_no_change_from_nonidentity_baseline_is_none():
    rot, trans = _rz(10.0), np.array([1.0, 2.0, 0.0])
    d = loop_closure_delta(
        rot, trans, rot, trans, min_translation_m=0.05, min_rotation_deg=0.5
    )
    assert d is None


def test_delta_translation_step():
    d = loop_closure_delta(
        np.eye(3), np.zeros(3), np.eye(3), np.array([0.8, 0.0, 0.0]),
        min_translation_m=0.05, min_rotation_deg=0.5,
    )
    assert d is not None
    rot_delta, trans_delta, norm, ang = d
    assert np.allclose(rot_delta, np.eye(3))
    assert np.allclose(trans_delta, [0.8, 0.0, 0.0])
    assert math.isclose(norm, 0.8, rel_tol=1e-9)
    assert ang < 1e-9


def test_delta_rotation_step():
    d = loop_closure_delta(
        np.eye(3), np.zeros(3), _rz(3.0), np.zeros(3),
        min_translation_m=0.05, min_rotation_deg=0.5,
    )
    assert d is not None
    rot_delta, _trans, norm, ang = d
    assert np.allclose(rot_delta, _rz(3.0))
    assert math.isclose(ang, 3.0, abs_tol=1e-6)
    assert norm < 1e-9


def test_delta_is_left_composition_T_new_equals_delta_times_T_old():
    rot_old, trans_old = _rz(10.0), np.array([1.0, 2.0, 3.0])
    rot_new, trans_new = _rz(25.0), np.array([1.5, -0.3, 3.0])
    d = loop_closure_delta(
        rot_old, trans_old, rot_new, trans_new,
        min_translation_m=0.05, min_rotation_deg=0.5,
    )
    assert d is not None
    rot_delta, trans_delta, _n, _a = d
    # delta . T_old == T_new
    assert np.allclose(rot_delta @ rot_old, rot_new)
    assert np.allclose(rot_delta @ trans_old + trans_delta, trans_new)
