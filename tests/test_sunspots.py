"""Leakage and correctness checks for solar/sunspots.py, offline (no SILSO download)."""

import numpy as np
import pandas as pd

from solar.sunspots import MAX_H, ROTATION, _baseline, _features, _LSTMNet


def _series(n=900, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    sn = np.clip(80 + 40 * np.sin(2 * np.pi * t / ROTATION) + rng.normal(0, 10, n), 0, None)
    return sn


def test_features_use_only_the_past():
    sn = _series()
    z = pd.Series(np.sqrt(sn))
    t = 600
    for h in (1, 7, 14, 27):
        before = _features(z, h).iloc[t]
        poisoned = z.copy()
        poisoned.iloc[t + 1:] = 1e6          # anything after the origin
        after = _features(poisoned, h).iloc[t]
        pd.testing.assert_series_equal(before, after)


def test_echo_feature_is_one_rotation_before_target():
    z = pd.Series(np.arange(500, dtype=float))
    t, h = 300, 5
    f = _features(z, h).iloc[t]
    assert f["echo27"] == z.iloc[t + h - ROTATION]
    assert f["lag0"] == z.iloc[t]


def test_baselines():
    sn = _series()
    o = np.array([400, 401, 650])
    assert np.array_equal(_baseline("persistence", sn, o, 7), sn[o])
    assert np.array_equal(_baseline("recurrence27", sn, o, 7), sn[o + 7 - ROTATION])
    expect = np.array([sn[i - ROTATION + 1:i + 1].mean() for i in o])
    assert np.allclose(_baseline("mean27", sn, o, 7), expect)


def test_lstm_window_ends_at_origin_and_ignores_the_future():
    sn = _series(400)
    z = np.sqrt(sn)
    o = np.array([200, 300])
    w = _LSTMNet.windows(z, o)
    assert w.shape[1] == 54 and w[0, -1] == z[200] and w[1, -1] == z[300]

    net = _LSTMNet()
    net.mu, net.sd = float(z.mean()), float(z.std())
    a = net.predict(z, o)
    poisoned = z.copy()
    poisoned[301:] = 1e3
    b = net.predict(poisoned, o)
    assert a.shape == (2, MAX_H) and np.allclose(a, b)
