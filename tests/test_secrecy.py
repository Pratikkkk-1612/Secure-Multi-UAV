from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np
from numpy.testing import assert_allclose

from config import DEFAULT_CONFIG
from security.secrecy_rate import evaluate_security


def _case(gu_h, eav_h, schedule, power):
    gu_h = np.asarray(gu_h, dtype=complex)
    eav_h = np.asarray(eav_h, dtype=complex)
    schedule = np.asarray(schedule, dtype=int)
    n_rb = schedule.shape[1]
    network = replace(
        DEFAULT_CONFIG.network,
        n_uav=gu_h.shape[0], n_gu=gu_h.shape[1], n_eav=eav_h.shape[1],
        n_resource_blocks_per_uav=n_rb,
        min_gu_per_uav=0, max_gu_per_uav=n_rb,
    )
    # Synchronize test-only RB counts without changing production defaults.
    cfg = replace(
        DEFAULT_CONFIG, network=network,
        a2g=replace(DEFAULT_CONFIG.a2g, n_resource_blocks=n_rb),
        a2a=replace(DEFAULT_CONFIG.a2a, n_resource_blocks=n_rb),
    )
    ground = SimpleNamespace(h=gu_h)
    air = SimpleNamespace(uav_eav=SimpleNamespace(h=eav_h))
    return ground, air, schedule, power, cfg


def _evaluate(case):
    return evaluate_security(*case[:4], config=case[4])


def _noise(cfg, bandwidth):
    dbm = (
        cfg.thermal_noise_density_dbm_per_hz + 10 * np.log10(bandwidth)
        + cfg.noise_figure_db + cfg.implementation_margin_db
    )
    return 10 ** ((dbm - 30) / 10)


class TestSecrecy(unittest.TestCase):
    def test_one_link_expected_secrecy_and_clipping(self):
        case = _case([[[2, 0]]], [[[1, 0]]], [[1]], 0.1)
        r = _evaluate(case)
        cfg = case[4]
        bw = cfg.a2g.total_bandwidth_hz
        # With two equal antenna coefficients, gains are 2 and 0.5.
        expected = max(bw * (
            np.log2(1 + 0.2 / _noise(cfg.a2g, bw))
            - np.log2(1 + 0.05 / _noise(cfg.a2a, bw))
        ), 0)
        assert_allclose(r.secrecy_rate_bps, [expected])
        assert_allclose(r.total_secrecy_rate_bps, expected)
        assert_allclose(r.transmit_weights, 1 / np.sqrt(2))
        strong = _case([[[1, 0]]], [[[3, 0]]], [[1]], 0.1)
        self.assertEqual(_evaluate(strong).total_secrecy_rate_bps, 0)

    def test_same_rb_interference_and_worst_eav(self):
        case = _case(
            [[[2, 0], [1, 0]], [[1, 0], [2, 0]]],
            [[[0.2, 0], [1, 0]], [[1, 0], [0.2, 0]]],
            [[1], [2]], [0.1, 0.1],
        )
        r = _evaluate(case)
        cfg = case[4]
        bw = cfg.a2g.total_bandwidth_hz
        assert_allclose(r.legitimate_sinr, [0.2 / (0.05 + _noise(cfg.a2g, bw))] * 2)
        ne = _noise(cfg.a2a, bw)
        expected_eav = [[0.002/(0.05+ne), 0.05/(0.002+ne)],
                        [0.05/(0.002+ne), 0.002/(0.05+ne)]]
        assert_allclose(r.eav_sinr, expected_eav)
        assert_allclose(r.worst_eav_rate_bps, bw*np.log2(1+np.max(expected_eav, axis=1)))
        assert_allclose(r.per_uav_secrecy_rate_bps.sum(), r.total_secrecy_rate_bps)

    def test_orthogonal_blocks_and_no_eavesdroppers(self):
        case = _case(
            [[[2, 0], [1, 0]], [[1, 0], [2, 0]]],
            np.empty((2, 0, 2)), [[1, 0], [0, 2]], [0.1, 0.1],
        )
        r = _evaluate(case)
        bw = case[4].a2g.total_bandwidth_hz / 2
        assert_allclose(r.legitimate_sinr, [0.2/_noise(case[4].a2g,bw)]*2)
        self.assertEqual(r.eav_sinr.shape, (2, 0))
        assert_allclose(r.secrecy_rate_bps, r.gu_rate_bps)
        assert_allclose(r.mean_secrecy_rate_bps, r.total_secrecy_rate_bps/2)
        assert_allclose(r.transmit_weights[case[2] == 0], 0)

    def test_invalid_power_schedule_and_channels(self):
        case = _case([[[2, 0]]], [[[1, 0]]], [[1]], 0.1)
        for bad in (-0.01, 0.5, float("nan")):
            with self.subTest(power=bad), self.assertRaises(ValueError):
                evaluate_security(case[0], case[1], case[2], bad, config=case[4])
        for bad_schedule in (np.array([[0]]), np.array([[2]]), np.array([[1.0]])):
            with self.assertRaises(ValueError):
                evaluate_security(case[0], case[1], bad_schedule, 0.1, config=case[4])
        case[0].h[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            _evaluate(case)

    def test_power_is_split_across_active_blocks(self):
        case = _case([[[2, 0], [2, 0]]], np.empty((1, 0, 2)), [[1, 2]], 0.2)
        r = _evaluate(case)
        bw = case[4].a2g.total_bandwidth_hz / 2
        assert_allclose(r.legitimate_sinr, [0.2/_noise(case[4].a2g,bw)]*2)
        assert_allclose(r.per_uav_secrecy_rate_bps, [r.total_secrecy_rate_bps])

    def test_fixed_weights_include_phase_cancellation(self):
        case = _case([[[1, -1]]], np.empty((1, 0, 2)), [[1]], 0.1)
        r = _evaluate(case)
        assert_allclose(r.legitimate_sinr, 0)
        assert_allclose(r.total_secrecy_rate_bps, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
