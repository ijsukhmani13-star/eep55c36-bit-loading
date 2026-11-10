"""Synthetic DMT/OFDM channels for the bit-loading experiments.

All observations produced here are SYNTHETIC. ``make_loading_case`` follows the
generator in the course mathematical guide (Track D, page 3); the two stress
variants are additions documented in the README.

Seed convention (fixed before any final test was run)
-----------------------------------------------------
development : 0 - 19      (debugging, choosing problem sizes)
validation  : 100 - 119   (choosing the energy-per-tone value rho and timing repeats)
final test  : 1000 - 1029 (reported results only)
"""
from dataclasses import dataclass

import numpy as np

DEV_SEEDS = range(0, 20)
VAL_SEEDS = range(100, 120)
TEST_SEEDS = range(1000, 1030)


@dataclass(frozen=True)
class LoadingCase:
    """One independent channel realisation.

    Attributes
    ----------
    seed : int
    g_true : np.ndarray, shape (N,)
        True channel-to-noise ratio per unit energy |h_i|^2 / sigma^2 (ideal information).
    g_hat : np.ndarray, shape (N,)
        Calibrated estimate from noisy pilots and noise-only observations.
    """

    seed: int
    g_true: np.ndarray
    g_hat: np.ndarray


def make_loading_case(seed, tones=64, taps=8, noise_var=0.1, pilots=20,
                      delay_decay=2.0):
    """Generate one multipath channel and its calibrated estimate.

    Parameters
    ----------
    seed : int
        Seed for ``numpy.random.default_rng`` (makes the case reproducible).
    tones : int
        Number of DMT tones N.
    taps : int
        Number of delayed channel taps L.
    noise_var : float
        Per-tone complex noise variance sigma^2 (> 0).
    pilots : int
        Number of unit-pilot and noise-only observations used for calibration.
    delay_decay : float
        Tap powers decay as exp(-l / delay_decay). 2.0 is the guide's profile;
        a larger value gives a more frequency-selective channel (stress case).

    Returns
    -------
    LoadingCase
        ``g_true`` and ``g_hat`` both have shape (tones,), are positive and finite.
    """
    rng = np.random.default_rng(seed)
    power = np.exp(-np.arange(taps) / delay_decay)
    power /= power.sum()

    channel_taps = np.sqrt(power / 2) * (
        rng.normal(size=taps) + 1j * rng.normal(size=taps))
    h = np.fft.fft(channel_taps, n=tones)

    # Noisy unit-pilot observations: y = h + noise; averaging estimates h.
    noise = np.sqrt(noise_var / 2) * (
        rng.normal(size=(pilots, tones)) + 1j * rng.normal(size=(pilots, tones)))
    h_hat = (h[None, :] + noise).mean(axis=0)

    # Independent noise-only calibration observations estimate sigma^2 per tone.
    noise_only = np.sqrt(noise_var / 2) * (
        rng.normal(size=(pilots, tones)) + 1j * rng.normal(size=(pilots, tones)))
    variance_hat = np.mean(np.abs(noise_only) ** 2, axis=0)

    g_true = np.maximum(np.abs(h) ** 2 / noise_var, 1e-12)
    g_hat = np.maximum(np.abs(h_hat) ** 2 / np.maximum(variance_hat, 1e-12), 1e-12)
    return LoadingCase(seed=int(seed), g_true=g_true, g_hat=g_hat)


def make_flat_case(tones=64, value=10.0):
    """Stress case: a perfectly flat channel (every tone has the same g).

    Every selection is then an exact tie, so this checks that the two methods apply
    the same first-index tie rule. Returns an array of shape (tones,).
    """
    return np.full(tones, float(value))
