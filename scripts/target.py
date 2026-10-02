"""Stable positional target mask; validate weights columnwise once, without repeated full-frame reductions."""

import numpy as np


def target_mask(frame, weight_columns):
    good = (
        np.isfinite(frame.ESCS.to_numpy())
        & (frame.Option_ICTQ.to_numpy() == 1)
        & (frame.W_FSTUWT.to_numpy() > 0)
    )
    for name in weight_columns:
        a = frame[name].to_numpy()
        good &= np.isfinite(a) & (a >= 0)
    return good
