"""Pytest utility classes, functions and fixtures."""

import os

import numpy as np
import pytest

import geomstats.backend as gs


def autograd_backend():
    """Check if autograd is set as backend."""
    return os.environ["GEOMSTATS_BACKEND"] == "autograd"


def np_backend():
    """Check if numpy is set as backend."""
    return os.environ["GEOMSTATS_BACKEND"] == "numpy"


def pytorch_backend():
    """Check if pytorch is set as backend."""
    return os.environ["GEOMSTATS_BACKEND"] == "pytorch"


def tf_backend():
    """Check if tensorflow is set as backend."""
    return os.environ["GEOMSTATS_BACKEND"] == "tensorflow"


if tf_backend():
    import tensorflow as tf

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

if pytorch_backend():
    import torch


autograd_only = pytest.mark.skipif(
    not autograd_backend(), reason="Test for autograd backend only."
)
np_only = pytest.mark.skipif(not np_backend(), reason="Test for numpy backend only.")
torch_only = pytest.mark.skipif(
    not pytorch_backend(), reason="Test for pytorch backends only."
)
tf_only = pytest.mark.skipif(
    not tf_backend(), reason="Test for tensorflow backends only."
)

np_and_tf_only = pytest.mark.skipif(
    not (np_backend() or tf_backend()),
    reason="Test for numpy and tensorflow backends only.",
)
np_and_torch_only = pytest.mark.skipif(
    not (np_backend() or pytorch_backend()),
    reason="Test for numpy and pytorch backends only.",
)
np_and_autograd_only = pytest.mark.skipif(
    not (np_backend() or autograd_backend()),
    reason="Test for numpy and autograd backends only.",
)
autograd_and_torch_only = pytest.mark.skipif(
    not (autograd_backend() or pytorch_backend()),
    reason="Test for autograd and torch backends only.",
)
autograd_and_tf_only = pytest.mark.skipif(
    not (autograd_backend() or tf_backend()),
    reason="Test for autograd and tf backends only.",
)

np_autograd_and_tf_only = pytest.mark.skipif(
    not (np_backend() or autograd_backend() or tf_backend()),
    reason="Test for numpy, autograd and tensorflow backends only.",
)
np_autograd_and_torch_only = pytest.mark.skipif(
    not (np_backend() or autograd_backend() or pytorch_backend()),
    reason="Test for numpy, autograd and pytorch backends only.",
)
autograd_tf_and_torch_only = pytest.mark.skipif(
    np_backend(), reason="Test for backends with automatic differentiation only."
)


def pytorch_error_msg(a, b, rtol, atol):
    msg = f"\ntensor 1\n{a}\ntensor 2\n{b}"
    if torch.is_tensor(a) and torch.is_tensor(b):
        if a.dtype == torch.bool and b.dtype == torch.bool:
            diff = torch.logical_xor(a, b)
            msg = msg + f"\ndifference \n{diff}"
        else:
            diff = torch.abs(a - b)
            msg = msg + f"\ndifference \n{diff}\nrtol {rtol}\natol {atol}"
    return msg


class TestCase:
    """Class for Geomstats tests."""

    def assertAllClose(self, a, b, rtol=gs.rtol, atol=gs.atol):
        if tf_backend():
            return tf.test.TestCase().assertAllClose(a, b, rtol=rtol, atol=atol)
        if np_backend() or autograd_backend():
            return np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)

        return self.assertTrue(
            gs.allclose(a, b, rtol=rtol, atol=atol),
            msg=pytorch_error_msg(a, b, rtol, atol),
        )
