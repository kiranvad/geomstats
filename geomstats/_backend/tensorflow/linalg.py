"""Tensorflow based linear algebra backend."""

import scipy as _scipy
import tensorflow as _tf

# "Forward-import" primitives. Due to the way the 'linalg' module is exported
# in TF, this does not work with 'from tensorflow.linalg import ...'.
det = _tf.linalg.det
eigh = _tf.linalg.eigh
expm = _tf.linalg.expm
inv = _tf.linalg.inv
sqrtm = _tf.linalg.sqrtm
solve = _tf.linalg.solve


def norm(x, dtype=_tf.float32, **kwargs):
    x = _tf.cast(x, dtype)
    return _tf.linalg.norm(x, **kwargs)


def eig(*args, **kwargs):
    raise NotImplementedError


def eigvalsh(a, **_unused_kwargs):
    return _tf.linalg.eigvalsh(a)


def cholesky(a, **_unused_kwargs):
    return _tf.linalg.cholesky(a)


def logm(x):
    original_type = x.dtype
    x = _tf.cast(x, _tf.complex64)
    tf_logm = _tf.linalg.logm(x)
    tf_logm = _tf.cast(tf_logm, original_type)
    return tf_logm


def matrix_rank(a, **_unused_kwargs):
    return _tf.linalg.matrix_rank(a)


def svd(x, full_matrices=True, compute_uv=True, **kwargs):
    is_vectorized = x.ndim == 3
    axis = (0, 2, 1) if is_vectorized else (1, 0)
    if compute_uv:
        s, u, v_t = _tf.linalg.svd(
            x, full_matrices=full_matrices, compute_uv=compute_uv
        )
        return u, s, _tf.transpose(v_t, perm=axis)
    return _tf.linalg.svd(x, compute_uv=compute_uv)


def solve_sylvester(a, b, q):
    axes = (0, 2, 1) if a.ndim == 3 else (1, 0)
    if a.shape == b.shape:
        if _tf.reduce_all(a == b) and _tf.reduce_all(
            _tf.abs(a - _tf.transpose(a, perm=axes)) < 1e-8
        ):
            eigvals, eigvecs = eigh(a)
            if _tf.reduce_all(eigvals >= 1e-8):
                tilde_q = _tf.transpose(eigvecs, perm=axes) @ q @ eigvecs
                tilde_x = tilde_q / (eigvals[..., :, None] + eigvals[..., None, :])
                return eigvecs @ tilde_x @ _tf.transpose(eigvecs, perm=axes)

            conditions = (
                a.shape[-1] >= 2
                and _tf.reduce_all(eigvals[..., 0] >= -1e-8)
                and _tf.reduce_all(eigvals[..., 1] >= 1e-8)
                and _tf.reduce_all(_tf.abs(q + _tf.transpose(q, perm=axes)) < 1e-8)
            )

            if conditions:
                tilde_q = _tf.transpose(eigvecs, perm=axes) @ q @ eigvecs
                safe_denominator = (
                    eigvals[..., :, None] + eigvals[..., None, :] + _tf.eye(a.shape[-1])
                )
                tilde_x = tilde_q / safe_denominator
                return eigvecs @ tilde_x @ _tf.transpose(eigvecs, perm=axes)

    raise RuntimeError(
        "solve_sylvester is only implemented if a = b, a symmetric and either a is "
        "positive definite or q is skew-symmetric and a is positive "
        "semi-definite with the first two eigen values positive"
    )


def quadratic_assignment(a, b, options):
    return list(_scipy.optimize.quadratic_assignment(a, b, options=options).col_ind)


def qr(x, mode="reduced"):
    return _tf.linalg.qr(x, full_matrices=(mode == "complete"))


def is_single_matrix_pd(mat):
    """Check if 2D square matrix is positive definite."""
    if mat.shape[0] != mat.shape[1]:
        return False
    try:
        cf = _tf.linalg.cholesky(mat)
        return ~_tf.math.reduce_any(_tf.math.is_nan(cf)).numpy()
    except _tf.errors.InvalidArgumentError as e:
        if "Cholesky decomposition was not successful" in e.message:
            return False
        raise e
