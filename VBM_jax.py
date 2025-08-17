import os
# This flag must be set before jax is imported.
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

import jax
import jax.numpy as jnp
from jax import random, vmap, jit, lax
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm
import corner
import time
import pickle

# --- JAX CONFIGURATION ---
from jax import config
config.update("jax_enable_x64", True)

# ==============================================================================
# SECTION 1: THE MICROLENSING MODEL WEAPON (Corrected from previous steps)
# ==============================================================================

def binary_magnification_point_source(s, q, y1, y2, return_roots=False):
    """Calculates point-source magnification with root filtering to discard spurious images."""
    s_internal, q_internal = lax.cond(
        q > 1,
        lambda: (s, 1.0 / q),
        lambda: (-s, q)
    )
    m1 = 1.0 / (1.0 + q_internal)
    m2 = q_internal * m1
    a = s_internal
    
    yi = y1 + 1j * y2
    y = yi + a * m1
    
    y_conj = jnp.conjugate(y)
    a_sq, a_cub = a * a, a * a * a; m2_sq, a_m2 = m2 * m2, a * m2
    a_sq_m2_sq = a_sq * m2_sq; c12 = a - y_conj; c16 = a * y; c17 = jnp.conjugate(c16)
    c0 = a_sq_m2_sq * y
    c1 = -a_sq_m2_sq + a_m2 * (a + (2 * c17 - 2 - a_sq) * y)
    c2 = a_m2 * (1 + c16 - 2 * y_conj * (a + y)) - (c17 - 1) * (c16 * c12 - jnp.conjugate(c12))
    c3 = a_m2 * jnp.conjugate(a + y + y) + (a_cub + 2 * (1 + a_sq) * y - c17 * (a + y + y)) * y_conj - a * (a + y)
    c4 = -a_m2 - c12 * (y_conj * (a + y + a) - 1)
    c5 = y_conj * c12
    coeffs = jnp.array([c5, c4, c3, c2, c1, c0])
    roots = jnp.roots(coeffs, strip_zeros=False)
    z = roots
    
    rhs_conj = jnp.conjugate(z) - m1 / (z - a) - m2 / z
    error = jnp.abs(rhs_conj - jnp.conjugate(y))
    is_physical = error < 1e-6
    
    J1 = m1 / ((z - a) * (z - a)) + m2 / (z * z)
    det_J = 1.0 - (J1 * jnp.conjugate(J1)).real
    magnifications = 1.0 / jnp.abs(det_J)
    
    total_magnification = jnp.sum(jnp.where(is_physical, magnifications, 0.0))
    
    return (total_magnification, roots) if return_roots else total_magnification

def LDprofile_linear(r, u1=0.6):
    r_safe = jnp.minimum(r, 0.999999)
    return 1. - u1 * (1. - jnp.sqrt(1. - r_safe**2))

def BinaryMagDark(s, q, y1_center, y2_center, rho, n_rings, n_points_per_ring, ld_coeff, gauss_nodes_r, gauss_weights_r):
    u_norm = (gauss_nodes_r + 1) / 2
    sample_radii_norm = jnp.sqrt(u_norm)
    sample_radii = sample_radii_norm * rho
    thetas = jnp.linspace(0, 2 * jnp.pi, n_points_per_ring, endpoint=False)
    R, T = jnp.meshgrid(sample_radii, thetas)
    y1_points = y1_center + (R * jnp.cos(T)).ravel()
    y2_points = y2_center + (R * jnp.sin(T)).ravel()
    point_mags = vmap(binary_magnification_point_source, in_axes=(None, None, 0, 0))(s, q, y1_points, y2_points)
    ring_mags = point_mags.reshape(n_points_per_ring, n_rings).mean(axis=0)
    ring_brightness = LDprofile_linear(sample_radii_norm, u1=ld_coeff)
    weighted_mags = ring_mags * ring_brightness * gauss_weights_r
    total_flux_weight = ring_brightness * gauss_weights_r
    return jnp.sum(weighted_mags) / jnp.sum(total_flux_weight)

def BinaryMag2(s, q, y1, y2, rho, n_rings, n_points_per_ring, ld_coeff, gauss_nodes_r, gauss_weights_r):
    y2_abs = jnp.abs(y2)
    return lax.cond(
        rho < 1e-4,
        lambda: binary_magnification_point_source(s, q, y1, y2_abs),
        lambda: BinaryMagDark(s, q, y1, y2_abs, rho, n_rings, n_points_per_ring, ld_coeff, gauss_nodes_r, gauss_weights_r)
    )

def VBM_BinaryLightCurve(pr, t, use_extended_source, n_rings, n_points_per_ring, ld_coeff, gauss_nodes_r, gauss_weights_r):
    log_s, log_q, u0, alpha, log_rho, log_tE, t0 = pr
    s, q, rho, tE = jnp.exp(log_s), jnp.exp(log_q), jnp.exp(log_rho), jnp.exp(log_tE)
    tau = (t - t0) / tE
    sin_alpha, cos_alpha = jnp.sin(alpha), jnp.cos(alpha)
    y1 = u0 * sin_alpha - tau * cos_alpha
    y2 = -u0 * cos_alpha - tau * sin_alpha
    return lax.cond(
        use_extended_source,
        lambda: BinaryMag2(s, q, y1, y2, rho, n_rings, n_points_per_ring, ld_coeff, gauss_nodes_r, gauss_weights_r),
        lambda: binary_magnification_point_source(s, q, y1, jnp.abs(y2))
    )