"""
Flux Visualiser v4.0 - 2D Compressible/Laminar CFD & Viscous Airfoil Solver
============================================================================
Upgraded 4th-Year Aerospace Systems Engineering architecture combining:
  1. Cosine-clustered NACA 4-digit generator, cylinder, .txt/.png contour loaders.
  2. Vectorized Hess-Smith constant source + uniform vortex panel solver
     with Kutta trailing-edge condition and Prandtl-Glauert compressibility.
  3. Viscous integral boundary layer coupling:
     Thwaites laminar -> Michel / separation transition -> Head entrainment
     turbulent BL + Ludwieg-Tillmann Cf -> Squire-Young profile drag Cd.
  4. 2nd-order Shortley-Weller immersed-boundary finite-difference Laplace solver
     with SciPy SuperLU sparse factorization, linear superposition
     (psi = psi_0 + C * psi_1), auto-Kutta streamfunction calibration, and
     wall-normal Taylor Cp extraction.
  5. Interactive dark telemetry dashboard + headless `--export-figures` V&V suite.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from matplotlib.path import Path
from scipy.interpolate import RegularGridInterpolator


# ---------------------------------------------------------------------------
# 1. Geometry & Coordinate Utilities
# ---------------------------------------------------------------------------


def ensure_clockwise(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure closed polygon coordinates follow clockwise order (TE->lower->LE->upper->TE)."""
    area2 = np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    if area2 > 0.0:
        return x[::-1].copy(), y[::-1].copy()
    return x, y


def generate_naca4(code: str = "4412", n_half: int = 80) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate closed cosine-clustered NACA 4-digit airfoil coordinates.
    Ordered clockwise: TE (lower) -> LE -> TE (upper), unit chord c = 1.0
    centered around mid-chord (x in [-0.5, 0.5]).
    """
    digits = "".join(ch for ch in str(code) if ch.isdigit())
    if len(digits) != 4:
        raise ValueError(f"Expected 4-digit NACA code, got: {code!r}")

    m = int(digits[0]) / 100.0
    p = int(digits[1]) / 10.0
    t = int(digits[2:]) / 100.0

    beta = np.linspace(0.0, np.pi, n_half + 1)
    xc = 0.5 * (1.0 - np.cos(beta))

    # Closed trailing edge coefficient a4 = -0.1036
    yt = 5.0 * t * (
        0.2969 * np.sqrt(np.maximum(xc, 0.0))
        - 0.1260 * xc
        - 0.3516 * xc**2
        + 0.2843 * xc**3
        - 0.1036 * xc**4
    )

    if m > 0.0 and p > 0.0:
        yc = np.where(
            xc < p,
            (m / (p**2)) * (2.0 * p * xc - xc**2),
            (m / ((1.0 - p) ** 2)) * ((1.0 - 2.0 * p) + 2.0 * p * xc - xc**2),
        )
        dyc = np.where(
            xc < p,
            (2.0 * m / (p**2)) * (p - xc),
            (2.0 * m / ((1.0 - p) ** 2)) * (p - xc),
        )
    else:
        yc = np.zeros_like(xc)
        dyc = np.zeros_like(xc)

    theta = np.arctan(dyc)
    xu = xc - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = xc + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # Force exact closure at trailing edge
    xu[-1] = 1.0
    xl[-1] = 1.0
    y_te = 0.5 * (yu[-1] + yl[-1])
    yu[-1] = y_te
    yl[-1] = y_te

    x_cw = np.concatenate([xl[::-1], xu[1:]]) - 0.5
    y_cw = np.concatenate([yl[::-1], yu[1:]])
    return ensure_clockwise(x_cw, y_cw)


def thin_airfoil_zero_lift_rad(code: str = "4412", n_pts: int = 400) -> float:
    """Compute theoretical Glauert thin-airfoil zero-lift angle alpha_L0 (radians)."""
    digits = "".join(ch for ch in str(code) if ch.isdigit())
    if len(digits) != 4:
        return 0.0
    m = int(digits[0]) / 100.0
    p = int(digits[1]) / 10.0
    if m == 0.0 or p == 0.0:
        return 0.0

    th = np.linspace(1e-6, np.pi - 1e-6, n_pts)
    xc = 0.5 * (1.0 - np.cos(th))
    dyc = np.where(
        xc < p,
        (2.0 * m / (p**2)) * (p - xc),
        (2.0 * m / ((1.0 - p) ** 2)) * (p - xc),
    )
    integrand = dyc * (np.cos(th) - 1.0)
    return float(-np.trapezoid(integrand, th) / np.pi)


def generate_cylinder(radius: float = 0.35, n_pts: int = 360) -> Tuple[np.ndarray, np.ndarray]:
    """Generate closed clockwise circular cylinder contour centered at origin."""
    th = np.linspace(0.0, 2.0 * np.pi, n_pts + 1)
    x_cw = radius * np.cos(th[::-1])
    y_cw = radius * np.sin(th[::-1])
    return x_cw, y_cw


def _normalize_contour(x: np.ndarray, y: np.ndarray, chord: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Translate contour centroid/bounds to mid-chord origin, scale to unit chord, close & orient CW."""
    if x[0] != x[-1] or y[0] != y[-1]:
        x = np.append(x, x[0])
        y = np.append(y, y[0])
    xmin, xmax = float(np.min(x)), float(np.max(x))
    scale = max(xmax - xmin, 1e-6)
    x_norm = (x - 0.5 * (xmin + xmax)) * (chord / scale)
    y_norm = (y - 0.5 * (float(np.min(y)) + float(np.max(y)))) * (chord / scale)
    # Place trailing edge (maximum x) first so panel Kutta ordering stays consistent
    te_idx = int(np.argmax(x_norm[:-1]))
    x_rot = np.roll(x_norm[:-1], -te_idx)
    y_rot = np.roll(y_norm[:-1], -te_idx)
    x_closed = np.append(x_rot, x_rot[0])
    y_closed = np.append(y_rot, y_rot[0])
    return ensure_clockwise(x_closed, y_closed)


def process_txt_file(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load Selig-formatted airfoil coordinates (skipping header line) and normalize CW."""
    data = np.loadtxt(filepath, skiprows=1)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Invalid coordinate table in {filepath}")
    return _normalize_contour(data[:, 0].astype(np.float64), data[:, 1].astype(np.float64))


def process_png_image(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract largest solid silhouette contour from binary/grayscale image using OpenCV."""
    import cv2

    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {filepath}")
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError(f"No closed contour detected in {filepath}")
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.002 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True).reshape(-1, 2)
    x_raw = approx[:, 0].astype(np.float64)
    y_raw = -approx[:, 1].astype(np.float64)  # Image y points downwards
    return _normalize_contour(x_raw, y_raw)


def rotate_coords(x: np.ndarray, y: np.ndarray, alpha_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate body by -alpha_deg around origin so freestream remains +x."""
    rad = np.radians(alpha_deg)
    ca, sa = np.cos(-rad), np.sin(-rad)
    return x * ca - y * sa, x * sa + y * ca


# ---------------------------------------------------------------------------
# 2. Vectorized Hess-Smith Panel Method + Compressibility
# ---------------------------------------------------------------------------


def solve_hess_smith_panel(
    x: np.ndarray,
    y: np.ndarray,
    Uinf: float = 60.0,
    alpha_deg: float = 0.0,
    a_sound: float = 340.3,
    enforce_kutta: bool = True,
) -> Dict[str, np.ndarray | float]:
    """
    Constant-strength source + uniform vortex Hess-Smith panel method.
    Nodes (x, y) must be ordered clockwise: TE (lower) -> LE -> TE (upper).
    Applies Prandtl-Glauert compressibility correction beta = sqrt(1 - M^2).
    """
    N = len(x) - 1
    dx_p = x[1:] - x[:-1]
    dy_p = y[1:] - y[:-1]
    L = np.hypot(dx_p, dy_p)
    theta = np.arctan2(dy_p, dx_p)
    xm = 0.5 * (x[:-1] + x[1:])
    ym = 0.5 * (y[:-1] + y[1:])
    nx = -np.sin(theta)
    ny = np.cos(theta)
    tx = np.cos(theta)
    ty = np.sin(theta)

    dx_ij = xm[:, None] - x[:-1][None, :]
    dy_ij = ym[:, None] - y[:-1][None, :]
    cos_j = np.cos(theta)[None, :]
    sin_j = np.sin(theta)[None, :]

    x_loc = dx_ij * cos_j + dy_ij * sin_j
    y_loc = -dx_ij * sin_j + dy_ij * cos_j
    L_j = L[None, :]

    r1_sq = np.maximum(x_loc**2 + y_loc**2, 1e-24)
    r2_sq = np.maximum((x_loc - L_j) ** 2 + y_loc**2, 1e-24)

    eye = np.eye(N, dtype=bool)
    log_r2_r1 = 0.5 * np.log(np.where(eye, 1.0, r2_sq) / np.where(eye, 1.0, r1_sq))
    np.fill_diagonal(log_r2_r1, 0.0)

    beta = np.arctan2(y_loc * L_j, x_loc**2 + y_loc**2 - x_loc * L_j)
    np.fill_diagonal(beta, np.pi)

    inv_2pi = 1.0 / (2.0 * np.pi)
    us_loc = -log_r2_r1 * inv_2pi
    vs_loc = beta * inv_2pi
    uv_loc = beta * inv_2pi
    vv_loc = log_r2_r1 * inv_2pi

    dtheta = theta[:, None] - theta[None, :]
    sin_dt = np.sin(dtheta)
    cos_dt = np.cos(dtheta)

    A_qn = -us_loc * sin_dt + vs_loc * cos_dt
    A_qt = us_loc * cos_dt + vs_loc * sin_dt
    A_gn = (-uv_loc * sin_dt + vv_loc * cos_dt).sum(axis=1)
    A_gt = (uv_loc * cos_dt + vv_loc * sin_dt).sum(axis=1)

    if enforce_kutta:
        A = np.zeros((N + 1, N + 1), dtype=np.float64)
        b = np.zeros(N + 1, dtype=np.float64)
        A[:N, :N] = A_qn
        A[:N, N] = A_gn
        b[:N] = -Uinf * nx
        A[N, :N] = A_qt[0, :] + A_qt[N - 1, :]
        A[N, N] = A_gt[0] + A_gt[N - 1]
        b[N] = -Uinf * (tx[0] + tx[N - 1])
        sol = np.linalg.solve(A, b)
        q = sol[:N]
        gamma = float(sol[N])
    else:
        q = np.linalg.solve(A_qn, -Uinf * nx)
        gamma = 0.0

    Vt = Uinf * tx + A_qt @ q + A_gt * gamma
    Cp0 = 1.0 - (Vt / max(Uinf, 1e-6)) ** 2

    Mach = Uinf / max(a_sound, 1e-6)
    beta_pg = float(np.sqrt(max(1e-4, 1.0 - Mach**2)))
    Cp = Cp0 / beta_pg

    perimeter = float(np.sum(L))
    Gamma = gamma * perimeter
    chord = max(float(np.max(x) - np.min(x)), 1e-6)
    # Use unit reference chord for airfoil; for rotated airfoils keep c = 1.0
    c_ref = 1.0 if chord <= 1.05 else chord
    Cl_kj = float((2.0 * Gamma / (max(Uinf, 1e-6) * c_ref)) / beta_pg)

    fx = float(np.sum(-Cp * nx * L) / c_ref)
    fy = float(np.sum(-Cp * ny * L) / c_ref)

    rad = np.radians(alpha_deg)
    x_qc = -0.25 * np.cos(-rad)
    y_qc = -0.25 * np.sin(-rad)
    Cm_qc = float(np.sum(-Cp * ((ym - y_qc) * nx - (xm - x_qc) * ny) * L) / (c_ref**2))

    # Inverse rotation by +alpha restores body-frame chord coordinate x/c in [0, 1]
    xc_body = xm * np.cos(rad) - ym * np.sin(rad) + 0.5

    return {
        "xm": xm,
        "ym": ym,
        "xc_body": np.clip(xc_body, 0.0, 1.0),
        "L": L,
        "nx": nx,
        "ny": ny,
        "Vt": Vt,
        "Cp0": Cp0,
        "Cp": Cp,
        "Gamma": Gamma,
        "Cl_kj": Cl_kj,
        "Cl_p": fy,
        "Cd_p": fx,
        "Cm_qc": Cm_qc,
        "Mach": Mach,
        "beta_pg": beta_pg,
    }


# ---------------------------------------------------------------------------
# 3. Viscous Integral Boundary Layer Solver (Thwaites + Michel + Head + SY)
# ---------------------------------------------------------------------------


def solve_boundary_layer_viscous(
    xm: np.ndarray,
    ym: np.ndarray,
    xc_body: np.ndarray,
    L: np.ndarray,
    Vt: np.ndarray,
    Uinf: float = 60.0,
    nu: float = 1.48e-5,
) -> Dict[str, object]:
    """
    Viscous boundary layer marching along upper & lower surfaces from stagnation:
      - Laminar: Thwaites momentum integral + Pohlhausen shape factor H(lambda)
      - Transition: Michel criterion (Re_theta >= 1.174*(1 + 22400/Re_s)*Re_s^0.46)
                    or laminar separation (lambda <= -0.088)
      - Turbulent: Head entrainment + Ludwieg-Tillmann skin friction Cf
      - Drag: Squire-Young trailing-edge momentum defect integration
    """
    stag_idx = int(np.argmin(np.abs(Vt)))
    for k in range(len(Vt) - 1):
        if Vt[k] <= 0.0 and Vt[k + 1] >= 0.0:
            stag_idx = k if abs(Vt[k]) < abs(Vt[k + 1]) else k + 1
            break

    branches = {
        "upper": (
            xm[stag_idx:],
            ym[stag_idx:],
            xc_body[stag_idx:],
            np.abs(Vt[stag_idx:]),
        ),
        "lower": (
            xm[: stag_idx + 1][::-1],
            ym[: stag_idx + 1][::-1],
            xc_body[: stag_idx + 1][::-1],
            np.abs(Vt[: stag_idx + 1][::-1]),
        ),
    }

    cd_total = 0.0
    x_tr: Dict[str, float] = {}
    branch_data: Dict[str, Dict[str, np.ndarray]] = {}

    for name, (xb, yb, xcb, Ue) in branches.items():
        if len(xb) < 4:
            x_tr[name] = 1.0
            continue

        ds = np.maximum(np.hypot(np.diff(xb), np.diff(yb)), 1e-8)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        Ue = np.maximum(Ue, 1e-3 * max(Uinf, 1.0))
        dUds = np.gradient(Ue, s, edge_order=1)

        n = len(s)
        theta = np.zeros(n, dtype=np.float64)
        H = np.zeros(n, dtype=np.float64)
        cf_arr = np.zeros(n, dtype=np.float64)

        dU0 = max(abs(dUds[0]), Uinf / 0.1)
        theta[0] = np.sqrt(0.075 * nu / dU0)
        H[0] = 2.2
        cf_arr[0] = 0.0
        theta2_int = 0.0

        tr_idx = n - 1
        transitioned = False

        for i in range(1, n):
            u5_prev = Ue[i - 1] ** 5
            u5_curr = Ue[i] ** 5
            theta2_int += 0.5 * (u5_prev + u5_curr) * ds[i - 1]
            theta_sq = (0.45 * nu / (Ue[i] ** 6)) * max(theta2_int, 1e-24)
            theta[i] = np.sqrt(theta_sq)
            lam_raw = (theta_sq / nu) * dUds[i]
            lam = float(np.clip(lam_raw, -0.09, 0.25))

            if lam >= 0.0:
                H[i] = 2.61 - 3.75 * lam + 5.24 * lam**2
            else:
                H[i] = 2.088 + 0.0731 / (lam + 0.14)

            Re_th = Ue[i] * theta[i] / nu
            l_lam = 0.22 + 1.57 * lam - 1.8 * lam**2 if lam >= 0.0 else 0.22 + 1.402 * lam + 0.018 * lam / (lam + 0.107)
            cf_arr[i] = 2.0 * l_lam / max(Re_th, 10.0)

            Re_s = max(Ue[i] * s[i] / nu, 1.0)
            michel_lim = 1.174 * (1.0 + 22400.0 / Re_s) * (Re_s**0.46)
            if Re_th >= michel_lim or lam_raw <= -0.088:
                tr_idx = i
                transitioned = True
                break

        if transitioned and tr_idx < n - 1:
            th_t = max(theta[tr_idx], 1e-6)
            H_t = 1.4
            H[tr_idx] = H_t
            for i in range(tr_idx, n - 1):
                Re_th = max(Ue[i] * th_t / nu, 200.0)
                cf = 0.246 * (10.0 ** (-0.678 * H_t)) * (Re_th**-0.268)
                cf_arr[i] = cf
                if H_t <= 1.6:
                    H1 = 3.3 + 0.8234 * (max(H_t - 1.1, 0.05) ** -1.287)
                else:
                    H1 = 3.3 + 1.5501 * (max(H_t - 0.6778, 0.05) ** -3.064)
                CE = 0.0306 * (max(H1 - 3.0, 0.2) ** -0.6169)

                dth_ds = 0.5 * cf - (H_t + 2.0) * (th_t / Ue[i]) * dUds[i]
                dH1_ds = (CE - H1 * (0.5 * cf - (H_t + 1.0) * (th_t / Ue[i]) * dUds[i])) / max(th_t, 1e-7)

                th_next = max(th_t + dth_ds * ds[i], 1e-7)
                H1_next = float(np.clip(H1 + dH1_ds * ds[i], 3.35, 16.0))
                if H1_next >= 5.3:
                    H_next = 1.1 + ((H1_next - 3.3) / 0.8234) ** (-1.0 / 1.287)
                else:
                    H_next = 0.6778 + ((H1_next - 3.3) / 1.5501) ** (-1.0 / 3.064)
                H_t = float(np.clip(H_next, 1.22, 2.5))
                th_t = th_next
                theta[i + 1] = th_t
                H[i + 1] = H_t
            cf_arr[-1] = cf_arr[-2]

        x_tr[name] = float(np.clip(xcb[tr_idx], 0.0, 1.0))
        cd_branch = float(2.0 * theta[-1] * ((Ue[-1] / max(Uinf, 1e-6)) ** ((H[-1] + 5.0) / 2.0)))
        cd_total += cd_branch
        branch_data[name] = {
            "xc": xcb,
            "s": s,
            "theta": theta,
            "H": H,
            "cf": cf_arr,
            "tr_idx": np.array([tr_idx]),
        }

    return {
        "cd_visc": cd_total,
        "x_tr_upper": x_tr.get("upper", 1.0),
        "x_tr_lower": x_tr.get("lower", 1.0),
        "branches": branch_data,
        "Re_c": float(Uinf * 1.0 / nu),
    }


# ---------------------------------------------------------------------------
# 4. Shortley-Weller Immersed-Boundary Sparse LU Laplace Solver
# ---------------------------------------------------------------------------


def ray_segment_intersection_1d(
    p0: Tuple[float, float], p1: Tuple[float, float], poly_x: np.ndarray, poly_y: np.ndarray
) -> float:
    """Return normalized distance theta in [1e-4, 1.0] where segment p0->p1 hits polygon."""
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    x1 = poly_x[:-1]
    y1 = poly_y[:-1]
    sx = poly_x[1:] - x1
    sy = poly_y[1:] - y1
    denom = dx * sy - dy * sx
    valid = np.abs(denom) > 1e-14
    qpx = x1 - p0[0]
    qpy = y1 - p0[1]
    safe_denom = np.where(valid, denom, 1.0)
    s = np.where(valid, (qpx * sy - qpy * sx) / safe_denom, np.inf)
    u = np.where(valid, (qpx * dy - qpy * dx) / safe_denom, np.inf)
    hit = valid & (s > -1e-9) & (s <= 1.05) & (u >= -1e-4) & (u <= 1.0 + 1e-4)
    if np.any(hit):
        return float(np.clip(np.min(s[hit]), 1e-4, 1.0))
    return 0.5


def assemble_laplace_sparse(
    x_cw: np.ndarray,
    y_cw: np.ndarray,
    Xrange: float = 3.2,
    Yrange: float = 2.2,
    dx: float = 0.015,
    on_wall_geom_radius: Optional[float] = None,
) -> Dict[str, object]:
    """
    Assemble and LU-factorize the 2nd-order Shortley-Weller immersed-boundary
    Laplace matrix for psi_xx + psi_yy = 0 on a Cartesian grid.
    """
    t0 = time.perf_counter()
    Xdiv = int(round(Xrange / dx)) + 1
    Ydiv = int(round(Yrange / dx)) + 1
    x = np.linspace(-Xrange / 2.0, Xrange / 2.0, Xdiv)
    y = np.linspace(-Yrange / 2.0, Yrange / 2.0, Ydiv)
    X, Y = np.meshgrid(x, y)
    dx_act = float(x[1] - x[0])
    dy_act = float(y[1] - y[0])

    poly = Path(np.column_stack([x_cw, y_cw]))
    solid = poly.contains_points(np.column_stack([X.ravel(), Y.ravel()])).reshape((Ydiv, Xdiv))
    if on_wall_geom_radius is not None:
        on_wall = np.abs(np.hypot(X, Y) - on_wall_geom_radius) < 1e-5 * dx_act
        solid = solid | on_wall

    is_outer = np.zeros((Ydiv, Xdiv), dtype=bool)
    is_outer[0, :] = True
    is_outer[-1, :] = True
    is_outer[:, 0] = True
    is_outer[:, -1] = True

    is_dirichlet = is_outer | solid
    fluid = ~is_dirichlet

    N_nodes = Xdiv * Ydiv
    idx_grid = np.arange(N_nodes).reshape((Ydiv, Xdiv))

    adj_solid = np.zeros((Ydiv, Xdiv), dtype=bool)
    adj_solid[1:-1, 1:-1] = (
        solid[1:-1, 2:] | solid[1:-1, :-2] | solid[2:, 1:-1] | solid[:-2, 1:-1]
    )

    reg_fluid = fluid & (~adj_solid)
    cut_fluid = fluid & adj_solid

    dir_idx = idx_grid[is_dirichlet]
    reg_idx = idx_grid[reg_fluid]

    ax = 1.0 / (dx_act**2)
    ay = 1.0 / (dy_act**2)
    ac = -2.0 * (ax + ay)

    rows = [dir_idx, reg_idx, reg_idx, reg_idx, reg_idx, reg_idx]
    cols = [dir_idx, reg_idx, reg_idx + 1, reg_idx - 1, reg_idx + Xdiv, reg_idx - Xdiv]
    vals = [
        np.ones(len(dir_idx), dtype=np.float64),
        np.full(len(reg_idx), ac, dtype=np.float64),
        np.full(len(reg_idx), ax, dtype=np.float64),
        np.full(len(reg_idx), ax, dtype=np.float64),
        np.full(len(reg_idx), ay, dtype=np.float64),
        np.full(len(reg_idx), ay, dtype=np.float64),
    ]

    cut_j, cut_i = np.where(cut_fluid)
    sw_rows, sw_cols, sw_vals = [], [], []
    wall_rhs1 = np.zeros((Ydiv, Xdiv), dtype=np.float64)
    wall_fracs: Dict[Tuple[int, int], Tuple[float, float, float, float]] = {}

    for j, i in zip(cut_j, cut_i):
        p0 = (float(x[i]), float(y[j]))
        p_idx = int(idx_grid[j, i])

        th_e = ray_segment_intersection_1d(p0, (float(x[i + 1]), float(y[j])), x_cw, y_cw) if solid[j, i + 1] else 1.0
        th_w = ray_segment_intersection_1d(p0, (float(x[i - 1]), float(y[j])), x_cw, y_cw) if solid[j, i - 1] else 1.0
        th_n = ray_segment_intersection_1d(p0, (float(x[i]), float(y[j + 1])), x_cw, y_cw) if solid[j + 1, i] else 1.0
        th_s = ray_segment_intersection_1d(p0, (float(x[i]), float(y[j - 1])), x_cw, y_cw) if solid[j - 1, i] else 1.0
        wall_fracs[(int(j), int(i))] = (th_e, th_w, th_n, th_s)

        ce = 2.0 / ((dx_act**2) * th_e * (th_e + th_w))
        cw = 2.0 / ((dx_act**2) * th_w * (th_e + th_w))
        cn = 2.0 / ((dy_act**2) * th_n * (th_n + th_s))
        cs = 2.0 / ((dy_act**2) * th_s * (th_n + th_s))
        cp = -(ce + cw + cn + cs)

        sw_rows.append(p_idx)
        sw_cols.append(p_idx)
        sw_vals.append(cp)

        for nb_solid, nb_idx, coeff in (
            (solid[j, i + 1], int(idx_grid[j, i + 1]), ce),
            (solid[j, i - 1], int(idx_grid[j, i - 1]), cw),
            (solid[j + 1, i], int(idx_grid[j + 1, i]), cn),
            (solid[j - 1, i], int(idx_grid[j - 1, i]), cs),
        ):
            if not nb_solid:
                sw_rows.append(p_idx)
                sw_cols.append(nb_idx)
                sw_vals.append(coeff)
            else:
                wall_rhs1[j, i] -= coeff

    if sw_rows:
        rows.append(np.array(sw_rows, dtype=np.int32))
        cols.append(np.array(sw_cols, dtype=np.int32))
        vals.append(np.array(sw_vals, dtype=np.float64))

    A = sp.csc_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(N_nodes, N_nodes),
    )
    lu = spla.factorized(A)
    t_lu = time.perf_counter() - t0

    return {
        "x": x,
        "y": y,
        "X": X,
        "Y": Y,
        "dx": dx_act,
        "dy": dy_act,
        "Xdiv": Xdiv,
        "Ydiv": Ydiv,
        "solid": solid,
        "is_outer": is_outer,
        "wall_rhs1": wall_rhs1,
        "wall_fracs": wall_fracs,
        "lu": lu,
        "t_assemble_lu": t_lu,
    }


def solve_laplace_sparse(
    mesh_sys: Dict[str, object],
    x_cw: np.ndarray,
    y_cw: np.ndarray,
    Uinf: float = 60.0,
    Gamma: float = 0.0,
    C_override: Optional[float] = None,
    outer_psi: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Solve linear superposition psi = psi_0 + C_body * psi_1 using cached sparse LU.
    When C_override is None, calibrates C_body from the trailing-edge Kutta probe.
    """
    t0 = time.perf_counter()
    X = mesh_sys["X"]
    Y = mesh_sys["Y"]
    x = mesh_sys["x"]
    y = mesh_sys["y"]
    dx = mesh_sys["dx"]
    solid = mesh_sys["solid"]
    is_outer = mesh_sys["is_outer"]
    wall_rhs1 = mesh_sys["wall_rhs1"]
    lu = mesh_sys["lu"]

    if outer_psi is not None:
        psi_far = outer_psi
    else:
        r = np.hypot(X, Y)
        psi_far = Uinf * Y + (Gamma / (2.0 * np.pi)) * np.log(np.maximum(r, 1e-6))

    rhs0 = np.zeros_like(X)
    rhs0[is_outer] = psi_far[is_outer]
    psi0 = lu(rhs0.ravel()).reshape(X.shape)

    rhs1 = wall_rhs1.copy()
    rhs1[solid] = 1.0
    psi1 = lu(rhs1.ravel()).reshape(X.shape)

    if C_override is None:
        interp0 = RegularGridInterpolator((y, x), psi0, bounds_error=False, fill_value=0.0)
        interp1 = RegularGridInterpolator((y, x), psi1, bounds_error=False, fill_value=0.0)
        probe_x = float(np.clip(x_cw[0] + 1.5 * dx, x[1], x[-2]))
        probe_y = float(np.clip(y_cw[0], y[1], y[-2]))
        p0_te = float(interp0(np.array([[probe_y, probe_x]]))[0])
        p1_te = float(interp1(np.array([[probe_y, probe_x]]))[0])
        C_body = p0_te / max(1.0 - p1_te, 1e-6)
    else:
        C_body = float(C_override)

    psi = psi0 + C_body * psi1
    t_solve = time.perf_counter() - t0
    return {
        "psi": psi,
        "psi0": psi0,
        "psi1": psi1,
        "C_body": C_body,
        "t_solve": t_solve,
    }


def calculate_velocity_field(
    psi: np.ndarray, C_body: float, mesh_sys: Dict[str, object]
) -> Tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    """
    2nd-order velocity field u = dpsi/dy, v = -dpsi/dx using exact wall value C_body
    at sub-cell distances theta*dx on Shortley-Weller cut cells.
    """
    dx = mesh_sys["dx"]
    dy = mesh_sys["dy"]
    solid = mesh_sys["solid"]
    wall_fracs = mesh_sys["wall_fracs"]

    u = np.zeros_like(psi)
    v = np.zeros_like(psi)

    u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * dy)
    v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2.0 * dx)

    for (j, i), (th_e, th_w, th_n, th_s) in wall_fracs.items():
        psi_p = psi[j, i]
        psi_e = C_body if solid[j, i + 1] else psi[j, i + 1]
        psi_w = C_body if solid[j, i - 1] else psi[j, i - 1]
        psi_n = C_body if solid[j + 1, i] else psi[j + 1, i]
        psi_s = C_body if solid[j - 1, i] else psi[j - 1, i]

        dpsi_dx = ((th_w / th_e) * (psi_e - psi_p) + (th_e / th_w) * (psi_p - psi_w)) / (
            (th_e + th_w) * dx
        )
        dpsi_dy = ((th_s / th_n) * (psi_n - psi_p) + (th_n / th_s) * (psi_p - psi_s)) / (
            (th_n + th_s) * dy
        )
        u[j, i] = dpsi_dy
        v[j, i] = -dpsi_dx

    # Fill outer edges by one-sided copy so visualization boundaries stay clean
    u[:, 0] = u[:, 1]
    u[:, -1] = u[:, -2]
    u[0, :] = u[1, :]
    u[-1, :] = u[-2, :]
    v[:, 0] = v[:, 1]
    v[:, -1] = v[:, -2]
    v[0, :] = v[1, :]
    v[-1, :] = v[-2, :]

    vmag = np.ma.masked_where(solid, np.hypot(u, v))
    return u, v, vmag


def extract_fdm_surface_cp(
    psi: np.ndarray,
    C_body: float,
    mesh_sys: Dict[str, object],
    hs_res: Dict[str, object],
    Uinf: float,
    a_sound: float = 340.3,
) -> Dict[str, np.ndarray | float]:
    """
    2nd-order wall-normal Taylor extraction of surface tangential speed from FDM grid:
      Vt = dpsi/dn |_wall ~= (4*f(d1) - f(d2)) / (2*d1),  d1 = 1.1*dx, d2 = 2.2*dx.
    """
    x = mesh_sys["x"]
    y = mesh_sys["y"]
    dx = mesh_sys["dx"]
    xm = hs_res["xm"]
    ym = hs_res["ym"]
    nx = hs_res["nx"]
    ny = hs_res["ny"]

    interp = RegularGridInterpolator((y, x), psi, bounds_error=False, fill_value=C_body)
    d1, d2 = 1.1 * dx, 2.2 * dx
    f1 = interp(np.column_stack([ym + d1 * ny, xm + d1 * nx])) - C_body
    f2 = interp(np.column_stack([ym + d2 * ny, xm + d2 * nx])) - C_body
    vw = (4.0 * f1 - f2) / (2.0 * d1)

    cp0_fdm = 1.0 - (vw / max(Uinf, 1e-6)) ** 2
    Mach = Uinf / max(a_sound, 1e-6)
    beta_pg = float(np.sqrt(max(1e-4, 1.0 - Mach**2)))
    cp_fdm = cp0_fdm / beta_pg

    # Interior RMS error away from razor-thin TE singularity
    rmse = float(np.sqrt(np.mean((cp_fdm[5:-5] - hs_res["Cp"][5:-5]) ** 2)))
    return {"Cp_fdm": cp_fdm, "Cp0_fdm": cp0_fdm, "rmse_cp": rmse}


def calculate_pressure_field(
    vmag: np.ma.MaskedArray,
    Pinf: float = 101325.0,
    rho: float = 1.225,
    Uinf: float = 60.0,
    a_sound: float = 340.3,
) -> Tuple[np.ma.MaskedArray, np.ma.MaskedArray]:
    """Compute compressible-corrected field Cp and static pressure P (Pa)."""
    qinf = max(0.5 * rho * (Uinf**2), 1e-6)
    Mach = Uinf / max(a_sound, 1e-6)
    beta_pg = float(np.sqrt(max(1e-4, 1.0 - Mach**2)))
    Cp0 = 1.0 - (vmag / max(Uinf, 1e-6)) ** 2
    Cp_field = Cp0 / beta_pg
    P_field = Pinf + qinf * Cp_field
    return P_field, Cp_field


# ---------------------------------------------------------------------------
# 5. Legacy Jacobi Reference Solver (Strictly for V&V Benchmark Panel)
# ---------------------------------------------------------------------------


def run_jacobi_benchmark(
    x_cw: np.ndarray,
    y_cw: np.ndarray,
    Xrange: float = 3.2,
    Yrange: float = 2.2,
    dx: float = 0.04,
    tol: float = 1e-4,
    max_iter: int = 25000,
    pure_python_cap: int = 6000,
) -> Dict[str, float]:
    """Compare vectorized Jacobi and 2nd-year pure-Python loop against Sparse LU."""
    Xdiv = int(round(Xrange / dx)) + 1
    Ydiv = int(round(Yrange / dx)) + 1
    x = np.linspace(-Xrange / 2.0, Xrange / 2.0, Xdiv)
    y = np.linspace(-Yrange / 2.0, Yrange / 2.0, Ydiv)
    X, Y = np.meshgrid(x, y)
    poly = Path(np.column_stack([x_cw, y_cw]))
    solid = poly.contains_points(np.column_stack([X.ravel(), Y.ravel()])).reshape((Ydiv, Xdiv))

    psi_init = 60.0 * Y
    psi_init[solid] = 0.0

    # Vectorized NumPy Jacobi
    t0 = time.perf_counter()
    psi = psi_init.copy()
    mask_fluid = ~solid
    mask_fluid[0, :] = mask_fluid[-1, :] = mask_fluid[:, 0] = mask_fluid[:, -1] = False
    iters = 0
    for it in range(1, max_iter + 1):
        p_new = psi.copy()
        upd = 0.25 * (psi[1:-1, 2:] + psi[1:-1, :-2] + psi[2:, 1:-1] + psi[:-2, 1:-1])
        p_new[1:-1, 1:-1] = np.where(mask_fluid[1:-1, 1:-1], upd, psi[1:-1, 1:-1])
        if it % 20 == 0 or it == 1:
            diff = np.max(np.abs(p_new - psi))
            psi = p_new
            if diff < tol:
                iters = it
                break
        else:
            psi = p_new
    else:
        iters = max_iter
    t_vec_jacobi = time.perf_counter() - t0

    # Pure-Python nested loop benchmark (2nd-year baseline, calibrated or measured)
    t0 = time.perf_counter()
    p_py = psi_init.copy()
    if iters <= pure_python_cap:
        for _ in range(iters):
            p_next = p_py.copy()
            for j in range(1, Ydiv - 1):
                for i in range(1, Xdiv - 1):
                    if not solid[j, i]:
                        p_next[j, i] = 0.25 * (
                            p_py[j, i + 1] + p_py[j, i - 1] + p_py[j + 1, i] + p_py[j - 1, i]
                        )
            p_py = p_next
        t_pure_py = time.perf_counter() - t0
    else:
        # Run 250 iterations to measure exact per-iteration wall time without hanging
        sample_iters = 250
        for _ in range(sample_iters):
            p_next = p_py.copy()
            for j in range(1, Ydiv - 1):
                for i in range(1, Xdiv - 1):
                    if not solid[j, i]:
                        p_next[j, i] = 0.25 * (
                            p_py[j, i + 1] + p_py[j, i - 1] + p_py[j + 1, i] + p_py[j - 1, i]
                        )
            p_py = p_next
        t_pure_py = (time.perf_counter() - t0) * (iters / sample_iters)

    return {
        "dx": dx,
        "nodes": float(Xdiv * Ydiv),
        "jacobi_iters": float(iters),
        "t_vec_jacobi_ms": t_vec_jacobi * 1000.0,
        "t_pure_py_ms": t_pure_py * 1000.0,
    }


# ---------------------------------------------------------------------------
# 6. Dark Aerospace Telemetry Dashboard & Verification Plots
# ---------------------------------------------------------------------------

DARK_BG = "#0c0f14"
PANEL_BG = "#141922"
BORDER_CLR = "#283244"
TEXT_MAIN = "#e6edf3"
TEXT_MUTED = "#8b949e"
ACCENT_CYAN = "#38bdf8"
ACCENT_AMBER = "#f59e0b"
ACCENT_GREEN = "#22c55e"
ACCENT_RED = "#f43f5e"


class FluxDashboard:
    """Encapsulates state, solvers, dark dashboard plots, and interactive widgets."""

    def __init__(
        self,
        naca_code: str = "4412",
        alpha_deg: float = 8.0,
        uinf: float = 60.0,
        pinf_kpa: float = 101.325,
        dx: float = 0.015,
        interactive: bool = True,
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.widgets import RadioButtons, Slider

        self.plt = plt
        self.interactive = interactive
        self.Xrange = 3.2
        self.Yrange = 2.2
        self.rho = 1.225
        self.a_sound = 340.3
        self.nu = 1.48e-5

        self.model_mode = f"NACA {naca_code}"
        self.naca_code = naca_code
        self.alpha_deg = float(alpha_deg)
        self.uinf = float(uinf)
        self.pinf = float(pinf_kpa) * 1000.0
        self.dx = float(dx)
        self.gamma_scale = 1.0  # 1.0 = exact Kutta Gamma; adjustable via slider

        self.base_x, self.base_y = generate_naca4(self.naca_code, n_half=80)
        self.is_cylinder = False
        self.custom_label = None

        self._mesh_cache_key: Optional[Tuple[str, float, float]] = None
        self._mesh_sys: Optional[Dict[str, object]] = None
        self._last_metrics: Dict[str, float] = {}

        plt.rcParams.update(
            {
                "figure.facecolor": DARK_BG,
                "axes.facecolor": PANEL_BG,
                "axes.edgecolor": BORDER_CLR,
                "axes.labelcolor": TEXT_MAIN,
                "xtick.color": TEXT_MUTED,
                "ytick.color": TEXT_MUTED,
                "text.color": TEXT_MAIN,
                "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
                "font.size": 9,
            }
        )

        self.fig = plt.figure(figsize=(16.0, 9.2), dpi=110)
        self.fig.patch.set_facecolor(DARK_BG)
        self.fig.canvas.manager.set_window_title("Flux Visualiser v4.0 // 2D Compressible CFD") if interactive and getattr(self.fig.canvas, "manager", None) else None

        gs = GridSpec(
            2,
            2,
            figure=self.fig,
            left=0.052,
            right=0.688,
            top=0.900,
            bottom=0.075,
            hspace=0.31,
            wspace=0.40,
        )

        self.ax_vel = self.fig.add_subplot(gs[0, 0])
        self.ax_cp_field = self.fig.add_subplot(gs[0, 1])
        self.ax_psi = self.fig.add_subplot(gs[1, 0])
        self.ax_cp_surf = self.fig.add_subplot(gs[1, 1])

        self.cbar_vel = None
        self.cbar_cp = None
        self.cbar_psi = None

        # Top banner title
        self.banner_title = self.fig.text(
            0.055,
            0.955,
            "FLUX VISUALISER v4.0  |  SHORTLEY-WELLER SPARSE LU + HESS-SMITH KUTTA + VISCOUS BL",
            fontsize=12.5,
            fontweight="bold",
            color=TEXT_MAIN,
            family="monospace",
        )
        self.banner_sub = self.fig.text(
            0.055,
            0.930,
            "Prandtl-Glauert Compressibility · Thwaites/Michel/Head/Squire-Young Boundary Layer Coupling",
            fontsize=9.2,
            color=ACCENT_CYAN,
        )

        # Right telemetry & control sidebar frame
        side_ax = self.fig.add_axes([0.745, 0.055, 0.235, 0.870], frameon=True)
        side_ax.set_facecolor("#10141c")
        for spine in side_ax.spines.values():
            spine.set_edgecolor(BORDER_CLR)
            spine.set_linewidth(1.1)
        side_ax.set_xticks([])
        side_ax.set_yticks([])

        self.fig.text(
            0.760,
            0.895,
            "SOLVER CONTROLS",
            fontsize=10,
            fontweight="bold",
            color=ACCENT_CYAN,
            family="monospace",
        )

        ax_alpha = self.fig.add_axes([0.765, 0.835, 0.195, 0.020], facecolor="#1d2430")
        self.sl_alpha = Slider(ax_alpha, "", -12.0, 14.0, valinit=self.alpha_deg, valstep=0.5, color=ACCENT_CYAN)
        self.lbl_alpha = self.fig.text(0.765, 0.862, f"Angle of Attack (alpha): {self.alpha_deg:.1f} deg", fontsize=8.5, color=TEXT_MAIN)

        ax_uinf = self.fig.add_axes([0.765, 0.770, 0.195, 0.020], facecolor="#1d2430")
        self.sl_uinf = Slider(ax_uinf, "", 10.0, 220.0, valinit=self.uinf, valstep=5.0, color=ACCENT_CYAN)
        self.lbl_uinf = self.fig.text(0.765, 0.797, f"Freestream U_inf: {self.uinf:.0f} m/s", fontsize=8.5, color=TEXT_MAIN)

        ax_dx = self.fig.add_axes([0.765, 0.705, 0.195, 0.020], facecolor="#1d2430")
        self.sl_dx = Slider(ax_dx, "", 0.010, 0.040, valinit=self.dx, valstep=0.002, color=ACCENT_GREEN)
        self.lbl_dx = self.fig.text(0.765, 0.732, f"Grid Spacing dx: {self.dx:.3f} c", fontsize=8.5, color=TEXT_MAIN)

        ax_gamma = self.fig.add_axes([0.765, 0.640, 0.195, 0.020], facecolor="#1d2430")
        self.sl_gamma = Slider(ax_gamma, "", 0.0, 1.5, valinit=self.gamma_scale, valstep=0.05, color=ACCENT_AMBER)
        self.lbl_gamma = self.fig.text(0.765, 0.667, f"Kutta Circulation Ratio: {self.gamma_scale:.2f}x", fontsize=8.5, color=TEXT_MAIN)

        ax_pinf = self.fig.add_axes([0.765, 0.575, 0.195, 0.020], facecolor="#1d2430")
        self.sl_pinf = Slider(ax_pinf, "", 50.0, 150.0, valinit=self.pinf / 1000.0, valstep=1.0, color="#64748b")
        self.lbl_pinf = self.fig.text(0.765, 0.602, f"Static Pressure P_inf: {self.pinf / 1000.0:.1f} kPa", fontsize=8.5, color=TEXT_MAIN)

        for sl in (self.sl_alpha, self.sl_uinf, self.sl_dx, self.sl_gamma, self.sl_pinf):
            sl.valtext.set_visible(False)

        # Geometry radio buttons
        self.fig.text(
            0.760,
            0.530,
            "GEOMETRY PROFILE",
            fontsize=9.5,
            fontweight="bold",
            color=ACCENT_CYAN,
            family="monospace",
        )
        ax_radio = self.fig.add_axes([0.760, 0.355, 0.205, 0.165], facecolor="#141922")
        for s in ax_radio.spines.values():
            s.set_edgecolor(BORDER_CLR)
        self.radio = RadioButtons(
            ax_radio,
            ("NACA 4412", "NACA 2412", "NACA 0012", "Cylinder", "Load .txt/.png"),
            active=0,
            activecolor=ACCENT_CYAN,
        )
        for lbl in self.radio.labels:
            lbl.set_color(TEXT_MAIN)
            lbl.set_fontsize(8.5)

        # Aero Telemetry Panel
        self.fig.text(
            0.760,
            0.320,
            "AERODYNAMIC TELEMETRY",
            fontsize=9.5,
            fontweight="bold",
            color=ACCENT_GREEN,
            family="monospace",
        )
        ax_telem = self.fig.add_axes([0.760, 0.070, 0.205, 0.240], frameon=True)
        ax_telem.set_facecolor("#0b0e14")
        for s in ax_telem.spines.values():
            s.set_edgecolor(BORDER_CLR)
        ax_telem.set_xticks([])
        ax_telem.set_yticks([])
        self.telem_text = ax_telem.text(
            0.05,
            0.94,
            "",
            transform=ax_telem.transAxes,
            va="top",
            ha="left",
            family="monospace",
            fontsize=8.3,
            linespacing=1.42,
            color=TEXT_MAIN,
        )

        # Connect callbacks
        self.sl_alpha.on_changed(self._on_slider_change)
        self.sl_uinf.on_changed(self._on_slider_change)
        self.sl_dx.on_changed(self._on_slider_change)
        self.sl_gamma.on_changed(self._on_slider_change)
        self.sl_pinf.on_changed(self._on_slider_change)
        self.radio.on_clicked(self._on_radio_clicked)

        self.solve_and_render()

    def _on_slider_change(self, _val: float) -> None:
        self.alpha_deg = float(self.sl_alpha.val)
        self.uinf = float(self.sl_uinf.val)
        self.dx = float(self.sl_dx.val)
        self.gamma_scale = float(self.sl_gamma.val)
        self.pinf = float(self.sl_pinf.val) * 1000.0

        self.lbl_alpha.set_text(f"Angle of Attack (alpha): {self.alpha_deg:.1f} deg")
        self.lbl_uinf.set_text(f"Freestream U_inf: {self.uinf:.0f} m/s")
        self.lbl_dx.set_text(f"Grid Spacing dx: {self.dx:.3f} c")
        self.lbl_gamma.set_text(f"Kutta Circulation Ratio: {self.gamma_scale:.2f}x")
        self.lbl_pinf.set_text(f"Static Pressure P_inf: {self.pinf / 1000.0:.1f} kPa")
        self.solve_and_render()

    def _on_radio_clicked(self, label: str) -> None:
        if label.startswith("NACA "):
            self.naca_code = label.split()[1]
            self.base_x, self.base_y = generate_naca4(self.naca_code, n_half=80)
            self.is_cylinder = False
            self.model_mode = label
        elif label == "Cylinder":
            self.base_x, self.base_y = generate_cylinder(radius=0.35, n_pts=360)
            self.is_cylinder = True
            self.model_mode = "Circular Cylinder"
        else:
            # Lazy import tkinter only when interactively requested
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            filepath = filedialog.askopenfilename(
                title="Select airfoil .txt or mask .png",
                filetypes=[("Airfoil / Image", "*.txt *.png *.jpg"), ("All files", "*.*")],
            )
            root.destroy()
            if not filepath:
                return
            if filepath.lower().endswith(".txt"):
                self.base_x, self.base_y = process_txt_file(filepath)
            else:
                self.base_x, self.base_y = process_png_image(filepath)
            self.is_cylinder = False
            self.model_mode = os.path.basename(filepath)

        self._mesh_cache_key = None
        self.solve_and_render()

    def solve_and_render(self) -> Dict[str, float]:
        # Rotate body by -alpha (for cylinder rotation leaves circle invariant)
        alpha_eff = 0.0 if self.is_cylinder else self.alpha_deg
        x_cw, y_cw = rotate_coords(self.base_x, self.base_y, alpha_eff)

        # Hess-Smith Panel Solution (with Kutta for lifting bodies, symmetric for cylinder)
        hs_res = solve_hess_smith_panel(
            x_cw,
            y_cw,
            Uinf=self.uinf,
            alpha_deg=alpha_eff,
            a_sound=self.a_sound,
            enforce_kutta=(not self.is_cylinder),
        )
        Gamma_kutta = float(hs_res["Gamma"]) * self.gamma_scale
        if self.is_cylinder and abs(self.gamma_scale - 1.0) > 1e-3:
            # Allow Magnus-effect circulation exploration on cylinder using Gamma slider
            Gamma_kutta = (self.gamma_scale - 1.0) * 2.0 * self.uinf * 0.35

        # Viscous boundary layer coupling
        bl_res = solve_boundary_layer_viscous(
            hs_res["xm"],
            hs_res["ym"],
            hs_res["xc_body"],
            hs_res["L"],
            hs_res["Vt"],
            Uinf=self.uinf,
            nu=self.nu,
        )

        # Assemble / reuse Sparse LU Shortley-Weller operator
        mesh_key = (self.model_mode, round(alpha_eff, 3), round(self.dx, 4))
        if self._mesh_cache_key != mesh_key or self._mesh_sys is None:
            r_cyl = 0.35 if self.is_cylinder else None
            self._mesh_sys = assemble_laplace_sparse(
                x_cw, y_cw, Xrange=self.Xrange, Yrange=self.Yrange, dx=self.dx, on_wall_geom_radius=r_cyl
            )
            self._mesh_cache_key = mesh_key

        fdm_res = solve_laplace_sparse(
            self._mesh_sys,
            x_cw,
            y_cw,
            Uinf=self.uinf,
            Gamma=Gamma_kutta,
            C_override=(0.0 if self.is_cylinder else None),
        )
        psi = fdm_res["psi"]
        C_body = float(fdm_res["C_body"])

        u, v, vmag = calculate_velocity_field(psi, C_body, self._mesh_sys)
        P_field, Cp_field = calculate_pressure_field(vmag, Pinf=self.pinf, rho=self.rho, Uinf=self.uinf, a_sound=self.a_sound)
        surf_fdm = extract_fdm_surface_cp(psi, C_body, self._mesh_sys, hs_res, Uinf=self.uinf, a_sound=self.a_sound)

        X = self._mesh_sys["X"]
        Y = self._mesh_sys["Y"]
        solid = self._mesh_sys["solid"]

        # ---- Plot 1: Velocity Magnitude |V|/Uinf ----
        self.ax_vel.clear()
        v_norm = np.ma.masked_where(solid, vmag / max(self.uinf, 1e-6))
        vmax_v = max(1.6, float(np.percentile(v_norm.compressed(), 99.5)) if v_norm.count() else 1.8)
        cf_vel = self.ax_vel.contourf(X, Y, v_norm, levels=np.linspace(0.0, vmax_v, 36), cmap="turbo", extend="max")
        self.ax_vel.fill(x_cw, y_cw, color="#0d1117", zorder=5)
        self.ax_vel.plot(x_cw, y_cw, color="#f8fafc", lw=1.2, zorder=6)
        self.ax_vel.set_title("Normalized Velocity Field  |V| / U_inf", fontsize=10, fontweight="bold", pad=7)
        self.ax_vel.set_xlabel("x / c")
        self.ax_vel.set_ylabel("y / c")
        self.ax_vel.set_aspect("equal")
        if self.cbar_vel is None:
            self.cbar_vel = self.fig.colorbar(cf_vel, ax=self.ax_vel, fraction=0.045, pad=0.03)
            self.cbar_vel.set_label("|V| / U_inf", fontsize=8)
        else:
            self.cbar_vel.update_normal(cf_vel)

        # ---- Plot 2: Field Pressure Coefficient Cp ----
        self.ax_cp_field.clear()
        cp_lim = max(1.2, float(np.percentile(np.abs(Cp_field.compressed()), 99.0)) if Cp_field.count() else 2.0)
        cp_lim = min(cp_lim, 3.2)
        cf_cp = self.ax_cp_field.contourf(
            X, Y, np.clip(Cp_field, -cp_lim, 1.0), levels=np.linspace(-cp_lim, 1.0, 36), cmap="coolwarm_r", extend="both"
        )
        self.ax_cp_field.fill(x_cw, y_cw, color="#0d1117", zorder=5)
        self.ax_cp_field.plot(x_cw, y_cw, color="#f8fafc", lw=1.2, zorder=6)
        self.ax_cp_field.set_title("Pressure Coefficient Field  Cp = (P - P_inf) / q_inf", fontsize=10, fontweight="bold", pad=7)
        self.ax_cp_field.set_xlabel("x / c")
        self.ax_cp_field.set_ylabel("y / c")
        self.ax_cp_field.set_aspect("equal")
        if self.cbar_cp is None:
            self.cbar_cp = self.fig.colorbar(cf_cp, ax=self.ax_cp_field, fraction=0.045, pad=0.03)
            self.cbar_cp.set_label("Cp", fontsize=8)
        else:
            self.cbar_cp.update_normal(cf_cp)

        # ---- Plot 3: Streamlines psi + Dividing Streamline ----
        self.ax_psi.clear()
        psi_masked = np.ma.masked_where(solid, psi)
        cf_psi = self.ax_psi.contourf(X, Y, psi_masked, levels=34, cmap="cividis", alpha=0.82)
        self.ax_psi.contour(X, Y, psi_masked, levels=28, colors="#94a3b8", linewidths=0.6, linestyles="solid", alpha=0.75)
        self.ax_psi.contour(X, Y, psi_masked, levels=[C_body], colors=[ACCENT_CYAN], linewidths=1.9, linestyles="solid", zorder=4)
        self.ax_psi.fill(x_cw, y_cw, color="#0d1117", zorder=5)
        self.ax_psi.plot(x_cw, y_cw, color="#f8fafc", lw=1.2, zorder=6)
        self.ax_psi.set_title("Streamfunction psi(x,y) [Cyan = Body Line]", fontsize=9.8, fontweight="bold", pad=7)
        self.ax_psi.set_xlabel("x / c")
        self.ax_psi.set_ylabel("y / c")
        self.ax_psi.set_aspect("equal")
        if self.cbar_psi is None:
            self.cbar_psi = self.fig.colorbar(cf_psi, ax=self.ax_psi, fraction=0.045, pad=0.03)
            self.cbar_psi.set_label("psi (m^2/s)", fontsize=8)
        else:
            self.cbar_psi.update_normal(cf_psi)

        # ---- Plot 4: Surface -Cp(x/c) Validation & Viscous Transition ----
        self.ax_cp_surf.clear()
        xc = hs_res["xc_body"]
        cp_hs = hs_res["Cp"]
        cp_fdm = surf_fdm["Cp_fdm"]
        # Split upper & lower surfaces at leading-edge minimum x/c
        le_idx = int(np.argmin(xc))
        self.ax_cp_surf.plot(xc[le_idx:], cp_hs[le_idx:], color=ACCENT_CYAN, lw=2.0, label="Hess-Smith Upper (Suction)")
        self.ax_cp_surf.plot(xc[: le_idx + 1], cp_hs[: le_idx + 1], color=ACCENT_AMBER, lw=1.8, label="Hess-Smith Lower (Pressure)")
        self.ax_cp_surf.plot(
            xc[5:-5:3],
            cp_fdm[5:-5:3],
            "o",
            ms=3.2,
            color="#f8fafc",
            alpha=0.82,
            label=f"SW-FDM Probe (RMSE={surf_fdm['rmse_cp']:.3f})",
        )

        if not self.is_cylinder:
            xtr_u = float(bl_res["x_tr_upper"])
            xtr_l = float(bl_res["x_tr_lower"])
            self.ax_cp_surf.axvline(xtr_u, color=ACCENT_GREEN, ls="--", lw=1.3, label=f"x_tr/c Up={xtr_u:.2f}, Lo={xtr_l:.2f}")

        self.ax_cp_surf.axhline(0.0, color=BORDER_CLR, lw=0.9, ls=":")
        self.ax_cp_surf.invert_yaxis()
        self.ax_cp_surf.grid(True, color="#222a38", lw=0.6, alpha=0.8)
        self.ax_cp_surf.set_xlim(-0.02, 1.02)
        self.ax_cp_surf.set_title("Surface Pressure -Cp(x/c): Hess-Smith vs Shortley-Weller FDM", fontsize=9.8, fontweight="bold", pad=7)
        self.ax_cp_surf.set_xlabel("Chordwise Position x / c")
        self.ax_cp_surf.set_ylabel("Pressure Coefficient Cp (inverted)")
        self.ax_cp_surf.legend(loc="best", fontsize=7.6, frameon=True, facecolor="#0b0e14", edgecolor=BORDER_CLR)

        # Update Telemetry Sidebar
        Cl_kj = float(hs_res["Cl_kj"]) * (1.0 if self.is_cylinder else self.gamma_scale)
        Cd_v = float(bl_res["cd_visc"])
        Cm_qc = float(hs_res["Cm_qc"])
        ld_ratio = Cl_kj / max(Cd_v, 1e-6)
        t_lu_ms = float(self._mesh_sys["t_assemble_lu"]) * 1000.0
        t_slv_ms = float(fdm_res["t_solve"]) * 1000.0
        n_nodes = self._mesh_sys["Xdiv"] * self._mesh_sys["Ydiv"]

        telem_lines = [
            f"Profile   : {self.model_mode}",
            f"alpha     : {alpha_eff:6.2f} deg",
            f"Mach (PG) : {hs_res['Mach']:6.3f} (b={hs_res['beta_pg']:.3f})",
            f"Re_c      : {bl_res['Re_c']:.2e}",
            f"Cl (KJ)   : {Cl_kj:7.4f}",
            f"Cd (visc) : {Cd_v:7.5f}",
            f"Cm (c/4)  : {Cm_qc:7.4f}",
            f"L / D     : {ld_ratio:6.1f}",
            f"x_tr Up/Lo: {bl_res['x_tr_upper']:.2f} / {bl_res['x_tr_lower']:.2f}",
            f"Mesh      : {self._mesh_sys['Xdiv']}x{self._mesh_sys['Ydiv']} ({n_nodes:,})",
            f"LU + Slv  : {t_lu_ms:.1f} + {t_slv_ms:.1f} ms",
        ]
        self.telem_text.set_text("\n".join(telem_lines))

        self._last_metrics = {
            "Cl_kj": Cl_kj,
            "Cl_p": float(hs_res["Cl_p"]),
            "Cd_visc": Cd_v,
            "Cm_qc": Cm_qc,
            "L_D": ld_ratio,
            "x_tr_upper": float(bl_res["x_tr_upper"]),
            "x_tr_lower": float(bl_res["x_tr_lower"]),
            "rmse_cp_fdm": float(surf_fdm["rmse_cp"]),
            "Mach": float(hs_res["Mach"]),
            "Re_c": float(bl_res["Re_c"]),
            "t_lu_ms": t_lu_ms,
            "t_slv_ms": t_slv_ms,
            "nodes": float(n_nodes),
        }

        if self.interactive:
            self.fig.canvas.draw_idle()
        return self._last_metrics


# ---------------------------------------------------------------------------
# 7. Headless Verification & Validation (V&V) Export Suite
# ---------------------------------------------------------------------------


def export_validation_suite(out_dir: str, dx_hero: float = 0.012) -> Dict[str, object]:
    """
    Generate publication-grade 4th-year figures without opening GUI windows:
      1. flux-visualiser-naca4412.png  (Hero 4-panel CFD dashboard at alpha=8 deg)
      2. flux-visualiser-validation.png (2x2 V&V quantitative verification suite)
    """
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    hero_path = os.path.join(out_dir, "flux-visualiser-naca4412.png")
    val_path = os.path.join(out_dir, "flux-visualiser-validation.png")

    # 1. Render Hero Dashboard on NACA 4412 at alpha = 8.0 deg, Uinf = 60 m/s
    dash = FluxDashboard(
        naca_code="4412",
        alpha_deg=8.0,
        uinf=60.0,
        pinf_kpa=101.325,
        dx=dx_hero,
        interactive=False,
    )
    dash.fig.savefig(hero_path, dpi=180, facecolor=DARK_BG)
    hero_metrics = dict(dash._last_metrics)
    plt.close(dash.fig)

    # 2. Panel (a) Data: Shortley-Weller Spatial Grid Convergence on Exact Cylinder Potential Flow
    R_cyl = 0.35
    U_cyl, G_cyl = 60.0, 15.0
    xc_cyl, yc_cyl = generate_cylinder(radius=R_cyl, n_pts=1440)
    dx_list = np.array([0.08, 0.04, 0.02, 0.01])
    l2_errs, linf_errs, lu_times = [], [], []
    for d in dx_list:
        ms = assemble_laplace_sparse(xc_cyl, yc_cyl, Xrange=2.4, Yrange=2.4, dx=float(d), on_wall_geom_radius=R_cyl)
        X, Y = ms["X"], ms["Y"]
        r = np.hypot(X, Y)
        psi_ex = U_cyl * (Y - (R_cyl**2) * Y / np.maximum(r**2, 1e-12)) + (G_cyl / (2.0 * np.pi)) * np.log(
            np.maximum(r / R_cyl, 1e-12)
        )
        res = solve_laplace_sparse(ms, xc_cyl, yc_cyl, Uinf=U_cyl, Gamma=G_cyl, C_override=0.0, outer_psi=psi_ex)
        mask = (~ms["solid"]) & (~ms["is_outer"])
        err = np.abs(res["psi"][mask] - psi_ex[mask]) / max(U_cyl * R_cyl, 1.0)
        l2_errs.append(float(np.sqrt(np.mean(err**2))))
        linf_errs.append(float(np.max(err)))
        lu_times.append((float(ms["t_assemble_lu"]) + float(res["t_solve"])) * 1000.0)

    orders_l2 = [np.log(l2_errs[k] / l2_errs[k + 1]) / np.log(dx_list[k] / dx_list[k + 1]) for k in range(len(dx_list) - 1)]
    mean_order = float(np.mean(orders_l2))

    # 3. Panel (b) Data: Sparse LU vs Vectorized Jacobi vs 2nd-Year Pure Python Jacobi
    x_naca, y_naca = rotate_coords(*generate_naca4("4412", 80), 8.0)
    bench_dxs = [0.08, 0.04, 0.02]
    lu_bench_ms, vjac_bench_ms, py_bench_ms = [], [], []
    for bd in bench_dxs:
        ms_b = assemble_laplace_sparse(x_naca, y_naca, Xrange=3.2, Yrange=2.2, dx=bd)
        r_b = solve_laplace_sparse(ms_b, x_naca, y_naca, Uinf=60.0, Gamma=25.0)
        lu_bench_ms.append((float(ms_b["t_assemble_lu"]) + float(r_b["t_solve"])) * 1000.0)
        jb = run_jacobi_benchmark(x_naca, y_naca, Xrange=3.2, Yrange=2.2, dx=bd)
        vjac_bench_ms.append(jb["t_vec_jacobi_ms"])
        py_bench_ms.append(jb["t_pure_py_ms"])

    # 4. Panel (c) Data: Surface -Cp(x/c) Cross-Validation at dx=0.010, alpha=8 deg
    hs_ref = solve_hess_smith_panel(x_naca, y_naca, Uinf=60.0, alpha_deg=8.0)
    ms_fine = assemble_laplace_sparse(x_naca, y_naca, Xrange=3.2, Yrange=2.2, dx=0.010)
    fdm_fine = solve_laplace_sparse(ms_fine, x_naca, y_naca, Uinf=60.0, Gamma=float(hs_ref["Gamma"]))
    surf_fine = extract_fdm_surface_cp(fdm_fine["psi"], float(fdm_fine["C_body"]), ms_fine, hs_ref, Uinf=60.0)

    # 5. Panel (d) Data: Viscous Polar & Transition Envelope NACA 4412 alpha in [-4, 12] deg
    alphas = np.linspace(-4.0, 12.0, 17)
    alpha_l0 = thin_airfoil_zero_lift_rad("4412")
    M_60 = 60.0 / 340.3
    beta_pg = np.sqrt(1.0 - M_60**2)
    cl_thin = (2.0 * np.pi * np.sin(np.radians(alphas) - alpha_l0)) / beta_pg
    cl_hs_arr, cd_visc_arr, xtr_u_arr = [], [], []
    bx0, by0 = generate_naca4("4412", 80)
    for a in alphas:
        xr, yr = rotate_coords(bx0, by0, float(a))
        hs_a = solve_hess_smith_panel(xr, yr, Uinf=60.0, alpha_deg=float(a))
        bl_a = solve_boundary_layer_viscous(hs_a["xm"], hs_a["ym"], hs_a["xc_body"], hs_a["L"], hs_a["Vt"], Uinf=60.0)
        cl_hs_arr.append(float(hs_a["Cl_kj"]))
        cd_visc_arr.append(float(bl_a["cd_visc"]))
        xtr_u_arr.append(float(bl_a["x_tr_upper"]))

    # ---- Render 2x2 Validation Suite Figure ----
    plt.rcParams.update(
        {
            "figure.facecolor": DARK_BG,
            "axes.facecolor": PANEL_BG,
            "axes.edgecolor": BORDER_CLR,
            "axes.labelcolor": TEXT_MAIN,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "text.color": TEXT_MAIN,
        }
    )
    vfig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), dpi=180)
    vfig.patch.set_facecolor(DARK_BG)
    vfig.suptitle(
        "VERIFICATION & VALIDATION SUITE  //  2D CFD SOLVER BENCHMARKS (NACA 4412, Re = 4.05e6, M = 0.176)",
        fontsize=12.5,
        fontweight="bold",
        color=TEXT_MAIN,
        family="monospace",
        y=0.965,
    )

    # (a) Grid convergence
    ax0 = axes[0, 0]
    ax0.loglog(dx_list, l2_errs, "-o", color=ACCENT_CYAN, lw=2.0, ms=6, label=f"Shortley-Weller L2 (order p = {mean_order:.2f})")
    ax0.loglog(dx_list, linf_errs, "-s", color=ACCENT_AMBER, lw=1.8, ms=5.5, label="Shortley-Weller Linf")
    ref_line = l2_errs[0] * (dx_list / dx_list[0]) ** 2
    ax0.loglog(dx_list, ref_line, "--", color="#94a3b8", lw=1.4, label="Theoretical 2nd-Order Slope O(dx^2)")
    ax0.invert_xaxis()
    ax0.grid(True, which="both", color="#222a38", lw=0.6)
    ax0.set_title("(a) Exact Analytical Verification: Potential Cylinder", fontsize=10, fontweight="bold")
    ax0.set_xlabel("Grid Spacing dx (decreasing -> finer)")
    ax0.set_ylabel("Normalized Relative Error ||e|| / (U_inf R)")
    ax0.legend(loc="lower left", fontsize=8, frameon=True, facecolor="#0b0e14", edgecolor=BORDER_CLR)

    # (b) Solver runtime benchmark
    ax1 = axes[0, 1]
    x_pos = np.arange(len(bench_dxs))
    w = 0.25
    b1 = ax1.bar(x_pos - w, py_bench_ms, width=w, color=ACCENT_RED, label="2nd-Yr Pure-Python Jacobi")
    b2 = ax1.bar(x_pos, vjac_bench_ms, width=w, color=ACCENT_AMBER, label="Vectorized NumPy Jacobi")
    b3 = ax1.bar(x_pos + w, lu_bench_ms, width=w, color=ACCENT_GREEN, label="4th-Yr Sparse LU (SuperLU)")
    ax1.set_yscale("log")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"dx={d:.2f}\n({int(round(3.2/d)+1)}x{int(round(2.2/d)+1)})" for d in bench_dxs])
    ax1.grid(True, axis="y", which="both", color="#222a38", lw=0.6)
    ax1.set_title("(b) Linear Solver Speedup (Wall Time in ms, Log Scale)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Elapsed Runtime (ms)")
    speedup_04 = py_bench_ms[1] / max(lu_bench_ms[1], 1e-6)
    ax1.annotate(
        f"{speedup_04:,.0f}x faster\n({lu_bench_ms[1]:.1f} ms vs {py_bench_ms[1]/1000:.1f} s)",
        xy=(1 + w, lu_bench_ms[1]),
        xytext=(0.9, lu_bench_ms[1] * 8.0),
        arrowprops=dict(arrowstyle="->", color=ACCENT_GREEN, lw=1.2),
        fontsize=8,
        color=ACCENT_GREEN,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#0b0e14", edgecolor=ACCENT_GREEN, lw=0.8),
    )
    ax1.legend(loc="upper left", fontsize=8, frameon=True, facecolor="#0b0e14", edgecolor=BORDER_CLR)

    # (c) Cross-solver surface Cp comparison
    ax2 = axes[1, 0]
    xc_ref = hs_ref["xc_body"]
    le_i = int(np.argmin(xc_ref))
    ax2.plot(xc_ref[le_i:], hs_ref["Cp"][le_i:], color=ACCENT_CYAN, lw=2.1, label="Hess-Smith Upper (160 panels)")
    ax2.plot(xc_ref[: le_i + 1], hs_ref["Cp"][: le_i + 1], color=ACCENT_AMBER, lw=1.9, label="Hess-Smith Lower")
    ax2.plot(
        xc_ref[4:-4:2],
        surf_fine["Cp_fdm"][4:-4:2],
        "o",
        ms=3.3,
        color="#f8fafc",
        alpha=0.85,
        label=f"SW-FDM Taylor Probe dx=0.01 (RMSE={surf_fine['rmse_cp']:.3f})",
    )
    ax2.invert_yaxis()
    ax2.grid(True, color="#222a38", lw=0.6)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_title("(c) Cross-Method Surface -Cp(x/c) Validation (NACA 4412, alpha=8 deg)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Chordwise Station x / c")
    ax2.set_ylabel("Pressure Coefficient Cp (inverted)")
    ax2.legend(loc="lower right", fontsize=8, frameon=True, facecolor="#0b0e14", edgecolor=BORDER_CLR)

    # (d) Aerodynamic & Viscous BL Polars
    ax3 = axes[1, 1]
    l1 = ax3.plot(alphas, cl_thin, "--", color="#94a3b8", lw=1.5, label="Glauert Thin Airfoil Cl (PG-corrected)")
    l2 = ax3.plot(alphas, cl_hs_arr, "-o", color=ACCENT_CYAN, lw=2.0, ms=4.5, label="Hess-Smith Coupled Cl")
    ax3.set_xlabel("Angle of Attack alpha (deg)")
    ax3.set_ylabel("Lift Coefficient Cl", color=ACCENT_CYAN)
    ax3.grid(True, color="#222a38", lw=0.6)

    ax3_r = ax3.twinx()
    l3 = ax3_r.plot(alphas, np.array(cd_visc_arr) * 1e4, "-s", color=ACCENT_GREEN, lw=1.8, ms=4.2, label="Viscous Profile Drag Cd x 10^4 (Squire-Young)")
    ax3_r.set_ylabel("Profile Drag Cd (counts = Cd x 10^4)", color=ACCENT_GREEN)
    ax3_r.tick_params(axis="y", colors=ACCENT_GREEN)
    ax3_r.spines["right"].set_edgecolor(BORDER_CLR)

    lines_all = l1 + l2 + l3
    labels_all = [l.get_label() for l in lines_all]
    ax3.set_title("(d) Attached-Flow Lift & Viscous Drag Polar (Re = 4.05e6)", fontsize=10, fontweight="bold")
    ax3.legend(lines_all, labels_all, loc="upper left", fontsize=7.8, frameon=True, facecolor="#0b0e14", edgecolor=BORDER_CLR)

    vfig.subplots_adjust(left=0.065, right=0.93, top=0.90, bottom=0.075, hspace=0.32, wspace=0.28)
    vfig.savefig(val_path, dpi=180, facecolor=DARK_BG)
    plt.close(vfig)

    return {
        "hero_path": hero_path,
        "val_path": val_path,
        "hero_metrics": hero_metrics,
        "mean_sw_order": mean_order,
        "l2_errs": l2_errs,
        "linf_errs": linf_errs,
        "lu_bench_ms": lu_bench_ms,
        "vjac_bench_ms": vjac_bench_ms,
        "py_bench_ms": py_bench_ms,
        "rmse_cp_fine": float(surf_fine["rmse_cp"]),
        "alphas": alphas.tolist(),
        "cl_hs": cl_hs_arr,
        "cd_visc": cd_visc_arr,
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Flux Visualiser v4.0 - 2D CFD & Viscous Airfoil Solver")
    parser.add_argument("--export-figures", action="store_true", help="Headlessly export dashboard & V&V PNGs")
    parser.add_argument("--out-dir", type=str, default=".", help="Output directory for exported PNGs")
    parser.add_argument("--naca", type=str, default="4412", help="4-digit NACA airfoil code")
    parser.add_argument("--alpha", type=float, default=8.0, help="Angle of attack in degrees")
    parser.add_argument("--uinf", type=float, default=60.0, help="Freestream velocity in m/s")
    parser.add_argument("--dx", type=float, default=0.012, help="Initial grid spacing in chords")
    args = parser.parse_args(argv)

    if args.export_figures:
        summary = export_validation_suite(args.out_dir, dx_hero=args.dx)
        print("Exported figures:")
        print(f"  Hero : {summary['hero_path']}")
        print(f"  V&V  : {summary['val_path']}")
        print(f"  Hero Metrics (NACA 4412, a=8 deg): {summary['hero_metrics']}")
        print(f"  Cylinder L2 errors (dx=0.08..0.01): {summary['l2_errs']} | mean order = {summary['mean_sw_order']:.3f}")
        print(f"  Benchmark ms (dx=0.08,0.04,0.02):")
        print(f"    Sparse LU    : {[round(x, 2) for x in summary['lu_bench_ms']]}")
        print(f"    Vec Jacobi   : {[round(x, 2) for x in summary['vjac_bench_ms']]}")
        print(f"    PurePy Jacobi: {[round(x, 2) for x in summary['py_bench_ms']]}")
        print(f"  Fine Cp RMSE (dx=0.010): {summary['rmse_cp_fine']:.4f}")
    else:
        import matplotlib.pyplot as plt

        _dash = FluxDashboard(
            naca_code=args.naca,
            alpha_deg=args.alpha,
            uinf=args.uinf,
            dx=args.dx,
            interactive=True,
        )
        print("Flux Visualiser v4.0 active. Adjust sliders or switch geometry in GUI.")
        plt.show()


if __name__ == "__main__":
    if any(arg == "--export-figures" for arg in sys.argv[1:]):
        import matplotlib

        matplotlib.use("Agg")
    main()
