"""
Utility functions for integrate_all_blocks_streaming_inplace_elec_only
All functions have type hints for inputs and outputs.
"""

import numpy as np
from numba import njit, prange, types
from typing import Tuple, List, Union, Optional
from tqdm.auto import tqdm
from scipy.special import j1

# Type alias for complex arrays
Complex = np.complex128

# ============================================================
# Voigt 6 ↔ 4th-rank tensor converters
# ============================================================

# Standard 6-index Voigt mapping:
# 0↔(0,0), 1↔(1,1), 2↔(2,2), 3↔(1,2), 4↔(0,2), 5↔(0,1)

_VOIGT_PAIRS = [(0,0),(1,1),(2,2),(1,2),(0,2),(0,1)]
_VOIGT_6_LOOKUP = {}
for a, (i, j) in enumerate(_VOIGT_PAIRS):
    _VOIGT_6_LOOKUP[(i, j)] = a
    _VOIGT_6_LOOKUP[(j, i)] = a  # minor symmetry


def voigt6_to_cijkl(C6: np.ndarray) -> np.ndarray:
    """
    Symmetric map 6×6 Voigt → C_{ijkl}.
    
    Args:
        C6: 6×6 Voigt stiffness matrix
        
    Returns:
        C: 3×3×3×3 stiffness tensor
    """
    C = np.zeros((3,3,3,3), dtype=float)
    for I,(i,j) in enumerate(_VOIGT_PAIRS):
        for J,(k,l) in enumerate(_VOIGT_PAIRS):
            C[i,j,k,l] = C6[I,J]
            C[j,i,k,l] = C6[I,J]
            C[i,j,l,k] = C6[I,J]
            C[j,i,l,k] = C6[I,J]
    return C


def cijkl_to_voigt6(C4: np.ndarray) -> np.ndarray:
    """
    Convert 4th-rank c_{ijkl} to 6×6 Voigt stiffness.
    
    Args:
        C4: 3×3×3×3 stiffness tensor
        
    Returns:
        C6: 6×6 Voigt stiffness matrix
    """
    C4 = np.asarray(C4)
    C6 = np.zeros((6, 6), dtype=C4.dtype)
    for i in range(3):
        for j in range(3):
            a = _VOIGT_6_LOOKUP[(i, j)]
            for k in range(3):
                for l in range(3):
                    b = _VOIGT_6_LOOKUP[(k, l)]
                    C6[a, b] = C4[i, j, k, l]
    return C6


def e6_voigt_to_eijk(e6: np.ndarray) -> np.ndarray:
    """
    Convert 3x6 Voigt-form piezo tensor e6[k, J] to 3×3×3 tensor e_ijk.
    
    Args:
        e6: 3×6 Voigt-form piezoelectric tensor
        
    Returns:
        e_ijk: 3×3×3 piezoelectric tensor
    """
    e_ijk = np.zeros((3, 3, 3), dtype=e6.dtype)
    # Voigt mapping: 0↔(0,0), 1↔(1,1), 2↔(2,2), 3↔(1,2), 4↔(0,2), 5↔(0,1)
    voigt_map = {0: (0,0), 1: (1,1), 2: (2,2), 3: (1,2), 4: (0,2), 5: (0,1)}
    
    for k in range(3):
        for J in range(6):
            i, j = voigt_map[J]
            e_ijk[k, i, j] = e6[k, J]
            if i != j:  # symmetric for stress
                e_ijk[k, j, i] = e6[k, J]
    return e_ijk


# ============================================================
# Tensor rotation functions
# ============================================================

def rotation_matrix_cartesian_x(theta_deg: float) -> np.ndarray:
    """
    Generate rotation matrix for rotation about x-axis.
    
    Args:
        theta_deg: Rotation angle in degrees
        
    Returns:
        R: 3×3 rotation matrix
    """
    theta_rad = np.deg2rad(theta_deg)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ], dtype=Complex)


def rotate_tensor_4_rank(C: np.ndarray, theta_deg: float) -> np.ndarray:
    """
    Rotate 4th-rank tensor by rotation about x-axis.
    
    Args:
        C: 3×3×3×3 tensor
        theta_deg: Rotation angle in degrees
        
    Returns:
        C_rot: Rotated 3×3×3×3 tensor
    """
    R = rotation_matrix_cartesian_x(theta_deg)
    return np.einsum('ip,jq,kr,ls,pqrs', R, R, R, R, C)


def rotate_tensor_3_rank(E: np.ndarray, theta_deg: float) -> np.ndarray:
    """
    Rotate 3rd-rank tensor by rotation about x-axis.
    
    Args:
        E: 3×3×3 tensor
        theta_deg: Rotation angle in degrees
        
    Returns:
        E_rot: Rotated 3×3×3 tensor
    """
    R = rotation_matrix_cartesian_x(theta_deg)
    return np.einsum('ip,jq,kr,pqr', R, R, R, E)


def rotate_tensor_2_rank(Eps: np.ndarray, theta_deg: float) -> np.ndarray:
    """
    Rotate 2nd-rank tensor by rotation about x-axis.
    
    Args:
        Eps: 3×3 tensor
        theta_deg: Rotation angle in degrees
        
    Returns:
        Eps_rot: Rotated 3×3 tensor
    """
    R = rotation_matrix_cartesian_x(theta_deg)
    return np.einsum('ip,jq,pq', R, R, Eps)


# ============================================================
# Helper functions for matrix operations
# ============================================================

@njit
def _zero_nm(A, n, m):
    """Zero out an n×m matrix."""
    for i in range(n):
        for j in range(m):
            A[i, j] = 0.0 + 0.0j


@njit
def _zero_nn(A, n):
    """Zero out an n×n matrix."""
    for i in range(n):
        for j in range(n):
            A[i, j] = 0.0 + 0.0j


@njit
def _eye_n(A, n):
    """Set A to identity matrix of size n×n."""
    _zero_nn(A, n)
    for i in range(n):
        A[i, i] = 1.0 + 0.0j


@njit
def _matmul_nn(A, B, C, n, tmp):
    """Matrix multiply: C = A @ B, all n×n."""
    for i in range(n):
        for j in range(n):
            s = 0.0 + 0.0j
            for k in range(n):
                s += A[i, k] * B[k, j]
            C[i, j] = s


@njit
def _matmul_nm(A, B, C, n, m):
    """Matrix multiply: C = A @ B, where A is n×n, B is n×m, C is n×m."""
    for i in range(n):
        for j in range(m):
            s = 0.0 + 0.0j
            for k in range(n):
                s += A[i, k] * B[k, j]
            C[i, j] = s


@njit
def _matmul_mn(A, B, C, m, n):
    """Matrix multiply: C = A @ B, where A is m×n, B is n×n, C is m×n."""
    for i in range(m):
        for j in range(n):
            s = 0.0 + 0.0j
            for k in range(n):
                s += A[i, k] * B[k, j]
            C[i, j] = s


# ============================================================
# LU solve for n<=4, multiple RHS columns (no alloc)
# ============================================================

@njit
def solve_lu_inplace_n4(A, B, X, n, m, LU, piv):
    """
    Solve A X = B for X, with n<=4.
    Inputs:
      A: (4,4) used only in top-left (n,n)
      B: (4,m) used only in top-left (n,m)
    Outputs:
      X: (4,m) filled in top-left (n,m)
    Scratch:
      LU: (4,4) complex
      piv:(4,) int64
    """
    # copy A -> LU
    for i in range(n):
        piv[i] = i
        for j in range(n):
            LU[i, j] = A[i, j]

    # LU with partial pivoting
    for k in range(n):
        # pivot row
        p = k
        best = np.abs(LU[k, k])
        for i in range(k + 1, n):
            v = np.abs(LU[i, k])
            if v > best:
                best = v
                p = i

        # swap rows if needed
        if p != k:
            for j in range(n):
                tmp = LU[k, j]
                LU[k, j] = LU[p, j]
                LU[p, j] = tmp
            tmpi = piv[k]
            piv[k] = piv[p]
            piv[p] = tmpi

        # eliminate
        diag = LU[k, k]
        for i in range(k + 1, n):
            LU[i, k] = LU[i, k] / diag
            lik = LU[i, k]
            for j in range(k + 1, n):
                LU[i, j] = LU[i, j] - lik * LU[k, j]

    # solve for each RHS column
    for col in range(m):
        # forward solve Ly = Pb
        y0 = 0.0 + 0.0j
        y1 = 0.0 + 0.0j
        y2 = 0.0 + 0.0j
        y3 = 0.0 + 0.0j

        # apply permutation on the fly
        for i in range(n):
            bi = B[piv[i], col]
            if i == 0:
                y0 = bi
            elif i == 1:
                y1 = bi - LU[1, 0] * y0
            elif i == 2:
                y2 = bi - LU[2, 0] * y0 - LU[2, 1] * y1
            else:
                y3 = bi - LU[3, 0] * y0 - LU[3, 1] * y1 - LU[3, 2] * y2

        # back solve Ux = y
        if n == 1:
            X[0, col] = y0 / LU[0, 0]
        elif n == 2:
            x1 = y1 / LU[1, 1]
            x0 = (y0 - LU[0, 1] * x1) / LU[0, 0]
            X[0, col] = x0
            X[1, col] = x1
        elif n == 3:
            x2 = y2 / LU[2, 2]
            x1 = (y1 - LU[1, 2] * x2) / LU[1, 1]
            x0 = (y0 - LU[0, 1] * x1 - LU[0, 2] * x2) / LU[0, 0]
            X[0, col] = x0
            X[1, col] = x1
            X[2, col] = x2
        else:
            x3 = y3 / LU[3, 3]
            x2 = (y2 - LU[2, 3] * x3) / LU[2, 2]
            x1 = (y1 - LU[1, 2] * x2 - LU[1, 3] * x3) / LU[1, 1]
            x0 = (y0 - LU[0, 1] * x1 - LU[0, 2] * x2 - LU[0, 3] * x3) / LU[0, 0]
            X[0, col] = x0
            X[1, col] = x1
            X[2, col] = x2
            X[3, col] = x3


# ============================================================
# Small matrix inversion helpers (Numba-compatible, no LAPACK)
# ============================================================

@njit
def inv_3x3(A):
    """Invert 3x3 matrix using direct formula."""
    det = (A[0,0] * (A[1,1] * A[2,2] - A[1,2] * A[2,1]) -
           A[0,1] * (A[1,0] * A[2,2] - A[1,2] * A[2,0]) +
           A[0,2] * (A[1,0] * A[2,1] - A[1,1] * A[2,0]))
    if abs(det) < 1e-20:
        raise ValueError("Matrix is singular")
    inv_det = 1.0 / det
    B = np.zeros((3, 3), dtype=Complex)
    B[0,0] = (A[1,1] * A[2,2] - A[1,2] * A[2,1]) * inv_det
    B[0,1] = (A[0,2] * A[2,1] - A[0,1] * A[2,2]) * inv_det
    B[0,2] = (A[0,1] * A[1,2] - A[0,2] * A[1,1]) * inv_det
    B[1,0] = (A[1,2] * A[2,0] - A[1,0] * A[2,2]) * inv_det
    B[1,1] = (A[0,0] * A[2,2] - A[0,2] * A[2,0]) * inv_det
    B[1,2] = (A[0,2] * A[1,0] - A[0,0] * A[1,2]) * inv_det
    B[2,0] = (A[1,0] * A[2,1] - A[1,1] * A[2,0]) * inv_det
    B[2,1] = (A[0,1] * A[2,0] - A[0,0] * A[2,1]) * inv_det
    B[2,2] = (A[0,0] * A[1,1] - A[0,1] * A[1,0]) * inv_det
    return B


@njit
def inv_4x4_using_lu(A, LU, piv, tmp_B, tmp_X):
    """Invert 4x4 matrix using LU decomposition."""
    I = np.zeros((4, 4), dtype=Complex)
    for i in range(4):
        I[i, i] = 1.0 + 0.0j
    A_inv = np.zeros((4, 4), dtype=Complex)
    for col in range(4):
        for i in range(4):
            tmp_B[i, 0] = I[i, col]
        solve_lu_inplace_n4(A, tmp_B, tmp_X, 4, 1, LU, piv)
        for i in range(4):
            A_inv[i, col] = tmp_X[i, 0]
    return A_inv


# ============================================================
# Stroh generator helper functions
# ============================================================

@njit
def compute_s_and_khat(sx, sy, tol_s):
    """Shared helper: get s_|| and khat for both elastic and piezo generators."""
    s_par = np.hypot(sx, sy)
    khat = np.empty(2, dtype=np.float64)
    if s_par <= tol_s:
        khat[0] = 1.0
        khat[1] = 0.0
        s = tol_s
    else:
        inv_s_par = 1.0 / s_par
        khat[0] = sx * inv_s_par
        khat[1] = sy * inv_s_par
        s = s_par
    return s, khat


@njit
def build_mech_QRT_numba(C4_eff, khat):
    """Numba-safe mechanical projections: Q_il = c_{i3l3}, R_il = c_{i3lα} k̂_α, T_il = c_{iαlβ} k̂_α k̂_β."""
    Q = np.zeros((3, 3), dtype=Complex)
    R = np.zeros((3, 3), dtype=Complex)
    T = np.zeros((3, 3), dtype=Complex)
    iz = 2
    for i in range(3):
        for l in range(3):
            Q[i, l] = C4_eff[i, iz, l, iz]
            R[i, l] = C4_eff[i, iz, l, 0] * khat[0] + C4_eff[i, iz, l, 1] * khat[1]
            T[i, l] = (C4_eff[i, 0, l, 0] * khat[0] * khat[0] +
                       C4_eff[i, 0, l, 1] * khat[0] * khat[1] +
                       C4_eff[i, 1, l, 0] * khat[1] * khat[0] +
                       C4_eff[i, 1, l, 1] * khat[1] * khat[1])
    return Q, R, T


@njit
def build_piezo_projections_numba(e_ijk, eps_kl, khat):
    """Numba-safe version of build_piezo_projections."""
    iz = 2
    alpha_idx0 = 0
    alpha_idx1 = 1
    e_vec = np.zeros(3, dtype=Complex)
    B_vec = np.zeros(3, dtype=Complex)
    W_vec = np.zeros(3, dtype=Complex)
    A_vec = np.zeros(3, dtype=Complex)
    eps_s = eps_kl[iz, iz]
    gamma_s = 0.0 + 0.0j
    alpha_s = 0.0 + 0.0j
    gamma_s += eps_kl[iz, alpha_idx0] * khat[alpha_idx0]
    gamma_s += eps_kl[iz, alpha_idx1] * khat[alpha_idx1]
    alpha_s += eps_kl[alpha_idx0, alpha_idx0] * khat[alpha_idx0] * khat[alpha_idx0]
    alpha_s += eps_kl[alpha_idx0, alpha_idx1] * khat[alpha_idx0] * khat[alpha_idx1]
    alpha_s += eps_kl[alpha_idx1, alpha_idx0] * khat[alpha_idx1] * khat[alpha_idx0]
    alpha_s += eps_kl[alpha_idx1, alpha_idx1] * khat[alpha_idx1] * khat[alpha_idx1]
    for l in range(3):
        e_vec[l] = e_ijk[iz, iz, l]
        tmpB = 0.0 + 0.0j
        tmpW = 0.0 + 0.0j
        tmpA = 0.0 + 0.0j
        a = alpha_idx0
        tmpB += e_ijk[iz, a, l] * khat[a]
        tmpW += e_ijk[a, iz, l] * khat[a]
        tmpA += e_ijk[a, alpha_idx0, l] * khat[a] * khat[alpha_idx0]
        tmpA += e_ijk[a, alpha_idx1, l] * khat[a] * khat[alpha_idx1]
        a = alpha_idx1
        tmpB += e_ijk[iz, a, l] * khat[a]
        tmpW += e_ijk[a, iz, l] * khat[a]
        tmpA += e_ijk[a, alpha_idx0, l] * khat[a] * khat[alpha_idx0]
        tmpA += e_ijk[a, alpha_idx1, l] * khat[a] * khat[alpha_idx1]
        B_vec[l] = tmpB
        W_vec[l] = tmpW
        A_vec[l] = tmpA
    return e_vec, B_vec, W_vec, A_vec, eps_s, gamma_s, alpha_s


@njit
def build_tilded_blocks_numba(Q, R, T, e_vec, B_vec, W_vec, A_vec, eps_s, gamma_s, alpha_s):
    """Numba-safe version of build_tilded_blocks."""
    Qtilde = np.zeros((4, 4), dtype=Complex)
    Rtilde = np.zeros((4, 4), dtype=Complex)
    Ttilde = np.zeros((4, 4), dtype=Complex)
    for i in range(3):
        for j in range(3):
            Qtilde[i, j] = Q[i, j]
        Qtilde[i, 3] = e_vec[i]
        Qtilde[3, i] = e_vec[i]
    Qtilde[3, 3] = -eps_s
    for i in range(3):
        for j in range(3):
            Rtilde[i, j] = R[i, j]
        Rtilde[i, 3] = B_vec[i]
        Rtilde[3, i] = W_vec[i]
    Rtilde[3, 3] = -gamma_s
    for i in range(3):
        for j in range(3):
            Ttilde[i, j] = T[i, j]
        Ttilde[i, 3] = A_vec[i]
        Ttilde[3, i] = A_vec[i]
    Ttilde[3, 3] = -alpha_s
    return Qtilde, Rtilde, Ttilde


@njit
def build_piezo_QRT_tilde_numba(C4_eff, e_ijk, eps_kl, khat):
    """Numba version of build_piezo_QRT_tilde."""
    Q, R, T = build_mech_QRT_numba(C4_eff, khat)
    e_vec, B_vec, W_vec, A_vec, eps_s, gamma_s, alpha_s = build_piezo_projections_numba(e_ijk, eps_kl, khat)
    Qtilde, Rtilde, Ttilde = build_tilded_blocks_numba(Q, R, T, e_vec, B_vec, W_vec, A_vec, eps_s, gamma_s, alpha_s)
    RTtilde = Rtilde.T.copy()
    return Qtilde, Rtilde, RTtilde, Ttilde


@njit
def stroh_generator_elastic_slow_numba(C4_eff, rho, sx, sy, tol_s):
    """Numba-safe 6×6 elastic Stroh generator N."""
    s, khat = compute_s_and_khat(sx, sy, tol_s)
    Q, R, T = build_mech_QRT_numba(C4_eff, khat)
    Qinv = inv_3x3(Q)
    RT = R.T
    rhoM = rho * np.eye(3, dtype=Complex)
    N11 = -1j * s * (Qinv @ RT)
    N12 = -Qinv
    N21 = rhoM + (s * s) * (R @ Qinv @ RT - T)
    N22 = -1j * s * (R @ Qinv)
    N = np.zeros((6, 6), dtype=Complex)
    N[0:3, 0:3] = N11
    N[0:3, 3:6] = N12
    N[3:6, 0:3] = N21
    N[3:6, 3:6] = N22
    return N


@njit
def stroh_generator_piezo_slow_numba(C4_eff, e_ijk, eps_tensor, rho, sx, sy, tol_s):
    """Numba-safe piezoelectric Stroh generator in slowness form."""
    s_par = np.hypot(sx, sy)
    if s_par < tol_s:
        khat = np.empty(2, dtype=np.float64)
        khat[0] = 1.0
        khat[1] = 0.0
        s_eff = tol_s
    else:
        inv_s = 1.0 / s_par
        khat = np.empty(2, dtype=np.float64)
        khat[0] = sx * inv_s
        khat[1] = sy * inv_s
        s_eff = s_par
    Qtil, Rtil, RTtil, Ttil = build_piezo_QRT_tilde_numba(C4_eff, e_ijk, eps_tensor, khat)
    # Use LU-based inversion for 4x4
    LU_work = np.zeros((4, 4), dtype=Complex)
    piv_work = np.zeros(4, dtype=np.int64)
    tmp_B_work = np.zeros((4, 1), dtype=Complex)
    tmp_X_work = np.zeros((4, 1), dtype=Complex)
    Qinv = inv_4x4_using_lu(Qtil, LU_work, piv_work, tmp_B_work, tmp_X_work)
    rhoM = np.zeros((4, 4), dtype=Complex)
    for i in range(3):
        rhoM[i, i] = rho
    s = s_eff
    s2 = s * s
    N11 = -1j * s * (Qinv @ RTtil)
    N12 = -Qinv
    tmp = Rtil @ Qinv @ RTtil
    for i in range(4):
        for j in range(4):
            tmp[i, j] = tmp[i, j] - Ttil[i, j]
    N21 = rhoM.copy()
    for i in range(4):
        for j in range(4):
            N21[i, j] = N21[i, j] + s2 * tmp[i, j]
    N22 = -1j * s * (Rtil @ Qinv)
    N = np.zeros((8, 8), dtype=Complex)
    for i in range(4):
        for j in range(4):
            N[i, j] = N11[i, j]
            N[i, j + 4] = N12[i, j]
            N[i + 4, j] = N21[i, j]
            N[i + 4, j+4] = N22[i, j]
    return N


@njit(error_model='numpy')
def slow_modes_robust_numba(N, tol_im=1e-13, tol_static=-1.0):
    """Numba-safe version of slow_modes_robust. Uses error_model='numpy' to allow LAPACK fallback."""
    vals, vecs = np.linalg.eig(N)
    sz = -1j * vals
    n_total = N.shape[0]
    if n_total % 2 != 0:
        raise ValueError("slow_modes_robust_numba: N must be 2n×2n.")
    n = n_total // 2
    if tol_static <= 0.0:
        tol_static = 10.0 * tol_im
    plus_idx = np.empty(2 * n, dtype=np.int64)
    minus_idx = np.empty(2 * n, dtype=np.int64)
    static_idx = np.empty(2 * n, dtype=np.int64)
    c_plus = 0
    c_minus = 0
    c_static = 0
    for j in range(2 * n):
        sz_j = sz[j]
        abs_k = np.abs(sz_j)
        if abs_k < tol_static:
            static_idx[c_static] = j
            c_static += 1
            continue
        imag_k = np.imag(sz_j)
        real_k = np.real(sz_j)
        s = 0
        if imag_k > tol_im:
            s = +1
        elif imag_k < -tol_im:
            s = -1
        else:
            if np.abs(real_k) < tol_im:
                s = +1
            else:
                if real_k >= 0.0:
                    s = +1
                else:
                    s = -1
        if s > 0:
            plus_idx[c_plus] = j
            c_plus += 1
        else:
            minus_idx[c_minus] = j
            c_minus += 1
    need_plus = n - c_plus
    need_minus = n - c_minus
    if (need_plus < 0) or (need_minus < 0) or (need_plus + need_minus != c_static):
        Imk = np.imag(sz)
        order = np.argsort(-Imk)
        plus_final = order[0:n]
        minus_final = order[2*n - n:2*n]
    else:
        for k in range(need_plus):
            plus_idx[c_plus + k] = static_idx[k]
        for k in range(need_minus):
            minus_idx[c_minus + k] = static_idx[need_plus + k]
        plus_all = plus_idx[0:n]
        minus_all = minus_idx[0:n]
        Imk = np.imag(sz)
        tmp_plus = np.empty(n, dtype=np.int64)
        for i in range(n):
            tmp_plus[i] = plus_all[i]
        order_plus = np.argsort(-Imk[tmp_plus])
        plus_final = np.empty(n, dtype=np.int64)
        for i in range(n):
            plus_final[i] = tmp_plus[order_plus[i]]
        tmp_minus = np.empty(n, dtype=np.int64)
        for i in range(n):
            tmp_minus[i] = minus_all[i]
        order_minus = np.argsort(Imk[tmp_minus])
        minus_final = np.empty(n, dtype=np.int64)
        for i in range(n):
            minus_final[i] = tmp_minus[order_minus[i]]
    Vp = vecs[:, plus_final]
    Vm = vecs[:, minus_final]
    Uplus = Vp[0:n, :]
    Pplus = Vp[n:, :]
    Uminus = Vm[0:n, :]
    Pminus = Vm[n:, :]
    Slplus = np.zeros((n, n), dtype=Complex)
    Slminus = np.zeros((n, n), dtype=Complex)
    for i in range(n):
        Slplus[i, i] = sz[plus_final[i]]
        Slminus[i, i] = sz[minus_final[i]]
    return Uplus, Pplus, Uminus, Pminus, Slplus, Slminus


@njit
def local_reflection_matrix_bc_numba(Uplus, Pplus, Uminus, Pminus, incident_side_flag, mech_flag, elec_flag, Z_elec, omega2, piezo_flag, elec_passthrough_idx):
    """Numba-safe local reflection matrix from boundary conditions."""
    n = Uplus.shape[1]
    n_eq = 3
    if piezo_flag == 1:
        n_eq += 1
    B_a = np.zeros((n_eq, n), dtype=Complex)
    B_b = np.zeros((n_eq, n), dtype=Complex)
    row = 0
    for comp in range(3):
        if mech_flag == 0:
            for j in range(n):
                B_b[row, j] = Pminus[comp, j]
                B_a[row, j] = -Pplus[comp, j]
        else:
            for j in range(n):
                B_b[row, j] = Uminus[comp, j]
                B_a[row, j] = -Uplus[comp, j]
        row += 1
    if piezo_flag == 1:
        if elec_flag == 0:
            for j in range(n):
                B_b[row, j] = Uminus[3, j]
                B_a[row, j] = -Uplus[3, j]
        elif elec_flag == 1:
            for j in range(n):
                B_b[row, j] = Pminus[3, j]
                B_a[row, j] = -Pplus[3, j]
        elif elec_flag == 2:
            for j in range(n):
                B_b[row, j] = Uminus[3, j] + Z_elec * omega2 * Pminus[3, j]
                B_a[row, j] = -(Uplus[3, j] + Z_elec * omega2 * Pplus[3, j])
        elif elec_flag == 3:
            idx = elec_passthrough_idx
            for j in range(n):
                if j == idx:
                    B_b[row, j] = 1.0 + 0.0j
                    B_a[row, j] = 1.0 + 0.0j
                else:
                    B_b[row, j] = 0.0 + 0.0j
                    B_a[row, j] = 0.0 + 0.0j
        row += 1
    if B_b.shape[0] != B_b.shape[1]:
        raise ValueError("BC system not square: check n_modes vs flags")
    # Use LU solver for small systems (n <= 4)
    n_solve = B_b.shape[0]
    if n_solve > 4:
        raise ValueError(f"System size {n_solve} > 4 not supported")
    LU_work = np.zeros((4, 4), dtype=Complex)
    piv_work = np.zeros(4, dtype=np.int64)
    R_work = np.zeros((4, 4), dtype=Complex)
    B_neg = np.zeros((4, 4), dtype=Complex)
    for i in range(n_solve):
        for j in range(n_solve):
            B_neg[i, j] = -B_a[i, j]
    solve_lu_inplace_n4(B_b, B_neg, R_work, n_solve, n_solve, LU_work, piv_work)
    R = np.zeros((n_solve, n_solve), dtype=Complex)
    for i in range(n_solve):
        for j in range(n_solve):
            R[i, j] = R_work[i, j]
    return R


@njit
def local_interface_scattering_general_inplace(U1plus, P1plus, U1minus, P1minus, kind1, U2plus, P2plus, U2minus, P2minus, kind2, pp_elec_mode, elec_flag_1, elec_flag_2, S11_bf, S12_bf, S21_bf, S22_bf):
    """Numba-safe local interface scattering matrix."""
    n1 = 4 if kind1 == 1 else 3
    n2 = 4 if kind2 == 1 else 3
    dim = n1 + n2
    n_eq = 6
    if kind1 == 1 and kind2 == 1:
        n_eq += 2
    elif (kind1 == 1) ^ (kind2 == 1):
        n_eq += 1
    if n_eq != dim:
        raise ValueError("Interface system not square: check mode counts/closure.")
    M_b = np.zeros((dim, dim), dtype=Complex)
    M_a = np.zeros((dim, dim), dtype=Complex)
    for comp in range(3):
        ru = comp
        rt = 3 + comp
        for j in range(n1):
            M_b[ru, j] = U1minus[comp, j]
            M_b[rt, j] = P1minus[comp, j]
            M_a[ru, j] = -U1plus[comp, j]
            M_a[rt, j] = -P1plus[comp, j]
        for j in range(n2):
            col = n1 + j
            M_b[ru, col] = -U2plus[comp, j]
            M_b[rt, col] = -P2plus[comp, j]
            M_a[ru, col] = U2minus[comp, j]
            M_a[rt, col] = P2minus[comp, j]
    row = 6
    if kind1 == 1 and kind2 == 1:
        if pp_elec_mode != 0:
            raise ValueError("For interfaces: pp_elec_mode must be 0 (Phi,q continuity).")
        for j in range(n1):
            M_b[row, j] = U1minus[3, j]
            M_a[row, j] = -U1plus[3, j]
        for j in range(n2):
            col = n1 + j
            M_b[row, col] = -U2plus[3, j]
            M_a[row, col] = U2minus[3, j]
        row += 1
        for j in range(n1):
            M_b[row, j] = P1minus[3, j]
            M_a[row, j] = -P1plus[3, j]
        for j in range(n2):
            col = n1 + j
            M_b[row, col] = -P2plus[3, j]
            M_a[row, col] = P2minus[3, j]
        row += 1
    elif (kind1 == 1) ^ (kind2 == 1):
        for j in range(dim):
            M_b[row, j] = 0.0 + 0.0j
            M_a[row, j] = 0.0 + 0.0j
        if kind1 == 1:
            if elec_flag_1 == 0:
                for j in range(n1):
                    M_b[row, j] = U1minus[3, j]
                    M_a[row, j] = -U1plus[3, j]
            else:
                for j in range(n1):
                    M_b[row, j] = P1minus[3, j]
                    M_a[row, j] = -P1plus[3, j]
        else:
            if elec_flag_2 == 0:
                for j in range(n2):
                    col = n1 + j
                    M_b[row, col] = U2plus[3, j]
                    M_a[row, col] = -U2minus[3, j]
            else:
                for j in range(n2):
                    col = n1 + j
                    M_b[row, col] = P2plus[3, j]
                    M_a[row, col] = -P2minus[3, j]
        row += 1
    # Use LU solver for small systems (dim <= 4)
    if dim > 4:
        raise ValueError(f"Interface system size {dim} > 4 not supported")
    LU_work = np.zeros((4, 4), dtype=Complex)
    piv_work = np.zeros(4, dtype=np.int64)
    S_std_work = np.zeros((4, 4), dtype=Complex)
    solve_lu_inplace_n4(M_b, M_a, S_std_work, dim, dim, LU_work, piv_work)
    S_std = np.zeros((dim, dim), dtype=Complex)
    for i in range(dim):
        for j in range(dim):
            S_std[i, j] = S_std_work[i, j]
    for i in range(n1):
        for j in range(n1):
            S11_bf[i, j] = S_std[i, j]
    for i in range(n1):
        for j in range(n2):
            S12_bf[i, j] = S_std[i, n1 + j]
    for i in range(n2):
        for j in range(n1):
            S21_bf[i, j] = S_std[n1 + i, j]
    for i in range(n2):
        for j in range(n2):
            S22_bf[i, j] = S_std[n1 + i, n1 + j]


@njit
def build_Btop_ftop_from_fulltop_numba(Uplus_top, Pplus_top, Uminus_top, Pminus_top, mech_bc_flag_topwall, idx_free, B_top_out, f_top_out):
    """Build B_top and f_top from full top modes."""
    Ns = Uplus_top.shape[0]
    Nphi = Uplus_top.shape[1]
    n = 4
    for i_s in range(Ns):
        for j_phi in range(Nphi):
            Uplus = Uplus_top[i_s, j_phi]
            Pplus = Pplus_top[i_s, j_phi]
            Uminus = Uminus_top[i_s, j_phi]
            Pminus = Pminus_top[i_s, j_phi]
            B_top = B_top_out[i_s, j_phi]
            f_top = f_top_out[i_s, j_phi]
            for i in range(n):
                for j in range(n):
                    B_top[i, j] = 0.0 + 0.0j
                f_top[i] = 0.0 + 0.0j
            if mech_bc_flag_topwall == 0:
                for j in range(n):
                    for i in range(3):
                        B_top[i, j] = -Pminus[i, j]
                    B_top[3, j] = -Uminus[3, j]
                    if j == idx_free:
                        f_top[j] = 1.0 + 0.0j
            else:
                for j in range(n):
                    for i in range(3):
                        B_top[i, j] = -Uminus[i, j]
                    B_top[3, j] = -Uminus[3, j]
                    if j == idx_free:
                        f_top[j] = 1.0 + 0.0j


# ============================================================
# Load block from reflection (in-place)
# ============================================================

@njit
def S_block_load_from_R_inplace_numba(R, S11, S12, S21, S22, n_act):
    """Build S-block from reflection matrix R."""
    n_max = 4

    # zero everything
    for i in range(n_max):
        for j in range(n_max):
            S11[i, j] = 0.0 + 0.0j
            S12[i, j] = 0.0 + 0.0j
            S21[i, j] = 0.0 + 0.0j
            S22[i, j] = 0.0 + 0.0j

    # copy active reflection block
    for i in range(n_act):
        for j in range(n_act):
            S11[i, j] = R[i, j]

    # pad inactive channels as perfect mirrors
    for k in range(n_act, n_max):
        S11[k, k] = 0.0 + 0.0j


# ============================================================
# Propagation block from slownesses (in-place, padded-safe)
# ============================================================

@njit
def S_block_propagation_from_slow_inplace_numba(
    omega, loss_ratio, Swplus, Swminus, L,
    n_active, n_max,
    S11, S12, S21, S22
):
    """
    Build propagation S-block from slownesses.
    IMPORTANT SIGN:
      phase_plus  = exp(+1j * omega * Swplus  * L)
      phase_minus = exp(-1j * omega * Swminus * L)
    """
    _zero_nn(S11, n_max)
    _zero_nn(S12, n_max)
    _zero_nn(S21, n_max)
    _zero_nn(S22, n_max)

    for i in range(n_active):
        sp = Swplus[i]
        sm = Swminus[i]
        sp_eff = sp.real + 1j * (sp.imag * loss_ratio)
        sm_eff = sm.real + 1j * (sm.imag * loss_ratio)
        S21[i, i] = np.exp(1j * omega * sp_eff * L)
        S12[i, i] = np.exp(-1j * omega * sm_eff * L)


# ============================================================
# General Redheffer star product (in-place, n<=4, no alloc)
# ============================================================

@njit
def general_redheffer_star_inplace_numba(
    A11, A12, A21, A22,
    B11, B12, B21, B22,
    n,
    O11, O12, O21, O22,
    W1, W2, W3, RHS, SOL, LU, piv
):
    """
    Standard cascade: connect port2 of A to port1 of B.
    Formulas (with solves, no inverses formed explicitly):
      X = I - B11 A22
      Y = I - A22 B11
      O12 = A12 X^{-1} B12
      O11 = A11 + A12 X^{-1} (B11 A21)
      O21 = B21 Y^{-1} A21
      O22 = B22 + B21 Y^{-1} (A22 B12)
    """
    # W1 = B11 @ A22
    _matmul_nn(B11, A22, W1, n, W3)

    # W2 = X = I - W1
    _eye_n(W2, n)
    for i in range(n):
        for j in range(n):
            W2[i, j] -= W1[i, j]

    # --- O12: solve X * SOL = B12, then O12 = A12 @ SOL ---
    for i in range(n):
        for j in range(n):
            RHS[i, j] = B12[i, j]
    solve_lu_inplace_n4(W2, RHS, SOL, n, n, LU, piv)
    _matmul_nn(A12, SOL, O12, n, W3)

    # --- O11: RHS = B11 @ A21, solve X * SOL = RHS, tmp = A12 @ SOL, O11 = A11 + tmp ---
    _matmul_nn(B11, A21, RHS, n, W3)
    solve_lu_inplace_n4(W2, RHS, SOL, n, n, LU, piv)
    _matmul_nn(A12, SOL, W1, n, W3)  # reuse W1 as tmp
    for i in range(n):
        for j in range(n):
            O11[i, j] = A11[i, j] + W1[i, j]

    # W1 = A22 @ B11  (reuse)
    _matmul_nn(A22, B11, W1, n, W3)

    # W2 = Y = I - W1
    _eye_n(W2, n)
    for i in range(n):
        for j in range(n):
            W2[i, j] -= W1[i, j]

    # --- O21: solve Y * SOL = A21, then O21 = B21 @ SOL ---
    for i in range(n):
        for j in range(n):
            RHS[i, j] = A21[i, j]
    solve_lu_inplace_n4(W2, RHS, SOL, n, n, LU, piv)
    _matmul_nn(B21, SOL, O21, n, W3)

    # --- O22: RHS = A22 @ B12, solve Y * SOL = RHS, tmp = B21 @ SOL, O22 = B22 + tmp ---
    _matmul_nn(A22, B12, RHS, n, W3)
    solve_lu_inplace_n4(W2, RHS, SOL, n, n, LU, piv)
    _matmul_nn(B21, SOL, W1, n, W3)
    for i in range(n):
        for j in range(n):
            O22[i, j] = B22[i, j] + W1[i, j]


# ============================================================
# Top TEM↔modal block, ELECTRICAL ONLY (in-place, no alloc)
# ============================================================

@njit
def S_block_top_TEM_piezo_elec_only_inplace_numba(
    B_top, f_top,
    phi_plus, phi_minus,
    q_plus, q_minus,
    omega, Area, rootZ,
    S11t, S12t, S21t, S22t
):
    """
    Solves ONLY the two electrical equations, assuming mechanics already encoded in (B_top,f_top).
    
    Uses electrical conventions:
      (phi/Area) - rootZ(α0+β0)=0
      (i omega^2 q) + (1/rootZ)β0 - (1/rootZ)α0 = 0
    """
    n = 4

    # phi_a3 = phi_plus · f_top ; q_a3 = q_plus · f_top
    phi_a3 = 0.0 + 0.0j
    q_a3   = 0.0 + 0.0j
    for i in range(n):
        phi_a3 += phi_plus[i] * f_top[i]
        q_a3   += q_plus[i]   * f_top[i]

    E1_a3 = phi_a3 / Area
    iw2   = 1j * (omega * omega)
    E2_a3 = iw2 * q_a3

    det = (E1_a3 / rootZ) + (rootZ * E2_a3)

    # α0 response (b=0)
    a_free_alpha = (2.0 + 0.0j) / det
    beta_alpha   = (-E2_a3 * (rootZ + 0.0j) + E1_a3 * (1.0 / rootZ)) / det

    S11t[0, 0] = beta_alpha

    # S21t = f_top * a_free_alpha
    for i in range(n):
        S21t[i, 0] = f_top[i] * a_free_alpha

    # For each basis b_j: compute (beta_bj, a_free_bj) and fill S12t, S22t columns
    for j in range(n):
        # phi_bj = phi_minus[j] + sum_i phi_plus[i]*B_top[i,j]
        # q_bj   = q_minus[j]   + sum_i q_plus[i]  *B_top[i,j]
        phi_bj = phi_minus[j]
        q_bj   = q_minus[j]
        for i in range(n):
            phi_bj += phi_plus[i] * B_top[i, j]
            q_bj   += q_plus[i]   * B_top[i, j]

        E1_bj = phi_bj / Area
        E2_bj = iw2 * q_bj

        rhs1 = -E1_bj
        rhs2 = -E2_bj

        a_free_bj = (rhs1 / rootZ + rootZ * rhs2) / det
        beta_bj   = (-E2_a3 * rhs1 + E1_a3 * rhs2) / det

        S12t[0, j] = beta_bj

        # S22t[:,j] = B_top[:,j] + f_top[:] * a_free_bj
        for i in range(n):
            S22t[i, j] = B_top[i, j] + f_top[i] * a_free_bj


# ============================================================
# Terminate modal port with reflection R_eff to get Γ_TEM
# ============================================================

@njit
def gamma_from_top_and_R_inplace_numba(
    S11t, S12t, S21t, S22t,
    R_eff, n,
    M, RHSv, SOLv, LU, piv, Wtmp
):
    """
    Γ = S11t + S12t R_eff (I - S22t R_eff)^{-1} S21t
    
    Returns Γ as a complex scalar (stored in S11t[0,0] convention).
    """
    # Wtmp = S22t @ R_eff
    _matmul_nn(S22t, R_eff, Wtmp, n, Wtmp)
    # Build M = I - Wtmp
    _eye_n(M, n)
    for i in range(n):
        for j in range(n):
            M[i, j] -= Wtmp[i, j]

    # RHSv = S21t
    for i in range(n):
        RHSv[i, 0] = S21t[i, 0]

    # SOLv = M^{-1} RHSv
    solve_lu_inplace_n4(M, RHSv, SOLv, n, 1, LU, piv)

    # y = R_eff @ SOLv  (store back into RHSv)
    for i in range(n):
        s = 0.0 + 0.0j
        for k in range(n):
            s += R_eff[i, k] * SOLv[k, 0]
        RHSv[i, 0] = s

    # Γ = S11t + S12t @ y
    acc = S11t[0, 0]
    for j in range(n):
        acc += S12t[0, j] * RHSv[j, 0]
    return acc


# ============================================================
# Compute gamma for multilayer (allocation-free)
# ============================================================

@njit
def compute_gamma_multilayer_elec_only_inplace_numba(
    omega, loss_ratio, Area, rootZ,
    N_layers,
    L_layers, n_modes_layer,
    Swplus_row, Swminus_row,
    S11_iface_row, S12_iface_row, S21_iface_row, S22_iface_row,
    R_back_row,
    B_top, f_top,
    phi_plus, phi_minus,
    q_plus, q_minus,
    S11_stack, S12_stack, S21_stack, S22_stack,
    S11_tmp, S12_tmp, S21_tmp, S22_tmp,
    S11p, S12p, S21p, S22p,
    W1, W2, W3, RHS, SOL, LU, piv,
    S11t, S12t, S21t, S22t,
    M, RHSv, SOLv, Wtmp
):
    """Compute gamma for multilayer structure."""
    n_act = n_modes_layer[N_layers-1]
    n_max = 4

    # 0) start from backwall load as a 2-port in modal space
    S_block_load_from_R_inplace_numba(R_back_row, S11_stack, S12_stack, S21_stack, S22_stack, n_act)

    # 1) march bottom → top
    for ell in range(N_layers - 1, -1, -1):
        n_act = n_modes_layer[ell]

        # Build propagation 2-port for this layer
        S_block_propagation_from_slow_inplace_numba(
            omega, loss_ratio,
            Swplus_row[ell], Swminus_row[ell], L_layers[ell],
            n_act, n_max,
            S11p, S12p, S21p, S22p
        )

        # stack = prop ⋆ stack
        general_redheffer_star_inplace_numba(
            S11p, S12p, S21p, S22p,
            S11_stack, S12_stack, S21_stack, S22_stack,
            n_max,
            S11_tmp, S12_tmp, S21_tmp, S22_tmp,
            W1, W2, W3, RHS, SOL, LU, piv
        )
        # copy tmp -> stack
        for i in range(n_max):
            for j in range(n_max):
                S11_stack[i, j] = S11_tmp[i, j]
                S12_stack[i, j] = S12_tmp[i, j]
                S21_stack[i, j] = S21_tmp[i, j]
                S22_stack[i, j] = S22_tmp[i, j]

        # interface above this layer (unless we're at the top)
        if ell > 0:
            general_redheffer_star_inplace_numba(
                S11_iface_row[ell - 1], S12_iface_row[ell - 1], S21_iface_row[ell - 1], S22_iface_row[ell - 1],
                S11_stack, S12_stack, S21_stack, S22_stack,
                n_max,
                S11_tmp, S12_tmp, S21_tmp, S22_tmp,
                W1, W2, W3, RHS, SOL, LU, piv
            )
            for i in range(n_max):
                for j in range(n_max):
                    S11_stack[i, j] = S11_tmp[i, j]
                    S12_stack[i, j] = S12_tmp[i, j]
                    S21_stack[i, j] = S21_tmp[i, j]
                    S22_stack[i, j] = S22_tmp[i, j]

    # 2) Build top TEM↔modal block (electrical only; mechanics already eliminated into B_top,f_top)
    S_block_top_TEM_piezo_elec_only_inplace_numba(
        B_top, f_top,
        phi_plus, phi_minus,
        q_plus, q_minus,
        omega, Area, rootZ,
        S11t, S12t, S21t, S22t
    )

    # 3) Treat the entire multilayer as modal load: R_eff := S11_stack
    return gamma_from_top_and_R_inplace_numba(
        S11t, S12t, S21t, S22t,
        S11_stack,
        n_max,
        M, RHSv, SOLv,
        LU, piv,
        Wtmp
    )


# ============================================================
# Simpson integration factors
# ============================================================

@njit
def simpson_factors_uniform_best(N):
    """
    Dimensionless factors f[i] such that
        ∫ f(s) ds ≈ ds * Σ_i f[i] f(s_i)
    (ds is already inside your const_prefac).

    Odd N: pure composite Simpson.
    Even N: composite Simpson on [0..N-4] + Simpson 3/8 on last 3 intervals [N-4..N-1].
    """
    fac = np.zeros(N, dtype=np.float64)
    if N < 2:
        fac[:] = 1.0
        return fac

    if (N % 2) == 1:
        # pure Simpson: coeffs/3
        fac[0]  = 1.0/3.0
        fac[-1] = 1.0/3.0
        fac[1:-1:2] = 4.0/3.0
        fac[2:-2:2] = 2.0/3.0
        return fac

    # even N: Simpson on 0..N-4 (inclusive), which is (N-3) points (odd)
    end = N - 4
    fac[0] = 1.0/3.0
    fac[end] = 1.0/3.0
    fac[1:end:2] = 4.0/3.0
    fac[2:end-1:2] = 2.0/3.0

    # add 3/8 rule on last 4 points (end..N-1):
    # integral ≈ (3ds/8) * (f_end + 3 f_{end+1} + 3 f_{end+2} + f_{end+3})
    # => factors add: 3/8 * [1,3,3,1]
    fac[end]     += 3.0/8.0
    fac[end + 1] += 9.0/8.0
    fac[end + 2] += 9.0/8.0
    fac[end + 3] += 3.0/8.0

    return fac


# ============================================================
# Slowness integral computation
# ============================================================

@njit(parallel=True)
def slowness_integral_numba_block_multilayer_elec_only_fast_inplace_simpson(
    omegas, loss_ratio, s_vals,
    N_layers, L_layers, n_modes_layer,
    Swplus_layers_arr, Swminus_layers_arr,
    S11_iface_arr, S12_iface_arr, S21_iface_arr, S22_iface_arr,
    R_back_arr,
    B_top_arr, f_top_arr,
    phi_plus_top_arr, phi_minus_top_arr,
    q_plus_top_arr, q_minus_top_arr,
    Area, rootZ, const_prefac, piston, s_w,
    I_omega,
    S11_stack_s, S12_stack_s, S21_stack_s, S22_stack_s,
    S11_tmp_s, S12_tmp_s, S21_tmp_s, S22_tmp_s,
    S11p_s, S12p_s, S21p_s, S22p_s,
    W1_s, W2_s, W3_s, RHS_s, SOL_s, LU_s, piv_s,
    S11t_s, S12t_s, S21t_s, S22t_s,
    M_s, RHSv_s, SOLv_s, Wtmp_s
):
    """Compute slowness integral for multilayer structure."""
    Nω   = omegas.shape[0]
    Ns   = s_vals.shape[0]
    Nphi = Swplus_layers_arr.shape[1]

    for iω in prange(Nω):
        omega = omegas[iω]
        lr = loss_ratio[iω]
        acc = 0.0 + 0.0j

        # per-ω scratch views
        S11_stack = S11_stack_s[iω]; S12_stack = S12_stack_s[iω]; S21_stack = S21_stack_s[iω]; S22_stack = S22_stack_s[iω]
        S11_tmp   = S11_tmp_s[iω];   S12_tmp   = S12_tmp_s[iω];   S21_tmp   = S21_tmp_s[iω];   S22_tmp   = S22_tmp_s[iω]
        S11p      = S11p_s[iω];      S12p      = S12p_s[iω];      S21p      = S21p_s[iω];      S22p      = S22p_s[iω]

        W1  = W1_s[iω]; W2 = W2_s[iω]; W3 = W3_s[iω]
        RHS = RHS_s[iω]; SOL = SOL_s[iω]; LU = LU_s[iω]; piv = piv_s[iω]

        S11t = S11t_s[iω]; S12t = S12t_s[iω]; S21t = S21t_s[iω]; S22t = S22t_s[iω]
        M    = M_s[iω];    RHSv = RHSv_s[iω];  SOLv = SOLv_s[iω]
        Wtmp = Wtmp_s[iω]

        for i_s in range(Ns):
            ps = piston[iω, i_s]
            if ps == 0.0:
                continue

            # weight for this s-node (dimensionless Simpson factor lives in s_w)
            w = const_prefac * ps * s_w[i_s]
            if w == 0.0:
                continue

            # F(s_i) = Σ_phi Gamma(omega, s_i, phi)
            sum_phi = 0.0 + 0.0j
            for j_phi in range(Nphi):
                sum_phi += compute_gamma_multilayer_elec_only_inplace_numba(
                    omega, lr, Area, rootZ,
                    N_layers,
                    L_layers, n_modes_layer,
                    Swplus_layers_arr[i_s, j_phi],
                    Swminus_layers_arr[i_s, j_phi],
                    S11_iface_arr[i_s, j_phi],
                    S12_iface_arr[i_s, j_phi],
                    S21_iface_arr[i_s, j_phi],
                    S22_iface_arr[i_s, j_phi],
                    R_back_arr[i_s, j_phi],
                    B_top_arr[i_s, j_phi],
                    f_top_arr[i_s, j_phi],
                    phi_plus_top_arr[i_s, j_phi],
                    phi_minus_top_arr[i_s, j_phi],
                    q_plus_top_arr[i_s, j_phi],
                    q_minus_top_arr[i_s, j_phi],
                    S11_stack, S12_stack, S21_stack, S22_stack,
                    S11_tmp,   S12_tmp,   S21_tmp,   S22_tmp,
                    S11p, S12p, S21p, S22p,
                    W1, W2, W3, RHS, SOL, LU, piv,
                    S11t, S12t, S21t, S22t,
                    M, RHSv, SOLv,
                    Wtmp
                )

            acc += w * sum_phi

        I_omega[iω] = acc


# ============================================================
# Block tuple flattening helper
# ============================================================

def _flatten_block_tuple(x: Union[tuple, list]) -> Tuple:
    """Flatten only tuples/lists; NEVER flatten numpy arrays."""
    out = []
    def rec(y):
        if isinstance(y, (tuple, list)):
            for z in y:
                rec(z)
        else:
            out.append(y)
    rec(x)
    return tuple(out)


# ============================================================
# Block compression function
# ============================================================

def compress_blocks_fulltop_to_elec_only(
    blocks_full: List,
    mech_bc_flag_topwall: int,
    top_passthrough_idx: int = 3
) -> List:
    """
    Compress full-top blocks to electrical-only format.
    
    Args:
        blocks_full: List of full block tuples
        mech_bc_flag_topwall: Mechanical boundary condition flag
        top_passthrough_idx: Top passthrough index
        
    Returns:
        blocks_elec: List of compressed electrical-only blocks
    """
    blocks_elec = []
    for blk in blocks_full:
        # allow 8-tuple or 9-tuple (with optional loss_ratio_b)
        loss_ratio_b = None
        if len(blk) == 9:
            idx_b, omegas_b, s_vals_b, const_prefac_b, piston_b, L_layers_b, n_modes_layer_b, out, loss_ratio_b = blk
        elif len(blk) == 8:
            idx_b, omegas_b, s_vals_b, const_prefac_b, piston_b, L_layers_b, n_modes_layer_b, out = blk
        else:
            raise ValueError(f"Expected block len 8 or 9, got {len(blk)}")

        out = tuple(out)

        if len(out) == 14:
            Swplus_layers, Swminus_layers = out[0], out[1]
            S11_iface, S12_iface, S21_iface, S22_iface = out[2], out[3], out[4], out[5]
            R_back_arr = out[7]
            B_top_arr  = out[8]
            f_top_arr  = out[9]
            phi_plus_top, phi_minus_top, q_plus_top, q_minus_top = out[10], out[11], out[12], out[13]

        elif len(out) in (16, 18):
            raise ValueError(
                "You built with full modes; either set store_top_full_modes=False, store_top_Bf=True "
                "or keep your older compressor path."
            )
        else:
            raise ValueError(f"Unexpected out length {len(out)}; expected 14 when store_top_Bf=True.")

        blocks_elec.append((
            idx_b, omegas_b, s_vals_b, const_prefac_b, piston_b,
            L_layers_b, n_modes_layer_b,
            (Swplus_layers, Swminus_layers,
             S11_iface, S12_iface, S21_iface, S22_iface,
             R_back_arr,
             B_top_arr, f_top_arr,
             phi_plus_top, phi_minus_top, q_plus_top, q_minus_top),
            loss_ratio_b
        ))

    return blocks_elec


# ============================================================
# Main integration function
# ============================================================

def integrate_all_blocks_streaming_inplace_elec_only(
    freqs: np.ndarray,
    blocks: List,
    L_layers: np.ndarray,
    a_radius: float,
    Z0: float,
    show_progress_blocks: bool = True
) -> np.ndarray:
    """
    Integrate all blocks for electrical-only computation.
    
    Args:
        freqs: Frequency array (Nf,)
        blocks: List of block tuples
        L_layers: Layer thicknesses (N_layers,)
        a_radius: Aperture radius
        Z0: Characteristic impedance
        show_progress_blocks: Whether to show progress
        
    Returns:
        I_omega: Complex frequency response (Nf,)
    """
    freqs  = np.asarray(freqs, dtype=np.float64)
    omegas = 2.0 * np.pi * freqs

    Area  = np.pi * a_radius**2
    rootZ = np.sqrt(Z0)

    I_omega  = np.zeros_like(omegas, dtype=Complex)
    N_layers = int(L_layers.shape[0])
    n_max    = 4

    it = range(len(blocks))
    if show_progress_blocks:
        try:
            from tqdm.auto import tqdm
            it = tqdm(it, desc="Integrating blocks")
        except ImportError:
            pass

    for b in it:
        blk = _flatten_block_tuple(blocks[b])
        nblk = len(blk)
        loss_ratio_b = None  # will default to ones if not provided

        # Accept either:
        #  - flat 19-tuple (your "clean" elec-only blocks), OR
        #  - flattened 20-tuple that still contains L_layers inside the block, OR
        #  - fully packed 8/9-tuple (idx, omegas, s, const, piston, L_layers, n_modes, out_tuple)
        if nblk == 9:
            # packed form with per-frequency loss_ratio_b
            idx_b, omegas_b, s_vals_b, const_prefac_b, piston_b, L_layers_b, n_modes_layer, out, loss_ratio_b = blk
            out = _flatten_block_tuple(out)
            if len(out) != 13:
                raise ValueError(f"Packed block 'out' should have 13 arrays, got {len(out)}")
            (Swplus_layers, Swminus_layers,
             S11_iface, S12_iface, S21_iface, S22_iface,
             R_back_arr,
             B_top_arr, f_top_arr,
             phi_plus_top, phi_minus_top, q_plus_top, q_minus_top) = out

        elif nblk == 8:
            # packed form: (..., out_tuple) where out_tuple has 13 arrays
            idx_b, omegas_b, s_vals_b, const_prefac_b, piston_b, L_layers_b, n_modes_layer, out = blk
            out = _flatten_block_tuple(out)
            if len(out) != 13:
                raise ValueError(f"Packed block 'out' should have 13 arrays, got {len(out)}")
            (Swplus_layers, Swminus_layers,
             S11_iface, S12_iface, S21_iface, S22_iface,
             R_back_arr,
             B_top_arr, f_top_arr,
             phi_plus_top, phi_minus_top, q_plus_top, q_minus_top) = out

        elif nblk == 21:
            # flattened packed form with per-frequency loss_ratio_b appended
            (idx_b, omegas_b, s_vals_b, const_prefac_b, piston_b,
             L_layers_b, n_modes_layer,
             Swplus_layers, Swminus_layers,
             S11_iface, S12_iface, S21_iface, S22_iface,
             R_back_arr,
             B_top_arr, f_top_arr,
             phi_plus_top, phi_minus_top, q_plus_top, q_minus_top,
             loss_ratio_b) = blk

        elif nblk == 20:
            # flattened packed form, with L_layers explicitly present
            (idx_b, omegas_b, s_vals_b, const_prefac_b, piston_b,
             L_layers_b, n_modes_layer,
             Swplus_layers, Swminus_layers,
             S11_iface, S12_iface, S21_iface, S22_iface,
             R_back_arr,
             B_top_arr, f_top_arr,
             phi_plus_top, phi_minus_top, q_plus_top, q_minus_top) = blk

        elif nblk == 19:
            # clean form (no L_layers stored in block)
            (idx_b, omegas_b, s_vals_b, const_prefac_b, piston_b,
             n_modes_layer,
             Swplus_layers, Swminus_layers,
             S11_iface, S12_iface, S21_iface, S22_iface,
             R_back_arr,
             B_top_arr, f_top_arr,
             phi_plus_top, phi_minus_top, q_plus_top, q_minus_top) = blk

        else:
            raise ValueError(
                f"Unexpected block tuple length {nblk}. Expected 8/9 (packed), 19 (flat), 20 (flat+L_layers), or 21 (flat+L_layers+loss_ratio)."
            )

        if (loss_ratio_b is None) or (not hasattr(loss_ratio_b, 'shape')):
            loss_ratio_b = np.ones(omegas_b.shape[0], dtype=np.float64)
        else:
            loss_ratio_b = np.asarray(loss_ratio_b, dtype=np.float64)

        # -------- per-block scratch (allocated once per block) --------
        Nωb = omegas_b.shape[0]

        z44 = lambda: np.zeros((Nωb, n_max, n_max), dtype=Complex)
        z41 = lambda: np.zeros((Nωb, n_max, 1), dtype=Complex)
        z14 = lambda: np.zeros((Nωb, 1, n_max), dtype=Complex)
        z11 = lambda: np.zeros((Nωb, 1, 1), dtype=Complex)

        S11_stack_s = z44(); S12_stack_s = z44(); S21_stack_s = z44(); S22_stack_s = z44()
        S11_tmp_s   = z44(); S12_tmp_s   = z44(); S21_tmp_s   = z44(); S22_tmp_s   = z44()
        S11p_s      = z44(); S12p_s      = z44(); S21p_s      = z44(); S22p_s      = z44()

        W1_s  = z44(); W2_s  = z44(); W3_s  = z44()
        RHS_s = z44(); SOL_s = z44(); LU_s  = z44()
        piv_s = np.zeros((Nωb, n_max), dtype=np.int64)

        S11t_s = z11()
        S12t_s = z14()
        S21t_s = z41()
        S22t_s = z44()

        M_s    = z44()
        RHSv_s = z41()
        SOLv_s = z41()
        Wtmp_s = z44()

        I_block = np.zeros(Nωb, dtype=Complex)
        Ns   = s_vals_b.shape[0]
        s_w_b = simpson_factors_uniform_best(Ns)
        slowness_integral_numba_block_multilayer_elec_only_fast_inplace_simpson(
            omegas=omegas_b,
            loss_ratio=loss_ratio_b,
            s_vals=s_vals_b,
            N_layers=N_layers,
            L_layers=L_layers,
            n_modes_layer=n_modes_layer,
            Swplus_layers_arr=Swplus_layers,
            Swminus_layers_arr=Swminus_layers,
            S11_iface_arr=S11_iface,
            S12_iface_arr=S12_iface,
            S21_iface_arr=S21_iface,
            S22_iface_arr=S22_iface,
            R_back_arr=R_back_arr,
            B_top_arr=B_top_arr,
            f_top_arr=f_top_arr,
            phi_plus_top_arr=phi_plus_top,
            phi_minus_top_arr=phi_minus_top,
            q_plus_top_arr=q_plus_top,
            q_minus_top_arr=q_minus_top,
            Area=Area,
            rootZ=rootZ,
            const_prefac=const_prefac_b,
            piston=piston_b,
            s_w=s_w_b,
            I_omega=I_block,
            S11_stack_s=S11_stack_s, S12_stack_s=S12_stack_s, S21_stack_s=S21_stack_s, S22_stack_s=S22_stack_s,
            S11_tmp_s=S11_tmp_s, S12_tmp_s=S12_tmp_s, S21_tmp_s=S21_tmp_s, S22_tmp_s=S22_tmp_s,
            S11p_s=S11p_s, S12p_s=S12p_s, S21p_s=S21p_s, S22p_s=S22p_s,
            W1_s=W1_s, W2_s=W2_s, W3_s=W3_s, RHS_s=RHS_s, SOL_s=SOL_s, LU_s=LU_s, piv_s=piv_s,
            S11t_s=S11t_s, S12t_s=S12t_s, S21t_s=S21t_s, S22t_s=S22t_s,
            M_s=M_s, RHSv_s=RHSv_s, SOLv_s=SOLv_s,
            Wtmp_s=Wtmp_s
        )

        I_omega[idx_b] = I_block

    return I_omega

def make_log_blocks_from_ratio(
    f_min: float,
    f_max: float,
    R_max: float = 1.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Divide a frequency range into logarithmically-spaced blocks.
    
    Each block spans at most a factor of R_max in frequency, using the
    minimal number of blocks necessary.
    
    Parameters
    ----------
    f_min : float
        Minimum frequency of the range (must be positive).
    f_max : float
        Maximum frequency of the range (must be > f_min).
    R_max : float, optional
        Maximum frequency ratio per block (must be > 1).
        Default is 1.5.
    
    Returns
    -------
    edges : np.ndarray
        Array of block edges with shape (N_blocks + 1,).
    f_ref : np.ndarray
        Array of geometric mean frequencies for each block with shape (N_blocks,).
    
    Raises
    ------
    ValueError
        If input parameters violate constraints.
    """
    # Validate inputs
    if f_min <= 0:
        raise ValueError(f"f_min must be positive, got {f_min}")
    if f_max <= f_min:
        raise ValueError(f"f_max must be greater than f_min, got f_max={f_max}, f_min={f_min}")
    if R_max <= 1:
        raise ValueError(f"R_max must be greater than 1, got {R_max}")
    
    # Calculate required number of blocks
    log_span = np.log(f_max / f_min)
    log_max_ratio = np.log(R_max)
    
    n_blocks = max(1, int(np.ceil(log_span / log_max_ratio)))
    
    # Generate block edges (logarithmically spaced)
    edges = np.logspace(
        start=np.log10(f_min),
        stop=np.log10(f_max),
        num=n_blocks + 1
    )
    
    # Calculate geometric mean frequencies for each block
    f_ref = np.sqrt(edges[:-1] * edges[1:])
    
    return edges, f_ref


def build_slowness_blocks_streaming(
    freqs: np.ndarray,
    edges: np.ndarray,
    f_ref_blocks: np.ndarray,
    omega_s_a_max: np.ndarray,
    du: float,
    Nphi: int,
    a_radius: float,
    layers: list,
    # topwall controls
    mech_bc_flag_topwall: int,          # 0 free, 1 clamped
    top_elec_flag: int,                # 0 short, 1 open, 3 passthrough (NOT 2 here)
    top_Z_elec: complex,               # only meaningful if top_elec_flag==2 (disallowed)
    top_elec_passthrough_idx: int,     # usually 3
    # backwall
    mech_bc_flag_backwall: int,
    R_max: float = 1.3,
    tol_im: float = 1e-15,
    tol_s: float = 1e-21,
    show_progress_blocks: bool = True,
    store_top_full_modes: bool = False,

    # NEW:
    store_top_Bf: bool = True,
    tan_delta_block_fns_per_layer=None,   # dict(name->fn) OR list aligned with layers OR None


    # NEW: within-block attenuation scaling (propagation-only)
    loss_fn=None,  # callable f_hz -> loss scalar (dimensionless) or None

    **kwargs,  # allows future kwargs without TypeError
):
    (kind, L_layers, rho, _tan_delta, _tan_eps,
     bottom_elec_flag, _bottom_Z_elec, C4, e, eps, n_modes_layer) = pack_layers_for_numba(layers)

    # Accept dict keyed by layer["name"] and convert to list aligned with `layers`
    if isinstance(tan_delta_block_fns_per_layer, dict):
        fns_list = tandelta_fns_from_layers(layers, tan_delta_block_fns_per_layer, strict=False)
    else:
        fns_list = tan_delta_block_fns_per_layer  # already list/None

    return build_slowness_blocks_streaming_packed(
        freqs=freqs,
        edges=edges,
        f_ref_blocks=f_ref_blocks,
        omega_s_a_max=omega_s_a_max,
        du=du,
        Nphi=Nphi,
        a_radius=a_radius,

        kind=kind,
        L_layers=L_layers,
        n_modes_layer=n_modes_layer,
        rho=rho,
        bottom_elec_flag=bottom_elec_flag,
        C4=C4, e=e, eps=eps,

        mech_bc_flag_topwall=mech_bc_flag_topwall,
        top_elec_flag=top_elec_flag,
        top_Z_elec=top_Z_elec,
        top_elec_passthrough_idx=top_elec_passthrough_idx,
        mech_bc_flag_backwall=mech_bc_flag_backwall,

        R_max=R_max,
        tol_im=tol_im,
        tol_s=tol_s,
        show_progress_blocks=show_progress_blocks,
        store_top_full_modes=store_top_full_modes,

        # NEW forward:
        store_top_Bf=store_top_Bf,
        tan_delta_block_fns_per_layer=fns_list,

        # NEW forward:
        loss_fn=loss_fn,

        **kwargs,
    )



KIND_ELASTIC = np.int8(0)
KIND_PIEZO   = np.int8(1)

def pack_layers_for_numba(layers):
    N = len(layers)

    kind = np.empty(N, dtype=np.int8)
    L    = np.empty(N, dtype=np.float64)
    rho  = np.empty(N, dtype=np.float64)

    tan_delta = np.zeros(N, dtype=np.float64)
    tan_eps   = np.zeros(N, dtype=np.float64)

    bottom_elec_flag = np.zeros(N, dtype=np.int8)
    bottom_Z_elec    = np.zeros(N, dtype=Complex)

    C4  = np.zeros((N, 3, 3, 3, 3), dtype=Complex)
    e   = np.zeros((N, 3, 3, 3),    dtype=Complex)
    eps = np.zeros((N, 3, 3),       dtype=Complex)

    for i, lay in enumerate(layers):
        k = lay.get("kind", "elastic")
        kind[i] = KIND_PIEZO if (k == "piezo") else KIND_ELASTIC

        L[i]   = float(lay["L"])
        rho[i] = float(lay["rho"])

        tan_delta[i] = float(lay.get("tan_delta", 0.0))
        tan_eps[i]   = float(lay.get("tan_eps", 0.0))

        C4[i] = lay["C4_eff"]

        if kind[i] == KIND_PIEZO:
            e[i]   = lay["e_ijk"]
            eps[i] = lay["eps_tensor"]
            bottom_elec_flag[i] = np.int8(lay.get("bottom_elec_flag", 0))
            bottom_Z_elec[i]    = Complex(lay.get("bottom_Z_elec", 0.0 + 0.0j))

    n_modes_layer = np.empty(N, dtype=np.int64)
    for i in range(N):
        n_modes_layer[i] = 4 if kind[i] == KIND_PIEZO else 3

    return (
        np.ascontiguousarray(kind),
        np.ascontiguousarray(L),
        np.ascontiguousarray(rho),
        np.ascontiguousarray(tan_delta),
        np.ascontiguousarray(tan_eps),
        np.ascontiguousarray(bottom_elec_flag),
        np.ascontiguousarray(bottom_Z_elec),
        np.ascontiguousarray(C4),
        np.ascontiguousarray(e),
        np.ascontiguousarray(eps),
        np.ascontiguousarray(n_modes_layer),
    )


def tandelta_fns_from_layers(layers, td_fns_by_name, *, strict=False):
    """
    layers: list of layer dicts with 'name'
    td_fns_by_name: dict name -> callable(f_ref_hz)->Δtanδ
    Returns list aligned with layer order.
    """
    fns = []
    for i, layer in enumerate(layers):
        name = layer.get("name", f"layer_{i}")
        fn = td_fns_by_name.get(name, None)
        if strict and (fn is None):
            raise KeyError(f"No tanδ block function provided for layer name {name!r}")
        fns.append(fn)
    return fns


def build_slowness_blocks_streaming_packed(
    freqs: np.ndarray,
    edges: np.ndarray,
    f_ref_blocks: np.ndarray,
    omega_s_a_max: np.ndarray,
    du: float,
    Nphi: int,
    a_radius: float,
    # packed layers
    kind: np.ndarray,
    L_layers: np.ndarray,
    n_modes_layer: np.ndarray,
    rho: np.ndarray,
    bottom_elec_flag: np.ndarray,
    C4: np.ndarray,
    e: np.ndarray,
    eps: np.ndarray,
    # topwall controls
    mech_bc_flag_topwall: int,
    top_elec_flag: int,
    top_Z_elec: complex,
    top_elec_passthrough_idx: int,
    # backwall
    mech_bc_flag_backwall: int,
    R_max: float = 1.3,
    tol_im: float = 1e-15,
    tol_s: float = 1e-21,
    show_progress_blocks: bool = True,
    store_top_full_modes: bool = False,

    # NEW:
    store_top_Bf: bool = True,
    tan_delta_block_fns_per_layer=None,   # list aligned with layers OR None


    # NEW: within-block attenuation scaling (propagation-only)
    loss_fn=None,  # callable f_hz -> loss scalar (dimensionless) or None

    **kwargs,
):
    if top_elec_flag == 2:
        raise ValueError("top_elec_flag==2 (loaded) is disallowed in cached blocks; use 0/1/3.")

    phis = np.linspace(0.0, 2.0*np.pi, Nphi, endpoint=False)
    edges = np.asarray(edges, dtype=freqs.dtype)
    edges[0]  = min(edges[0],  freqs[0])
    edges[-1] = max(edges[-1], freqs[-1])

    blocks = []

    N_layers = int(kind.shape[0])

    # normalise list length once
    if tan_delta_block_fns_per_layer is None:
        layer_fns = [None] * N_layers
    else:
        layer_fns = list(tan_delta_block_fns_per_layer)
        if len(layer_fns) != N_layers:
            raise ValueError(f"Need {N_layers} tanδ fns (or None), got {len(layer_fns)}")

    block_indices = range(len(f_ref_blocks))
    if show_progress_blocks and (tqdm is not None):
        block_indices = tqdm(block_indices, desc="Building slowness blocks")

    for b in block_indices:
        f_ref = float(f_ref_blocks[b])
        f_lo, f_hi = edges[b], edges[b + 1]

        if b < len(edges) - 2:
            mask_b = (freqs >= f_lo) & (freqs <  f_hi)
        else:
            mask_b = (freqs >= f_lo) & (freqs <= f_hi)

        if not np.any(mask_b):
            continue

        idx_b    = np.nonzero(mask_b)[0]
        freqs_b  = freqs[idx_b]
        omegas_b = 2.0 * np.pi * freqs_b

        # within-block attenuation scaling ratio: f(freq)/f(f_ref)
        if loss_fn is None:
            loss_ratio_b = np.ones_like(freqs_b, dtype=np.float64)
        else:
            loss_ref = float(loss_fn(f_ref))
            if loss_ref == 0.0:
                loss_ref = 1e-30  # avoid divide-by-zero

            # try vectorised loss_fn(freqs_b); fallback to scalar loop
            try:
                loss_now = np.asarray(loss_fn(freqs_b), dtype=np.float64)
            except Exception:
                loss_now = np.asarray([float(loss_fn(float(ff))) for ff in freqs_b], dtype=np.float64)

            loss_ratio_b = loss_now / loss_ref

        omega_ref = 2.0 * np.pi * f_ref
        s_max_b   = omega_s_a_max[b] / (omega_ref * a_radius)

        Ns_b     = int(np.ceil(omega_s_a_max[b] / du)) + 1
        s_vals_b = np.linspace(0.0, s_max_b, Ns_b)

        ds_b = (s_vals_b[2] - s_vals_b[1]) if (Ns_b > 2) else s_vals_b[-1]/Ns_b
        dphi = 2.0 * np.pi / Nphi
        const_prefac_b = ds_b * dphi / np.pi

        # --- NEW: compute extra tanδ at this block’s f_ref and apply to C4 ---
        td_block_add = np.zeros(N_layers, dtype=np.float64)
        for i, fn in enumerate(layer_fns):
            if fn is not None:
                td_block_add[i] = float(fn(f_ref))

        C4_eff_block = apply_block_extra_tandelta_to_C4_stack(C4, td_block_add)

        # heavy part (Numba inside)
        out = build_block_caches_streaming(
            kind=kind,
            rho=rho,
            bottom_elec_flag=bottom_elec_flag,
            C4=C4_eff_block,  # <-- use modified stiffness for this block
            e=e, eps=eps,
            s_vals_b=s_vals_b,
            phis=phis,

            mech_bc_flag_topwall=mech_bc_flag_topwall,
            top_elec_flag=top_elec_flag,
            top_Z_elec=top_Z_elec,
            top_elec_passthrough_idx=top_elec_passthrough_idx,
            mech_bc_flag_backwall=mech_bc_flag_backwall,

            tol_im=tol_im,
            tol_s=tol_s,
            store_top_full_modes=store_top_full_modes,

            # only include this if your build_block_caches_streaming signature has it
            store_top_Bf=store_top_Bf,
        )

        # piston kernel
        omega_s_a_b = (omegas_b[:, None] * a_radius) * s_vals_b[None, :]
        piston_b    = np.empty_like(omega_s_a_b)
        piston_b[:, 0]  = 0.0
        piston_b[:, 1:] = j1(omega_s_a_b[:, 1:])**2 / s_vals_b[None, 1:]

        blocks.append(
            (idx_b, omegas_b, s_vals_b, const_prefac_b, piston_b,
             L_layers, n_modes_layer,
             out,
             loss_ratio_b)
        )

    return blocks


def tan_delta_powerlaw(f_hz: float, *, tan0: float, f0_hz: float, exponent: float,
                       tan_min: float = 0.0, tan_max: float = np.inf) -> float:
    f = float(f_hz)
    val = tan0 * (f / float(f0_hz))**float(exponent)
    return float(np.clip(val, tan_min, tan_max))


def tan_delta_debye(w, tau, Delta, tan_bg=0.0):
    #w = 2*np.pi*np.asarray(f_hz)
    return tan_bg + Delta*(w*tau)/(1 + (w*tau)**2)


def apply_block_extra_tandelta_to_C4_stack(C4_stack: np.ndarray, td_block_add: np.ndarray) -> np.ndarray:
    """
    Option (2): C4_stack is already baseline lossy. Apply ONLY extra block loss:
      C_block = C4_stack * (1 - i * Δtanδ_block)

    Shapes:
      C4_stack: (N_layers, 3, 3, 3, 3)
      td_block_add: (N_layers,)
    """
    C4c = np.asarray(C4_stack, dtype=np.complex128)
    td  = np.asarray(td_block_add, dtype=np.float64)
    return C4c * (1.0 - 1j * td[:, None, None, None, None])


def build_block_caches_streaming(
    kind, rho, bottom_elec_flag,
    C4, e, eps,
    s_vals_b, phis,
    mech_bc_flag_topwall,
    top_elec_flag,
    top_Z_elec,
    top_elec_passthrough_idx,
    mech_bc_flag_backwall,
    tol_im, tol_s,
    store_top_full_modes: bool = False,
    store_top_Bf: bool = True,          # <-- NEW (default ON)
):
    """
    Per-block cache builder.

    Always returns:
      Swplus_layers, Swminus_layers : (Ns_b, Nphi, N_layers, 4)
      S11_iface,S12_iface,S21_iface,S22_iface : (Ns_b, Nphi, N_ifaces, 4, 4)
      R_top_arr, R_back_arr : (Ns_b, Nphi, 4, 4)
      If store_top_Bf=True:
        B_top_arr : (Ns_b, Nphi, 4, 4)
        f_top_arr : (Ns_b, Nphi, 4)
      Uphi_plus_top, Uphi_minus_top, q_plus_top, q_minus_top : (Ns_b, Nphi, 4)

    If store_top_full_modes=True, also returns:
      Uplus_top,Pplus_top,Uminus_top,Pminus_top : (Ns_b, Nphi, 4, 4)
    """
    if int(top_elec_flag) == 2:
        raise ValueError("top_elec_flag==2 (impedance load) depends on omega^2; build per-omega, not in block caches.")

    Ns_b     = s_vals_b.shape[0]
    Nphi     = phis.shape[0]
    N_layers = kind.shape[0]
    N_ifaces = N_layers - 1

    cos_phi = np.ascontiguousarray(np.cos(phis))
    sin_phi = np.ascontiguousarray(np.sin(phis))

    # outputs
    Swplus_layers  = np.zeros((Ns_b, Nphi, N_layers, 4), dtype=Complex)
    Swminus_layers = np.zeros_like(Swplus_layers)

    S11_iface = np.zeros((Ns_b, Nphi, N_ifaces, 4, 4), dtype=Complex)
    S12_iface = np.zeros_like(S11_iface)
    S21_iface = np.zeros_like(S11_iface)
    S22_iface = np.zeros_like(S11_iface)

    R_top_arr  = np.zeros((Ns_b, Nphi, 4, 4), dtype=Complex)
    R_back_arr = np.zeros((Ns_b, Nphi, 4, 4), dtype=Complex)

    # electrical row-vectors (always kept)
    Uphi_plus_top  = np.zeros((Ns_b, Nphi, 4), dtype=Complex)
    Uphi_minus_top = np.zeros((Ns_b, Nphi, 4), dtype=Complex)
    q_plus_top     = np.zeros((Ns_b, Nphi, 4), dtype=Complex)
    q_minus_top    = np.zeros((Ns_b, Nphi, 4), dtype=Complex)

    # If we want B,f we MUST have the full top mode matrices available (at least temporarily)
    need_full_top_temporarily = bool(store_top_full_modes or store_top_Bf)

    if need_full_top_temporarily:
        Uplus_top  = np.zeros((Ns_b, Nphi, 4, 4), dtype=Complex)
        Pplus_top  = np.zeros_like(Uplus_top)
        Uminus_top = np.zeros_like(Uplus_top)
        Pminus_top = np.zeros_like(Uplus_top)
        store_flag = np.int8(1)
    else:
        Uplus_top  = np.zeros((1, 1, 1, 1), dtype=Complex)
        Pplus_top  = np.zeros((1, 1, 1, 1), dtype=Complex)
        Uminus_top = np.zeros((1, 1, 1, 1), dtype=Complex)
        Pminus_top = np.zeros((1, 1, 1, 1), dtype=Complex)
        store_flag = np.int8(0)

    # If we want B,f stored, allocate them (they are much smaller than storing full modes in the blocks)
    if store_top_Bf:
        B_top_arr = np.zeros((Ns_b, Nphi, 4, 4), dtype=Complex)
        f_top_arr = np.zeros((Ns_b, Nphi, 4),    dtype=Complex)
    else:
        B_top_arr = np.zeros((1, 1, 1, 1), dtype=Complex)
        f_top_arr = np.zeros((1, 1, 1),    dtype=Complex)

    # work ping-pong (unchanged)
    Uplus_A  = np.zeros((Ns_b, Nphi, 4, 4), dtype=Complex)
    Pplus_A  = np.zeros_like(Uplus_A)
    Uminus_A = np.zeros_like(Uplus_A)
    Pminus_A = np.zeros_like(Uplus_A)
    Swp_A    = np.zeros((Ns_b, Nphi, 4), dtype=Complex)
    Swm_A    = np.zeros_like(Swp_A)

    Uplus_B  = np.zeros((Ns_b, Nphi, 4, 4), dtype=Complex)
    Pplus_B  = np.zeros_like(Uplus_B)
    Uminus_B = np.zeros_like(Uplus_B)
    Pminus_B = np.zeros_like(Uplus_B)
    Swp_B    = np.zeros((Ns_b, Nphi, 4), dtype=Complex)
    Swm_B    = np.zeros_like(Swp_B)

    # heavy lifting in njit kernel (UNCHANGED signature: only store_flag)
    build_block_caches_streaming_fill_kernel(
        kind, rho, bottom_elec_flag,
        C4, e, eps,
        s_vals_b, cos_phi, sin_phi,
        tol_im, tol_s,

        mech_bc_flag_topwall,
        np.int8(top_elec_flag),
        Complex(top_Z_elec),
        np.int64(top_elec_passthrough_idx),

        mech_bc_flag_backwall,
        store_flag,

        Swplus_layers, Swminus_layers,
        S11_iface, S12_iface, S21_iface, S22_iface,
        R_top_arr,
        R_back_arr,

        Uphi_plus_top, Uphi_minus_top, q_plus_top, q_minus_top,
        Uplus_top, Pplus_top, Uminus_top, Pminus_top,

        Uplus_A, Pplus_A, Uminus_A, Pminus_A, Swp_A, Swm_A,
        Uplus_B, Pplus_B, Uminus_B, Pminus_B, Swp_B, Swm_B,
    )

    # ---- NEW: build B_top and f_top right here, then we can drop full modes from the returned tuple ----
    if store_top_Bf:
        build_Btop_ftop_from_fulltop_numba(
            Uplus_top, Pplus_top, Uminus_top, Pminus_top,
            mech_bc_flag_topwall,
            int(top_elec_passthrough_idx),
            B_top_arr, f_top_arr
        )

    # Return ordering:
    if store_top_full_modes:
        # include everything (note: includes B,f if store_top_Bf=True)
        if store_top_Bf:
            return (Swplus_layers, Swminus_layers,
                    S11_iface, S12_iface, S21_iface, S22_iface,
                    R_top_arr, R_back_arr,
                    B_top_arr, f_top_arr,
                    Uphi_plus_top, Uphi_minus_top, q_plus_top, q_minus_top,
                    Uplus_top, Pplus_top, Uminus_top, Pminus_top)
        else:
            return (Swplus_layers, Swminus_layers,
                    S11_iface, S12_iface, S21_iface, S22_iface,
                    R_top_arr, R_back_arr,
                    Uphi_plus_top, Uphi_minus_top, q_plus_top, q_minus_top,
                    Uplus_top, Pplus_top, Uminus_top, Pminus_top)

    # default: no full modes stored
    if store_top_Bf:
        return (Swplus_layers, Swminus_layers,
                S11_iface, S12_iface, S21_iface, S22_iface,
                R_top_arr, R_back_arr,
                B_top_arr, f_top_arr,
                Uphi_plus_top, Uphi_minus_top, q_plus_top, q_minus_top)
    else:
        return (Swplus_layers, Swminus_layers,
                S11_iface, S12_iface, S21_iface, S22_iface,
                R_top_arr, R_back_arr,
                Uphi_plus_top, Uphi_minus_top, q_plus_top, q_minus_top)


from numba import njit
import numpy as np



@njit
def build_block_caches_streaming_fill_kernel(
    # packed layers (Numba-friendly)
    kind,                 # (N_layers,) int8  (0 elastic, 1 piezo)
    rho,                  # (N_layers,) float64
    bottom_elec_flag,     # (N_layers,) int8  (0 short, 1 open, 2 Z, 3 passthrough)
    C4, e, eps,           # tensors (N_layers, ...)

    # block grid
    s_vals_b,             # (Ns_b,)
    cos_phi, sin_phi,     # (Nphi,)
    tol_im, tol_s,

    # boundary flags
    mech_bc_flag_topwall,     # 0 free, 1 clamped
    top_elec_flag,            # 0 short, 1 open, 2 Z (NOT cacheable), 3 passthrough
    top_Z_elec,               # complex (only used if top_elec_flag==2)
    top_elec_passthrough_idx, # usually 3

    mech_bc_flag_backwall,    # 0 free, 1 clamped

    store_top_full_modes_flag,  # int8: 0/1

    # outputs
    Swplus_layers, Swminus_layers,     # (Ns_b, Nphi, N_layers, 4)
    S11_iface, S12_iface, S21_iface, S22_iface,  # (Ns_b, Nphi, N_ifaces, 4, 4) bottom-first
    R_top_arr,                         # (Ns_b, Nphi, 4, 4)
    R_back_arr,                        # (Ns_b, Nphi, 4, 4)

    Uphi_plus_top, Uphi_minus_top,     # (Ns_b, Nphi, 4)
    q_plus_top, q_minus_top,           # (Ns_b, Nphi, 4)

    # optional top full modes
    Uplus_top, Pplus_top, Uminus_top, Pminus_top,  # (Ns_b, Nphi, 4, 4) or dummy

    # work (ping-pong)
    Uplus_A, Pplus_A, Uminus_A, Pminus_A, Swp_A, Swm_A,   # (Ns_b, Nphi, 4, 4) and (Ns_b, Nphi, 4)
    Uplus_B, Pplus_B, Uminus_B, Pminus_B, Swp_B, Swm_B,
):
    Ns_b     = s_vals_b.shape[0]
    Nphi     = cos_phi.shape[0]
    N_layers = kind.shape[0]
    N_ifaces = N_layers - 1

    # ----------------------------
    # layer 0 -> work A
    # ----------------------------
    fill_layer_modes_kernel(
        kind[0],
        C4[0], e[0], eps[0], rho[0],
        s_vals_b, cos_phi, sin_phi,
        tol_im, tol_s,
        Uplus_A, Pplus_A, Uminus_A, Pminus_A,
        Swp_A, Swm_A
    )

    # store slownesses + top electrical rows
    for i_s in range(Ns_b):
        for j_phi in range(Nphi):
            for m in range(4):
                Swplus_layers[i_s, j_phi, 0, m]  = Swp_A[i_s, j_phi, m]
                Swminus_layers[i_s, j_phi, 0, m] = Swm_A[i_s, j_phi, m]

                Uphi_plus_top[i_s, j_phi, m]  = Uplus_A[i_s, j_phi, 3, m]
                Uphi_minus_top[i_s, j_phi, m] = Uminus_A[i_s, j_phi, 3, m]
                q_plus_top[i_s, j_phi, m]     = Pplus_A[i_s, j_phi, 3, m]
                q_minus_top[i_s, j_phi, m]    = Pminus_A[i_s, j_phi, 3, m]

    # optionally store full top modes
    if store_top_full_modes_flag == 1:
        for i_s in range(Ns_b):
            for j_phi in range(Nphi):
                for i in range(4):
                    for j in range(4):
                        Uplus_top[i_s, j_phi, i, j]  = Uplus_A[i_s, j_phi, i, j]
                        Pplus_top[i_s, j_phi, i, j]  = Pplus_A[i_s, j_phi, i, j]
                        Uminus_top[i_s, j_phi, i, j] = Uminus_A[i_s, j_phi, i, j]
                        Pminus_top[i_s, j_phi, i, j] = Pminus_A[i_s, j_phi, i, j]

    # ----------------------------
    # PHYSICAL TOP-WALL reflection on layer 0 (still in A)
    # ----------------------------
    top_kind = kind[0]
    if top_kind == 1:
        n_fill_top     = 4
        piezo_flag_top = 0
        elec_flag_top  = top_elec_flag

        # NOTE: elec_flag==2 needs omega2 -> not cacheable here
        if elec_flag_top == 2:
            raise ValueError("top_elec_flag==2 (impedance load) is omega-dependent; do it per-omega, not in caches.")
    else:
        n_fill_top     = 3
        piezo_flag_top = 1
        elec_flag_top  = 0  # ignored for n_fill=3

    build_reflection_grid_kernel(
        Uplus_A, Pplus_A, Uminus_A, Pminus_A,
        n_fill_top,
        1,                      # incident_side_flag: 1 top (− incident)
        mech_bc_flag_topwall,
        elec_flag_top,
        top_Z_elec,             # unused unless elec_flag_top==2 (disallowed here)
        0.0,                    # omega2 not available at cache-build time
        piezo_flag_top,
        top_elec_passthrough_idx,
        R_top_arr
    )

    # ----------------------------
    # interfaces streaming: (A) with next layer (B)
    # ----------------------------
    for k in range(N_ifaces):
        fill_layer_modes_kernel(
            kind[k+1],
            C4[k+1], e[k+1], eps[k+1], rho[k+1],
            s_vals_b, cos_phi, sin_phi,
            tol_im, tol_s,
            Uplus_B, Pplus_B, Uminus_B, Pminus_B,
            Swp_B, Swm_B
        )

        for i_s in range(Ns_b):
            for j_phi in range(Nphi):
                for m in range(4):
                    Swplus_layers[i_s, j_phi, k+1, m]  = Swp_B[i_s, j_phi, m]
                    Swminus_layers[i_s, j_phi, k+1, m] = Swm_B[i_s, j_phi, m]

        elec1 = bottom_elec_flag[k]
        elec2 = bottom_elec_flag[k+1]

        build_interface_arrays_general_kernel(
            Uplus_A, Pplus_A, Uminus_A, Pminus_A, kind[k],
            Uplus_B, Pplus_B, Uminus_B, Pminus_B, kind[k+1],
            0,        # pp_elec_mode=0 for physical interfaces
            elec1, elec2,
            S11_iface[:, :, k], S12_iface[:, :, k], S21_iface[:, :, k], S22_iface[:, :, k]
        )

        # swap A <-> B
        tmp = Uplus_A;  Uplus_A  = Uplus_B;  Uplus_B  = tmp
        tmp = Pplus_A;  Pplus_A  = Pplus_B;  Pplus_B  = tmp
        tmp = Uminus_A; Uminus_A = Uminus_B; Uminus_B = tmp
        tmp = Pminus_A; Pminus_A = Pminus_B; Pminus_B = tmp
        tmp = Swp_A;    Swp_A    = Swp_B;    Swp_B    = tmp
        tmp = Swm_A;    Swm_A    = Swm_B;    Swm_B    = tmp

    # ----------------------------
    # backwall reflection on last layer (now in A)
    # ----------------------------
    last_kind = kind[N_layers - 1]
    if last_kind == 1:
        n_fill_back     = 4
        piezo_flag_back = 0
        elec_flag_back  = bottom_elec_flag[N_layers - 1]
        if elec_flag_back == 2:
            raise ValueError("backwall elec_flag==2 needs omega2; do per-omega, not in caches.")
    else:
        n_fill_back     = 3
        piezo_flag_back = 1
        elec_flag_back  = 0

    build_reflection_grid_kernel(
        Uplus_A, Pplus_A, Uminus_A, Pminus_A,
        n_fill_back,
        0,                    # incident_side_flag: 0 bottom (+ incident)
        mech_bc_flag_backwall,
        elec_flag_back,
        0.0 + 0.0j,
        0.0,
        piezo_flag_back,
        3,
        R_back_arr
    )


@njit(parallel=True)
def build_reflection_grid_kernel(
    Uplus_arr, Pplus_arr, Uminus_arr, Pminus_arr,   # (Ns_b, Nphi, 4, 4)
    n_fill,                                         # 3 or 4
    incident_side_flag,                             # 0 bottom, 1 top
    mech_flag,                                      # 0 free, 1 clamped
    elec_flag, Z_elec, omega2,
    piezo_flag, elec_passthrough_idx,
    R_grid                                          # (Ns_b, Nphi, 4, 4)
):
    Ns_b = Uplus_arr.shape[0]
    Nphi = Uplus_arr.shape[1]

    for i_s in prange(Ns_b):
        for j_phi in range(Nphi):

            # --- CRITICAL: zero full 4×4 padding every point ---
            for i in range(4):
                for j in range(4):
                    R_grid[i_s, j_phi, i, j] = 0.0 + 0.0j

            # work on the active sub-block only
            Uplus  = Uplus_arr[i_s, j_phi, :n_fill, :n_fill]
            Pplus  = Pplus_arr[i_s, j_phi, :n_fill, :n_fill]
            Uminus = Uminus_arr[i_s, j_phi, :n_fill, :n_fill]
            Pminus = Pminus_arr[i_s, j_phi, :n_fill, :n_fill]

            R_loc = local_reflection_matrix_bc_numba(
                Uplus, Pplus, Uminus, Pminus,
                incident_side_flag,
                mech_flag,
                elec_flag,
                Z_elec,
                omega2,
                piezo_flag,
                elec_passthrough_idx
            )

            for i in range(n_fill):
                for j in range(n_fill):
                    R_grid[i_s, j_phi, i, j] = R_loc[i, j]


@njit(parallel=True)
def fill_layer_modes_kernel(
    kind_flag,          # 1 piezo, 0 elastic
    C4_eff, e_ijk, eps_tensor, rho,
    s_vals_b, cos_phi, sin_phi,
    tol_im, tol_s,
    Uplus_arr, Pplus_arr,
    Uminus_arr, Pminus_arr,
    Swplus_arr, Swminus_arr,
):
    """
    Fills *work arrays* of fixed size 4×4 and 4 slowness slots.

    Required shapes:
      Uplus_arr.shape  == (Ns_b, Nphi, 4, 4)
      Pplus_arr.shape  == (Ns_b, Nphi, 4, 4)
      Swplus_arr.shape == (Ns_b, Nphi, 4)

    Elastic layers only populate the leading 3×3 / first 3 slownesses,
    and explicitly zero the padding row/col/slot so reuse is safe.
    """
    Ns_b = s_vals_b.shape[0]
    Nphi = cos_phi.shape[0]

    for i_s in prange(Ns_b):
        s = s_vals_b[i_s]
        for j_phi in range(Nphi):
            sx = s * cos_phi[j_phi]
            sy = s * sin_phi[j_phi]

            if kind_flag == 1:
                Nmat = stroh_generator_piezo_slow_numba(
                    C4_eff, e_ijk, eps_tensor, rho, sx, sy, tol_s
                )
                Uplus, Pplus, Uminus, Pminus, Swp_mat, Swm_mat = \
                    slow_modes_robust_numba(N=Nmat, tol_im=tol_im, tol_static=-1.0)

                # Copy full 4×4 + 4 slownesses
                for i in range(4):
                    for j in range(4):
                        Uplus_arr[i_s, j_phi, i, j]  = Uplus[i, j]
                        Pplus_arr[i_s, j_phi, i, j]  = Pplus[i, j]
                        Uminus_arr[i_s, j_phi, i, j] = Uminus[i, j]
                        Pminus_arr[i_s, j_phi, i, j] = Pminus[i, j]
                for j in range(4):
                    Swplus_arr[i_s, j_phi, j]  = Swp_mat[j, j]
                    Swminus_arr[i_s, j_phi, j] = Swm_mat[j, j]

            else:
                Nmat = stroh_generator_elastic_slow_numba(
                    C4_eff, rho, sx, sy, tol_s
                )
                Uplus, Pplus, Uminus, Pminus, Swp_mat, Swm_mat = \
                    slow_modes_robust_numba(N=Nmat, tol_im=tol_im, tol_static=-1.0)

                # Copy 3×3 + 3 slownesses
                for i in range(3):
                    for j in range(3):
                        Uplus_arr[i_s, j_phi, i, j]  = Uplus[i, j]
                        Pplus_arr[i_s, j_phi, i, j]  = Pplus[i, j]
                        Uminus_arr[i_s, j_phi, i, j] = Uminus[i, j]
                        Pminus_arr[i_s, j_phi, i, j] = Pminus[i, j]
                for j in range(3):
                    Swplus_arr[i_s, j_phi, j]  = Swp_mat[j, j]
                    Swminus_arr[i_s, j_phi, j] = Swm_mat[j, j]

                # Zero padding row/col 3 + padding slowness (this is the key!)
                for k in range(4):
                    Uplus_arr[i_s, j_phi, 3, k]  = 0.0 + 0.0j
                    Uplus_arr[i_s, j_phi, k, 3]  = 0.0 + 0.0j
                    Pplus_arr[i_s, j_phi, 3, k]  = 0.0 + 0.0j
                    Pplus_arr[i_s, j_phi, k, 3]  = 0.0 + 0.0j

                    Uminus_arr[i_s, j_phi, 3, k] = 0.0 + 0.0j
                    Uminus_arr[i_s, j_phi, k, 3] = 0.0 + 0.0j
                    Pminus_arr[i_s, j_phi, 3, k] = 0.0 + 0.0j
                    Pminus_arr[i_s, j_phi, k, 3] = 0.0 + 0.0j

                Swplus_arr[i_s, j_phi, 3]  = 0.0 + 0.0j
                Swminus_arr[i_s, j_phi, 3] = 0.0 + 0.0j




@njit(parallel=True)
def build_interface_arrays_general_kernel(
    U1plus, P1plus, U1minus, P1minus, kind1,
    U2plus, P2plus, U2minus, P2minus, kind2,
    pp_elec_mode, elec_flag_1, elec_flag_2,
    S11, S12, S21, S22
):
    Ns_b = U1plus.shape[0]
    Nphi = U1plus.shape[1]

    for i_s in prange(Ns_b):
        for j_phi in range(Nphi):

            # --- CRITICAL: zero full 4×4 padding every point ---
            for i in range(4):
                for j in range(4):
                    S11[i_s, j_phi, i, j] = 0.0 + 0.0j
                    S12[i_s, j_phi, i, j] = 0.0 + 0.0j
                    S21[i_s, j_phi, i, j] = 0.0 + 0.0j
                    S22[i_s, j_phi, i, j] = 0.0 + 0.0j

            local_interface_scattering_general_inplace(
                U1plus[i_s, j_phi], P1plus[i_s, j_phi],
                U1minus[i_s, j_phi], P1minus[i_s, j_phi], kind1,
                U2plus[i_s, j_phi], P2plus[i_s, j_phi],
                U2minus[i_s, j_phi], P2minus[i_s, j_phi], kind2,
                pp_elec_mode, elec_flag_1, elec_flag_2,
                S11[i_s, j_phi], S12[i_s, j_phi],
                S21[i_s, j_phi], S22[i_s, j_phi]
            )
