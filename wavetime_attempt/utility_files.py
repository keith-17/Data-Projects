import numpy as np
from numba import njit, prange
from types import SimpleNamespace
from tqdm.auto import tqdm  # optional
Complex = np.complex128


def voigt6_to_cijkl(C6: np.ndarray, _VOIGHT_PAIRS: list) -> np.ndarray:
    """Symmetric map 6×6 Voigt → C_{ijkl}."""
    C = np.zeros((3,3,3,3), dtype=float)
    for I,(i,j) in enumerate(_VOIGHT_PAIRS):
        for J,(k,l) in enumerate(_VOIGHT_PAIRS):
            C[i,j,k,l] = C6[I,J]
            C[j,i,k,l] = C6[I,J]
            C[i,j,l,k] = C6[I,J]
            C[j,i,l,k] = C6[I,J]
    return C



def e6_voigt_to_eijk(e6: np.ndarray) -> np.ndarray:
    """
    Convert 3x6 Voigt-form piezo tensor e6[k, J]
    (rows: D1,D2,D3; cols: S1..S6 = 11,22,33,23,13,12)
    into a symmetric 3x3x3 tensor e_ijk[i,j,l].

    Mapping:
      J=0 → (i,j)=(0,0)
      J=1 → (1,1)
      J=2 → (2,2)
      J=3 → (1,2) and (2,1)
      J=4 → (0,2) and (2,0)
      J=5 → (0,1) and (1,0)

    We use the *row* index of e6 as the third index l in e_ijk[i,j,l],
    so that e_ijk[2,2,l] = e6[l, J(33)], which gives e_vec[l] = e_{33,l}
    exactly as your build_piezo_projections expects.
    """
    if e6.shape != (3, 6):
        raise ValueError(f"e6 must be 3x6, got {e6.shape}")

    e_ijk = np.zeros((3, 3, 3), dtype=e6.dtype)

    voigt_pairs = [
        (0, 0),  # J=0 -> 11
        (1, 1),  # J=1 -> 22
        (2, 2),  # J=2 -> 33
        (1, 2),  # J=3 -> 23
        (0, 2),  # J=4 -> 13
        (0, 1),  # J=5 -> 12
    ]

    for l in range(3):           # row of e6
        for J, (i, j) in enumerate(voigt_pairs):
            v = e6[l, J]
            if v == 0.0:
                continue
            e_ijk[i, j, l] += v
            if i != j:
                e_ijk[j, i, l] += v  # enforce symmetry in (i,j)

    return e_ijk


def effective_C(slab, omega: float) -> np.ndarray:
    """
    Complex stiffness for e^{-i ω t} convention.

    Kelvin–Voigt:  σ = C:ε  - i ω η:ε   ⇒   C_eff = C - i ω η
    Optional loss-tangent:  C_eff ≈ C * (1 - i tanδ)   (small tanδ)

    Priority:
      1) If slab.eta is provided (4th-rank or 6×6 Voigt), use C - i ω η.
      2) Else if slab.tan_delta is provided (scalar or tensor-like), use C*(1 - i tanδ).
      3) Else return C.
    """
    C = np.asarray(slab.C)
    eta = getattr(slab, "eta", None)
    tan_delta = getattr(slab, "tan_delta", None)

    if eta is not None:
        eta = np.asarray(eta)
        return C - 1j * omega * eta

    if tan_delta is not None:
        td = np.asarray(tan_delta)
        return C * (1.0 - 1j * td)

    return C


def voigt6_to_C4(C6: np.ndarray, _voigt_to_pair: np.ndarray) -> np.ndarray:
    """
    Convert a 6×6 Voigt stiffness matrix C6 into a 3×3×3×3 stiffness tensor C4.
    """
    C6 = np.asarray(C6, dtype=Complex)
    C4 = np.zeros((3, 3, 3, 3), dtype=Complex)

    for I in range(6):
        i, j = _voigt_to_pair[I]
        for J in range(6):
            k, l = _voigt_to_pair[J]
            C4[i, j, k, l] = C6[I, J]

    return C4


@njit
def compute_s_and_khat(sx: float, sy: float, tol_s: float):
    """
    Shared helper: get s_|| and khat for both elastic and piezo generators.

    Returns:
        s   : float, in-plane slowness magnitude (>= tol_s)
        khat: (2,) float64 array, unit in-plane direction
    """
    s_par = np.hypot(sx, sy)

    khat = np.empty(2, dtype=np.float64)

    if s_par <= tol_s:
        # Arbitrary in-plane direction at normal incidence
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
    """
    Numba-safe mechanical projections:

      Q_il = c_{i3l3}
      R_il = c_{i3lα} k̂_α
      T_il = c_{iα lβ} k̂_α k̂_β

    Assumes:
      C4_eff.shape == (3,3,3,3)
      khat.shape   == (2,)
    """
    iz = 2

    Q = np.zeros((3, 3), dtype=Complex)
    R = np.zeros((3, 3), dtype=Complex)
    T = np.zeros((3, 3), dtype=Complex)

    for i in range(3):
        for l in range(3):
            # Q_il = c_{i3l3}
            Q[i, l] = C4_eff[i, iz, l, iz]

            tmpR = 0.0 + 0.0j
            tmpT = 0.0 + 0.0j

            # alpha, beta ∈ {0,1}
            for a in range(2):
                tmpR += C4_eff[i, iz, l, a] * khat[a]
                for b in range(2):
                    tmpT += C4_eff[i, a, l, b] * khat[a] * khat[b]

            R[i, l] = tmpR
            T[i, l] = tmpT

    return Q, R, T


@njit
def build_piezo_projections_numba(e_ijk, eps_kl, khat):
    """
    Numba-safe version of build_piezo_projections.

    e_ijk: (3,3,3)
    eps_kl: (3,3)
    khat: (2,) in-plane unit vector
    """
    iz = 2
    # in-plane indices {0,1}
    alpha_idx0 = 0
    alpha_idx1 = 1

    e_vec = np.zeros(3, dtype=Complex)
    B_vec = np.zeros(3, dtype=Complex)
    W_vec = np.zeros(3, dtype=Complex)
    A_vec = np.zeros(3, dtype=Complex)

    # dielectric projections
    eps_s   = eps_kl[iz, iz]
    gamma_s = 0.0 + 0.0j
    alpha_s = 0.0 + 0.0j

    # gamma_s = eps_{3a} k_a, a∈{0,1}
    gamma_s += eps_kl[iz, alpha_idx0] * khat[alpha_idx0]
    gamma_s += eps_kl[iz, alpha_idx1] * khat[alpha_idx1]

    # alpha_s = eps_{ab} k_a k_b
    alpha_s += eps_kl[alpha_idx0, alpha_idx0] * khat[alpha_idx0] * khat[alpha_idx0]
    alpha_s += eps_kl[alpha_idx0, alpha_idx1] * khat[alpha_idx0] * khat[alpha_idx1]
    alpha_s += eps_kl[alpha_idx1, alpha_idx0] * khat[alpha_idx1] * khat[alpha_idx0]
    alpha_s += eps_kl[alpha_idx1, alpha_idx1] * khat[alpha_idx1] * khat[alpha_idx1]

    for l in range(3):
        # e_l = e_{33l}
        e_vec[l] = e_ijk[iz, iz, l]

        # B,W,A projections
        tmpB = 0.0 + 0.0j
        tmpW = 0.0 + 0.0j
        tmpA = 0.0 + 0.0j

        # a=0
        a = alpha_idx0
        tmpB += e_ijk[iz, a, l] * khat[a]
        tmpW += e_ijk[a, iz, l] * khat[a]

        # contributions to A from (a=0,b=0,1)
        tmpA += e_ijk[a, alpha_idx0, l] * khat[a] * khat[alpha_idx0]
        tmpA += e_ijk[a, alpha_idx1, l] * khat[a] * khat[alpha_idx1]

        # a=1
        a = alpha_idx1
        tmpB += e_ijk[iz, a, l] * khat[a]
        tmpW += e_ijk[a, iz, l] * khat[a]

        # contributions to A from (a=1,b=0,1)
        tmpA += e_ijk[a, alpha_idx0, l] * khat[a] * khat[alpha_idx0]
        tmpA += e_ijk[a, alpha_idx1, l] * khat[a] * khat[alpha_idx1]

        B_vec[l] = tmpB
        W_vec[l] = tmpW
        A_vec[l] = tmpA

    return e_vec, B_vec, W_vec, A_vec, eps_s, gamma_s, alpha_s


@njit
def build_tilded_blocks_numba(Q, R, T,
                              e_vec, B_vec, W_vec, A_vec,
                              eps_s, gamma_s, alpha_s):
    """
    Numba-safe version of build_tilded_blocks.
    """
    Qtilde = np.zeros((4, 4), dtype=Complex)
    Rtilde = np.zeros((4, 4), dtype=Complex)
    Ttilde = np.zeros((4, 4), dtype=Complex)

    # Q̃
    for i in range(3):
        for j in range(3):
            Qtilde[i, j] = Q[i, j]
        Qtilde[i, 3] = e_vec[i]
        Qtilde[3, i] = e_vec[i]
    Qtilde[3, 3] = -eps_s

    # R̃
    for i in range(3):
        for j in range(3):
            Rtilde[i, j] = R[i, j]
        Rtilde[i, 3] = B_vec[i]
        Rtilde[3, i] = W_vec[i]
    Rtilde[3, 3] = -gamma_s

    # T̃
    for i in range(3):
        for j in range(3):
            Ttilde[i, j] = T[i, j]
        Ttilde[i, 3] = A_vec[i]
        Ttilde[3, i] = A_vec[i]
    Ttilde[3, 3] = -alpha_s

    return Qtilde, Rtilde, Ttilde


@njit
def build_piezo_QRT_tilde_numba(C4_eff, e_ijk, eps_kl, khat):
    """
    Numba version of build_piezo_QRT_tilde.
    """
    # 3×3 mechanical Q, R, T from anisotropic elasticity
    Q, R, T = build_mech_QRT_numba(C4_eff, khat)

    # piezo projections
    (e_vec, B_vec, W_vec, A_vec,
     eps_s, gamma_s, alpha_s) = build_piezo_projections_numba(e_ijk, eps_kl, khat)

    # tilded blocks
    Qtilde, Rtilde, Ttilde = build_tilded_blocks_numba(
        Q, R, T,
        e_vec, B_vec, W_vec, A_vec,
        eps_s, gamma_s, alpha_s
    )

    # return also R̃ᵀ as a separate array
    RTtilde = Rtilde.T.copy()
    return Qtilde, Rtilde, RTtilde, Ttilde


@njit
def stroh_generator_elastic_slow_numba(
    C4_eff: np.ndarray,
    rho: float,
    sx: float,
    sy: float,
    tol_s: float = 1e-15,
) -> np.ndarray:
    """
    Numba-safe 6×6 elastic Stroh generator N for the canonical state

        Φ = [ u_x, u_y, u_z,  p_x, p_y, p_z ]ᵀ.

    Slowness form: s_|| = |(sx, sy)|, so N has no explicit ω.
    """
    # 1. Shared slowness + direction
    s, khat = compute_s_and_khat(sx, sy, tol_s)

    # 2. Mechanical Q,R,T (3×3 each)
    # You’ll need a numba-ised version of this:
    #   build_mech_QRT_numba(C4_eff, khat)
    Q, R, T = build_mech_QRT_numba(C4_eff, khat)

    # 3. Inverse and transposes
    Qinv = np.linalg.inv(Q)
    RT = R.T

    # 4. "Mass" term: in slowness form this is just rho * I₃
    rhoM = rho * np.eye(3)

    # 5. Blocks of the 6×6 generator
    #    Φ' = N Φ, with λ = i k_z ⇒ k_z = -i λ
    #
    #    N11 = -i s Q⁻¹ Rᵀ
    #    N12 = -Q⁻¹
    #    N21 = ρ I₃ + s² (R Q⁻¹ Rᵀ - T)
    #    N22 = -i s R Q⁻¹
    #
    N11 = -1j * s * (Qinv @ RT)
    N12 = -Qinv
    N21 = rhoM + (s * s) * (R @ Qinv @ RT - T)
    N22 = -1j * s * (R @ Qinv)

    # 6. Assemble N (must be complex!)
    N = np.zeros((6, 6), dtype=Complex)

    N[0:3, 0:3] = N11
    N[0:3, 3:6] = N12
    N[3:6, 0:3] = N21
    N[3:6, 3:6] = N22

    return N


@njit
def stroh_generator_piezo_slow_numba(C4_eff,
                                     e_ijk,
                                     eps_tensor,
                                     rho,
                                     sx,
                                     sy,
                                     tol_s):
    """
    Numba-safe piezoelectric Stroh generator in slowness form.

    State Φ = [u_x,u_y,u_z, Φ,  p_x,p_y,p_z, q]^T.

    Returns N (8×8) such that Φ' = N Φ, with slowness s_|| = |(sx,sy)|.
    """
    # in-plane slowness magnitude
    s_par = np.hypot(sx, sy)

    if not np.isfinite(s_par):
        print("stroh: non-finite s_par for sx,sy =", sx, sy)

    if s_par < tol_s:
        # choose arbitrary in-plane direction
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

    # Build 4×4 Q̃, R̃, R̃ᵀ, T̃
    Qtil, Rtil, RTtil, Ttil = build_piezo_QRT_tilde_numba(
        C4_eff, e_ijk, eps_tensor, khat
    )

    # # Basic nan/inf checks on the blocks
    # if not np.isfinite(Qtil).all():
    #     print("stroh: Qtil non-finite at sx,sy =", sx, sy,
    #           "khat =", khat)
    # if not np.isfinite(Rtil).all():
    #     print("stroh: Rtil non-finite at sx,sy =", sx, sy,
    #           "khat =", khat)
    # if not np.isfinite(Ttil).all():
    #     print("stroh: Ttil non-finite at sx,sy =", sx, sy,
    #           "khat =", khat)

    # Determinant check before inversion
    # detQ = np.linalg.det(Qtil)
    # if not np.isfinite(detQ):
    #     print("stroh: det(Qtil) non-finite =", detQ,
    #           "at sx,sy =", sx, sy, "khat =", khat)
    # elif abs(detQ) < 1e-20:
    #     print("stroh: det(Qtil) very small =", detQ,
    #           "at sx,sy =", sx, sy, "khat =", khat)

    # invert Q̃
    Qinv = np.linalg.inv(Qtil)

    # inertia matrix on mechanical components
    rhoM = np.zeros((4, 4), dtype=Complex)
    for i in range(3):
        rhoM[i, i] = rho

    s = s_eff
    s2 = s * s

    # Stroh blocks in slowness normalisation:
    # N11 = -i s Q̃^{-1} R̃ᵀ
    N11 = -1j * s * (Qinv @ RTtil)
    # N12 = - Q̃^{-1}
    N12 = -Qinv
    # N21 = rhoM + s^2 (R̃ Q̃^{-1} R̃ᵀ - T̃)
    tmp = Rtil @ Qinv @ RTtil
    for i in range(4):
        for j in range(4):
            tmp[i, j] = tmp[i, j] - Ttil[i, j]
    N21 = rhoM.copy()
    for i in range(4):
        for j in range(4):
            N21[i, j] = N21[i, j] + s2 * tmp[i, j]
    # N22 = -i s R̃ Q̃^{-1}
    N22 = -1j * s * (Rtil @ Qinv)

    # assemble full 8×8
    N = np.zeros((8, 8), dtype=Complex)

    for i in range(4):
        for j in range(4):
            N[i, j]       = N11[i, j]
            N[i, j + 4]   = N12[i, j]
            N[i + 4, j]   = N21[i, j]
            N[i + 4, j+4] = N22[i, j]

    # # Final sanity check: if this blows, you know N went bad, not Q
    # if not np.isfinite(N).all():
    #     print("stroh: N has non-finite entries at sx,sy =", sx, sy,
    #           "khat =", khat, "detQ =", detQ)

    return N




@njit
def slow_modes_robust_numba(N, tol_im=1e-13, tol_static=-1.0):
    """
    Numba-safe version of slow_modes_robust.

    Assumes:
        Φ'(z) = N Φ(z),
        Φ ~ exp(i k_z z)  ⇒  eigenvalues λ = i k_z  ⇒  k_z = -i λ.

    Input
    -----
    N : (2n, 2n) complex array
        Stroh generator (mechanical: n=3, piezo: n=4, etc.)
    tol_im : float
        Imaginary-part threshold for classifying decaying/growing.
    tol_static : float
        If <= 0, it is replaced by 10*tol_im.
        Modes with |k_z| < tol_static are treated as "static" and
        assigned later to fill out an exact n/n split.

    Returns
    -------
    Uplus, Pplus, Uminus, Pminus : (n, n) complex
        Mode matrices (upper half = U, lower half = P).
    Slplus, Slminus              : (n, n) complex
        Diagonal matrices of slownesses s_z = k_z/ω (here just k_z,
        since you already folded ω into N or not as you prefer).
    """
    # Eigen-decomposition
    vals, vecs = np.linalg.eig(N)
    sz = -1j * vals  # k_z = -i λ

    n_total = N.shape[0]
    if n_total % 2 != 0:
        # N must be 2n×2n
        raise ValueError("slow_modes_robust_numba: N must be 2n×2n.")

    n = n_total // 2

    # Static threshold
    if tol_static <= 0.0:
        tol_static = 10.0 * tol_im

    # Arrays to hold indices
    plus_idx   = np.empty(2 * n, dtype=np.int64)
    minus_idx  = np.empty(2 * n, dtype=np.int64)
    static_idx = np.empty(2 * n, dtype=np.int64)

    c_plus = 0
    c_minus = 0
    c_static = 0

    # ----------------------------------------------------------
    # 1) First pass: classify as +, -, or static
    # ----------------------------------------------------------
    for j in range(2 * n):
        sz_j = sz[j]
        abs_k = np.abs(sz_j)

        if abs_k < tol_static:
            # "Static" / electrostatic mode
            static_idx[c_static] = j
            c_static += 1
            continue

        imag_k = np.imag(sz_j)
        real_k = np.real(sz_j)

        # sign classification
        s = 0
        if imag_k > tol_im:
            s = +1
        elif imag_k < -tol_im:
            s = -1
        else:
            # nearly propagating; fallback on sign of Re k_z
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

    # At this point:
    #   plus_idx[0:c_plus], minus_idx[0:c_minus], static_idx[0:c_static]
    # partition {0,...,2n-1}.

    # ----------------------------------------------------------
    # 2) Use static modes to get exact n/n split if possible
    # ----------------------------------------------------------
    need_plus  = n - c_plus
    need_minus = n - c_minus

    # Check if we can fill plus/minus from statics
    if (need_plus < 0) or (need_minus < 0) or (need_plus + need_minus != c_static):
        # Fallback: sort by Im(k) and take top/bottom n
        Imk = np.imag(sz)
        order = np.argsort(-Imk)  # descending Im(k_z)

        plus_final  = order[0:n]
        minus_final = order[2*n - n:2*n]
    else:
        # Fill from static pool
        # Give first need_plus statics to plus, remainder to minus
        for k in range(need_plus):
            plus_idx[c_plus + k] = static_idx[k]
        for k in range(need_minus):
            minus_idx[c_minus + k] = static_idx[need_plus + k]

        # Now we should have exactly n entries in each
        plus_all  = plus_idx[0:n]
        minus_all = minus_idx[0:n]

        # Sort for consistency:
        #   plus: descending Im(k_z)
        #   minus: ascending Im(k_z)
        Imk = np.imag(sz)

        # plus
        tmp_plus = np.empty(n, dtype=np.int64)
        for i in range(n):
            tmp_plus[i] = plus_all[i]
        order_plus = np.argsort(-Imk[tmp_plus])
        plus_final = np.empty(n, dtype=np.int64)
        for i in range(n):
            plus_final[i] = tmp_plus[order_plus[i]]

        # minus
        tmp_minus = np.empty(n, dtype=np.int64)
        for i in range(n):
            tmp_minus[i] = minus_all[i]
        order_minus = np.argsort(Imk[tmp_minus])
        minus_final = np.empty(n, dtype=np.int64)
        for i in range(n):
            minus_final[i] = tmp_minus[order_minus[i]]

    # ----------------------------------------------------------
    # 3) Build mode matrices
    # ----------------------------------------------------------
    # Vp, Vm: eigenvectors for + and - branches
    Vp = vecs[:, plus_final]
    Vm = vecs[:, minus_final]

    # Split Φ = [Λ; P] with Λ,P ∈ C^n
    Uplus  = Vp[0:n, :]
    Pplus  = Vp[n:, :]
    Uminus = Vm[0:n, :]
    Pminus = Vm[n:, :]

    # Diagonal slowness matrices
    Slplus  = np.zeros((n, n), dtype=Complex)
    Slminus = np.zeros((n, n), dtype=Complex)
    for i in range(n):
        Slplus[i, i]  = sz[plus_final[i]]
        Slminus[i, i] = sz[minus_final[i]]

    return Uplus, Pplus, Uminus, Pminus, Slplus, Slminus




@njit
def local_reflection_matrix_bc_numba(
    Uplus, Pplus, Uminus, Pminus,
    incident_side_flag,    # 0="bottom" (+'incident), 1="top" (- incident)
    mech_flag,             # 0 free (tractions=0), 1 clamped (displacements=0)
    elec_flag,             # 0 short (Phi=0), 1 open (q=0), 2 load (Phi + Z*i*omega^2*q=0), 3 passthrough (b_k=a_k)
    Z_elec,                # complex load impedance (used if elec_flag==2)
    omega2,                # omega^2 (used if elec_flag==2)
    piezo_flag,            # 0 piezo (4 modes typical), 1 purely mechanical (3 modes typical)
    elec_passthrough_idx   # which modal amplitude to pass through when elec_flag==3 (e.g. 3)
):
    """
    Returns R such that b = R @ a.

    For n_modes=4 (piezo modal set), mechanical gives 3 BCs.
    If elec_flag==3, we close the system with b[idx] = a[idx] (no electrical BC imposed).
    """

    n_modes = Uplus.shape[1]

    # Choose incident/outgoing sets
    if incident_side_flag == 0:
        # bottom: incident '+', reflected '-'
        U_in, P_in = Uplus, Pplus
        U_out, P_out = Uminus, Pminus
    elif incident_side_flag == 1:
        # top: incident '-', reflected '+'
        U_in, P_in = Uminus, Pminus
        U_out, P_out = Uplus, Pplus
    else:
        raise ValueError("incident_side_flag must be 0 (bottom) or 1 (top)")

    # Count BCs
    if mech_flag != 0 and mech_flag != 1:
        raise ValueError("mech_flag must be 0 (free) or 1 (clamped)")

    n_bc = 3  # mechanical always contributes 3

    if piezo_flag == 0:
        # We include one extra row to make the system square (typical n_modes=4)
        n_bc += 1
    else:
        # Purely mechanical case: typical n_modes=3; allow n_modes=4 only if elec_flag==3 to close it.
        if n_modes == 3:
            n_bc += 0
        elif n_modes == 4:
            # Need a 4th row, but only meaningful if we're doing passthrough
            if elec_flag != 3:
                raise ValueError("piezo_flag==1 with n_modes==4 requires elec_flag==3 passthrough to close system")
            n_bc += 1
        else:
            raise ValueError("Unsupported n_modes for piezo_flag==1")

    # Allocate BC matrices
    B_a = np.zeros((n_bc, n_modes), dtype=Complex)
    B_b = np.zeros((n_bc, n_modes), dtype=Complex)
    row = 0

    # Mechanical rows
    if mech_flag == 0:
        # Free: tractions vanish -> use P rows 0..2
        for comp in range(3):
            for j in range(n_modes):
                B_a[row, j] = P_in[comp, j]
                B_b[row, j] = P_out[comp, j]
            row += 1
    else:
        # Clamped: displacements vanish -> use U rows 0..2
        for comp in range(3):
            for j in range(n_modes):
                B_a[row, j] = U_in[comp, j]
                B_b[row, j] = U_out[comp, j]
            row += 1

    # Electrical / closure row (only if we need it)
    if row < n_bc:
        # clear row
        for j in range(n_modes):
            B_a[row, j] = 0.0 + 0.0j
            B_b[row, j] = 0.0 + 0.0j

        if elec_flag == 0:
            # Short: Phi = 0
            for j in range(n_modes):
                B_a[row, j] = U_in[3, j]
                B_b[row, j] = U_out[3, j]

        elif elec_flag == 1:
            # Open: q = 0
            for j in range(n_modes):
                B_a[row, j] = P_in[3, j]
                B_b[row, j] = P_out[3, j]

        elif elec_flag == 2:
            # Load/matched: Phi + Z * J = 0, with J = i*omega^2*q under your q=-D/omega convention
            fac = 1j * omega2
            for j in range(n_modes):
                B_a[row, j] = U_in[3, j] + (Z_elec * fac) * P_in[3, j]
                B_b[row, j] = U_out[3, j] + (Z_elec * fac) * P_out[3, j]

        elif elec_flag == 3:
            # Passthrough/no-op on "electrical coordinate": b[idx] = a[idx]
            idx = elec_passthrough_idx
            if idx < 0 or idx >= n_modes:
                raise ValueError("elec_passthrough_idx out of range")
            B_a[row, idx] = -1.0 + 0.0j
            B_b[row, idx] = +1.0 + 0.0j

        else:
            raise ValueError("elec_flag must be 0,1,2,3")

        row += 1

    # Must be square
    if B_b.shape[0] != B_b.shape[1]:
        raise ValueError("BC system not square: check n_modes vs flags")

    # Solve B_b R = -B_a
    R = np.linalg.solve(B_b, -B_a)
    return R


def pack_layers_for_numba(layers, KIND_ELASTIC, KIND_PIEZO):
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


def tan_delta_powerlaw(f_hz: float, *, tan0: float, f0_hz: float, exponent: float,
                       tan_min: float = 0.0, tan_max: float = np.inf) -> float:
    f = float(f_hz)
    val = tan0 * (f / float(f0_hz))**float(exponent)
    return float(np.clip(val, tan_min, tan_max))

def make_log_blocks_from_ratio(f_min, f_max, R_max=1.5):
    """
    Choose the minimal number of log-blocks so that
    each block spans at most factor R_max in frequency.
    """
    # total log span
    logR_total = np.log(f_max / f_min)
    # minimal N_blocks s.t. (f_max/f_min)^(1/N_blocks) <= R_max
    N_blocks = max(1, int(np.ceil(logR_total / np.log(R_max))))

    edges = np.logspace(np.log10(f_min), np.log10(f_max), N_blocks + 1)
    f_ref = np.sqrt(edges[:-1] * edges[1:])
    return edges, f_ref


@njit
def local_interface_scattering_general_inplace(
        U1plus, P1plus, U1minus, P1minus, kind1,
        U2plus, P2plus, U2minus, P2minus, kind2,
        pp_elec_mode,  # only relevant if both piezo
        elec_flag_1,  # used if port 1 is piezo in mixed case (0 short, 1 open)
        elec_flag_2,  # used if port 2 is piezo in mixed case (0 short, 1 open)
        S11_bf, S12_bf, S21_bf, S22_bf
):
    n1 = 4 if kind1 == 1 else 3
    n2 = 4 if kind2 == 1 else 3
    dim = n1 + n2

    # equation count
    n_eq = 6
    if kind1 == 1 and kind2 == 1:
        n_eq += 2
    elif (kind1 == 1) ^ (kind2 == 1):
        n_eq += 1

    if n_eq != dim:
        raise ValueError("Interface system not square: check mode counts/closure.")

    M_b = np.zeros((dim, dim), dtype=Complex)
    M_a = np.zeros((dim, dim), dtype=Complex)

    # --- mechanical continuity (6 rows) ---
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

    # --- electrical closure ---
    if kind1 == 1 and kind2 == 1:
        # piezo–piezo: continuity only
        if pp_elec_mode != 0:
            raise ValueError("For interfaces: pp_elec_mode must be 0 (Phi,q continuity).")

        # Phi continuity
        for j in range(n1):
            M_b[row, j] = U1minus[3, j]
            M_a[row, j] = -U1plus[3, j]
        for j in range(n2):
            col = n1 + j
            M_b[row, col] = -U2plus[3, j]
            M_a[row, col] = U2minus[3, j]
        row += 1

        # q continuity
        for j in range(n1):
            M_b[row, j] = P1minus[3, j]
            M_a[row, j] = -P1plus[3, j]
        for j in range(n2):
            col = n1 + j
            M_b[row, col] = -P2plus[3, j]
            M_a[row, col] = P2minus[3, j]
        row += 1

    elif (kind1 == 1) ^ (kind2 == 1):
        # mixed piezo–elastic: one closure row on the piezo side
        # clear row first
        for j in range(dim):
            M_b[row, j] = 0.0 + 0.0j
            M_a[row, j] = 0.0 + 0.0j

        if kind1 == 1:
            # port1 is piezo, port2 is elastic
            if elec_flag_1 == 0:
                # Phi = 0  -> Phi^- β1 + Phi^+ α1 = 0  (arranged in M_b β = M_a α form)
                for j in range(n1):
                    M_b[row, j] = U1minus[3, j]
                    M_a[row, j] = -U1plus[3, j]
            else:
                # q = 0
                for j in range(n1):
                    M_b[row, j] = P1minus[3, j]
                    M_a[row, j] = -P1plus[3, j]
        else:
            # port2 is piezo, port1 is elastic
            # port2 total is U2minus*α2 + U2plus*β2 ; α2 sits in the n1.. block of M_a, β2 in n1.. block of M_b
            if elec_flag_2 == 0:
                # Phi = 0
                for j in range(n2):
                    col = n1 + j
                    M_b[row, col] = U2plus[3, j]  # β2
                    M_a[row, col] = -U2minus[3, j]  # α2
            else:
                # q = 0
                for j in range(n2):
                    col = n1 + j
                    M_b[row, col] = P2plus[3, j]
                    M_a[row, col] = -P2minus[3, j]

        row += 1

    # S_std = np.linalg.solve(M_b, M_a)

    S_std = np.linalg.solve(M_b, M_a)

    # S11: (n1,n1)  rows 0..n1-1, cols 0..n1-1
    for i in range(n1):
        for j in range(n1):
            S11_bf[i, j] = S_std[i, j]

    # S12: (n1,n2)  rows 0..n1-1, cols n1..n1+n2-1
    for i in range(n1):
        for j in range(n2):
            S12_bf[i, j] = S_std[i, n1 + j]

    # S21: (n2,n1)  rows n1..n1+n2-1, cols 0..n1-1
    for i in range(n2):
        for j in range(n1):
            S21_bf[i, j] = S_std[n1 + i, j]

    # S22: (n2,n2)  rows n1..n1+n2-1, cols n1..n1+n2-1
    for i in range(n2):
        for j in range(n2):
            S22_bf[i, j] = S_std[n1 + i, n1 + j]


@njit
def interface_scattering_general_numba(
    U1plus, P1plus, U1minus, P1minus, piezo_flag_1,
    U2plus, P2plus, U2minus, P2minus, piezo_flag_2,
    pp_elec_mode,   # only used if both piezo: 0 continuity (Phi,q), 1 short/open “closure” on each side
    elec_flag_1,    # 0 short (Phi=0), 1 open (q=0) — used if port1 is piezo and needs closure
    elec_flag_2     # 0 short (Phi=0), 1 open (q=0) — used if port2 is piezo and needs closure
):
    """
    Convenience wrapper: allocates and returns S blocks.

    Returns S11,S12,S21,S22 in the standard 2-port layout:
        [β1]   [S11 S12] [α1]
        [β2] = [S21 S22] [α2]

    Notes:
      - No impedance/radiative terms here (short/open only).
      - Prefer calling the in-place routine in hot loops.
    """
    n1 = 4 if piezo_flag_1 == 1 else 3
    n2 = 4 if piezo_flag_2 == 1 else 3

    # Standard block shapes
    S11 = np.zeros((n1, n1), dtype=Complex)
    S12 = np.zeros((n1, n2), dtype=Complex)
    S21 = np.zeros((n2, n1), dtype=Complex)
    S22 = np.zeros((n2, n2), dtype=Complex)

    # In-place filler (must exist)
    local_interface_scattering_general_inplace(
        U1plus, P1plus, U1minus, P1minus, piezo_flag_1, n1,
        U2plus, P2plus, U2minus, P2minus, piezo_flag_2, n2,
        pp_elec_mode, elec_flag_1, elec_flag_2,
        S11, S12, S21, S22
    )
    return S11, S12, S21, S22


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


def tan_delta_debye(w, tau, Delta, tan_bg=0.0):
    #w = 2*np.pi*np.asarray(f_hz)
    return tan_bg + Delta*(w*tau)/(1 + (w*tau)**2)


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
def build_Btop_ftop_from_fulltop_numba(
    Uplus_top, Pplus_top, Uminus_top, Pminus_top,   # (Ns,Nphi,4,4)
    mech_bc_flag_topwall,                            # 0 free, 1 clamped
    idx_free,                                        # typically 3
    B_top_out,                                       # (Ns,Nphi,4,4)
    f_top_out,                                       # (Ns,Nphi,4)
    diag_eps=1e-30
):
    Ns   = Uplus_top.shape[0]
    Nphi = Uplus_top.shape[1]

    r0, r1, r2 = _rest_indices4(idx_free)

    # Make eps the right complex dtype for this specialisation
    eps_c = (diag_eps + 0.0j) * (Uplus_top[0,0,0,0] / Uplus_top[0,0,0,0])

    for i_s in prange(Ns):
        for j in range(Nphi):

            # Choose mechanical rows (3x4)
            if mech_bc_flag_topwall == 0:
                Mplus0 = Pplus_top[i_s, j, 0, :]
                Mplus1 = Pplus_top[i_s, j, 1, :]
                Mplus2 = Pplus_top[i_s, j, 2, :]
                Mminus0 = Pminus_top[i_s, j, 0, :]
                Mminus1 = Pminus_top[i_s, j, 1, :]
                Mminus2 = Pminus_top[i_s, j, 2, :]
            else:
                Mplus0 = Uplus_top[i_s, j, 0, :]
                Mplus1 = Uplus_top[i_s, j, 1, :]
                Mplus2 = Uplus_top[i_s, j, 2, :]
                Mminus0 = Uminus_top[i_s, j, 0, :]
                Mminus1 = Uminus_top[i_s, j, 1, :]
                Mminus2 = Uminus_top[i_s, j, 2, :]

            # A = Mplus[:, rest] (3x3)
            A = np.empty((3, 3), dtype=Uplus_top.dtype)
            A[0,0] = Mplus0[r0] + eps_c;  A[0,1] = Mplus0[r1];       A[0,2] = Mplus0[r2]
            A[1,0] = Mplus1[r0];         A[1,1] = Mplus1[r1] + eps_c; A[1,2] = Mplus1[r2]
            A[2,0] = Mplus2[r0];         A[2,1] = Mplus2[r1];       A[2,2] = Mplus2[r2] + eps_c

            # RHSB = -Mminus (3x4), then solve A * Brest = RHSB  (store in RHSB)
            RHSB = np.empty((3, 4), dtype=Uplus_top.dtype)
            for k in range(4):
                RHSB[0, k] = -Mminus0[k]
                RHSB[1, k] = -Mminus1[k]
                RHSB[2, k] = -Mminus2[k]
            solve3x3_inplace_multi(A, RHSB, 4)   # RHSB becomes Brest

            # RHSf = -Mplus[:, idx_free], solve A * frest = RHSf  (store in first column of RHSB)
            RHSf = np.empty((3, 1), dtype=Uplus_top.dtype)
            RHSf[0,0] = -Mplus0[idx_free]
            RHSf[1,0] = -Mplus1[idx_free]
            RHSf[2,0] = -Mplus2[idx_free]
            solve3x3_inplace_multi(A, RHSf, 1)
            frest0 = RHSf[0,0]; frest1 = RHSf[1,0]; frest2 = RHSf[2,0]

            # Zero outputs (4x4 and 4)
            for ii in range(4):
                f_top_out[i_s, j, ii] = 0.0 + 0.0j
                for kk in range(4):
                    B_top_out[i_s, j, ii, kk] = 0.0 + 0.0j

            # a[idx_free] = a_free
            f_top_out[i_s, j, idx_free] = 1.0 + 0.0j

            # Fill rest rows
            f_top_out[i_s, j, r0] = frest0
            f_top_out[i_s, j, r1] = frest1
            f_top_out[i_s, j, r2] = frest2

            for kk in range(4):
                B_top_out[i_s, j, r0, kk] = RHSB[0, kk]
                B_top_out[i_s, j, r1, kk] = RHSB[1, kk]
                B_top_out[i_s, j, r2, kk] = RHSB[2, kk]


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


# ------------------------------------------------------------
# THIN WRAPPER: takes your list-of-dicts layers (packs once), then calls the packed version
# ------------------------------------------------------------
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
    KIND_ELASTIC: np.ndarray,
    KIND_PIEZO: np.ndarray,
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
     bottom_elec_flag, _bottom_Z_elec, C4, e, eps, n_modes_layer) = pack_layers_for_numba(layers, KIND_ELASTIC, KIND_PIEZO)

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


@njit
def _zero_nn(A, n):
    for i in range(n):
        for j in range(n):
            A[i, j] = 0.0 + 0.0j

@njit
def _eye_n(A, n):
    _zero_nn(A, n)
    for i in range(n):
        A[i, i] = 1.0 + 0.0j


@njit
def _matmul_nn(A, B, C, n, tmp):
    # C = A @ B, all n×n. tmp unused here but kept for symmetry if you extend.
    for i in range(n):
        for j in range(n):
            s = 0.0 + 0.0j
            for k in range(n):
                s += A[i, k] * B[k, j]
            C[i, j] = s


# ============================================================
# 1) In-place LU solve for n<=4, multiple RHS columns (no alloc)
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
        # (assume your physics keeps this nonsingular; if not, you already have bigger problems)
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
# 2) Load block from reflection (in-place)
# ============================================================
@njit
def S_block_load_from_R_inplace_numba(R, S11, S12, S21, S22, n_act):
    # hard-code max dim used everywhere else
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
# 3) Propagation block from slownesses (in-place, padded-safe)
# ============================================================
@njit
def S_block_propagation_from_slow_inplace_numba(
    omega, loss_ratio, Swplus, Swminus, L,
    n_active, n_max,
    S11, S12, S21, S22
):
    """
    Same structure as your allocating version, but fills in-place.

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
# 4) General Redheffer star product (in-place, n<=4, no alloc)
# ============================================================
@njit
def general_redheffer_star_inplace_numba(
    A11, A12, A21, A22,
    B11, B12, B21, B22,
    n,          # active dimension (<=4)
    # outputs
    O11, O12, O21, O22,
    # work (all 4×4 unless noted)
    W1, W2, W3,               # 4×4 complex
    RHS, SOL,                 # 4×4 complex (RHS, SOL)
    LU, piv                   # LU:4×4 complex, piv:4 int64
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
# 5) Top TEM↔modal block, ELECTRICAL ONLY (in-place, no alloc)
#    Requires you to have pre-eliminated mechanics: a = B_top b + f_top a_free
# ============================================================
@njit
def S_block_top_TEM_piezo_elec_only_inplace_numba(
    B_top, f_top,              # (4,4), (4,)
    phi_plus, phi_minus,        # (4,)  == Uplus[3,:], Uminus[3,:]   (no /Area here)
    q_plus, q_minus,            # (4,)  == Pplus[3,:], Pminus[3,:]
    omega, Area, rootZ,
    # outputs
    S11t, S12t, S21t, S22t      # S11t: (1,1), S12t:(1,4), S21t:(4,1), S22t:(4,4)
):
    """
    Solves ONLY the two electrical equations, assuming mechanics already encoded in (B_top,f_top).

    Uses your existing electrical conventions:
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
# 6) Terminate modal port with reflection R_eff to get Γ_TEM (no mixed-star)
# ============================================================
@njit
def gamma_from_top_and_R_inplace_numba(
    S11t, S12t, S21t, S22t,     # top mixed block (TEM<->modal)
    R_eff,                      # (4,4) effective modal reflection seen at top
    n,                          # we pass 4
    # work
    M, RHSv, SOLv,              # M:(4,4), RHSv:(4,1), SOLv:(4,1)
    LU, piv,
    Wtmp                        # (4,4) scratch to avoid aliasing
):
    """
    Γ = S11t + S12t R_eff (I - S22t R_eff)^{-1} S21t

    Returns Γ as a complex scalar (stored in S11t[0,0] convention).
    """
    # Wtmp = S22t @ R_eff
    _matmul_nn(S22t, R_eff, Wtmp, n, Wtmp)  # OK because _matmul_nn writes Wtmp; last arg is scratch too
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


@njit
def compute_gamma_multilayer_elec_only_inplace_numba(
    omega, loss_ratio, Area, rootZ,
    N_layers,
    L_layers, n_modes_layer,        # (N_layers,)
    Swplus_row, Swminus_row,         # (N_layers, 4)
    S11_iface_row, S12_iface_row, S21_iface_row, S22_iface_row,  # (N_layers-1,4,4)
    R_back_row,                      # (4,4)

    # top: pre-eliminated mechanics + only electrical objects (per (s,phi))
    B_top, f_top,                    # (4,4), (4,)
    phi_plus, phi_minus,             # (4,)
    q_plus, q_minus,                 # (4,)

    # scratch S-blocks
    S11_stack, S12_stack, S21_stack, S22_stack,   # (4,4)
    S11_tmp,   S12_tmp,   S21_tmp,   S22_tmp,     # (4,4)
    S11p, S12p, S21p, S22p,                       # (4,4)

    # scratch for star
    W1, W2, W3, RHS, SOL, LU, piv,                # W*,RHS,SOL,LU:(4,4), piv:(4,)

    # scratch for top termination
    S11t, S12t, S21t, S22t,                       # S11t:(1,1), S12t:(1,4), S21t:(4,1), S22t:(4,4)
    M, RHSv, SOLv,                                # M:(4,4), RHSv,SOLv:(4,1)
    Wtmp                                          # (4,4) extra scratch for termination
):
    n_act = n_modes_layer[N_layers-1]
    n_max=4

    # 0) start from backwall load as a 2-port in modal space
    # expects to write full 4x4 blocks
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


@njit
def simpson_factors_uniform_best(N: int) -> np.ndarray:
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


@njit(parallel=True)
def slowness_integral_numba_block_multilayer_elec_only_fast_inplace_simpson(
    omegas: np.ndarray,                    # (Nω,)
    loss_ratio: np.ndarray,                # (Nω,) per-frequency scaling for Im(slowness)
    s_vals: np.ndarray,                    # (Ns,)
    N_layers: int,
    L_layers: np.ndarray,                  # (N_layers,)
    n_modes_layer: np.ndarray,             # (N_layers,)

    Swplus_layers_arr: np.ndarray,         # (Ns, Nphi, N_layers, 4)
    Swminus_layers_arr: np.ndarray,
    S11_iface_arr: np.ndarray,             # (Ns, Nphi, N_layers-1, 4, 4)
    S12_iface_arr: np.ndarray,
    S21_iface_arr: np.ndarray,
    S22_iface_arr: np.ndarray,
    R_back_arr: np.ndarray,                # (Ns, Nphi, 4, 4)

    B_top_arr: np.ndarray,                 # (Ns, Nphi, 4, 4)
    f_top_arr: np.ndarray,                 # (Ns, Nphi, 4)
    phi_plus_top_arr: np.ndarray,          # (Ns, Nphi, 4)
    phi_minus_top_arr: np.ndarray,         # (Ns, Nphi, 4)
    q_plus_top_arr: np.ndarray,            # (Ns, Nphi, 4)
    q_minus_top_arr: np.ndarray,           # (Ns, Nphi, 4)

    Area: float,
    rootZ: float,
    const_prefac: float,                   # IMPORTANT: keep ds*dphi inside here
    piston: np.ndarray,                    # (Nω, Ns)

    s_w: np.ndarray,                       # (Ns,) dimensionless Simpson node weights

    # output
    I_omega: np.ndarray,                   # (Nω,)

    # -------- scratch, allocated once by caller as (Nω,4,4) etc --------
    S11_stack_s: np.ndarray, S12_stack_s: np.ndarray, S21_stack_s: np.ndarray, S22_stack_s: np.ndarray,
    S11_tmp_s:   np.ndarray, S12_tmp_s:   np.ndarray, S21_tmp_s:   np.ndarray, S22_tmp_s:   np.ndarray,
    S11p_s:      np.ndarray, S12p_s:      np.ndarray, S21p_s:      np.ndarray, S22p_s:      np.ndarray,

    W1_s: np.ndarray, W2_s: np.ndarray, W3_s: np.ndarray, RHS_s: np.ndarray, SOL_s: np.ndarray, LU_s: np.ndarray, piv_s: np.ndarray,

    S11t_s: np.ndarray, S12t_s: np.ndarray, S21t_s: np.ndarray, S22t_s: np.ndarray,
    M_s: np.ndarray, RHSv_s: np.ndarray, SOLv_s: np.ndarray,
    Wtmp_s: np.ndarray,
):
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


def _flatten_block_tuple(x):
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





def integrate_all_blocks_streaming_inplace_elec_only(
    freqs: np.ndarray,
    blocks: list,
    L_layers: np.ndarray,          # (N_layers,)
    a_radius: float,
    Z0: float,
    show_progress_blocks: bool = True,
):
    freqs  = np.asarray(freqs, dtype=np.float64)
    omegas = 2.0 * np.pi * freqs

    Area  = np.pi * a_radius**2
    rootZ = np.sqrt(Z0)

    I_omega  = np.zeros_like(omegas, dtype=Complex)
    N_layers = int(L_layers.shape[0])
    n_max    = 4

    it = range(len(blocks))
    if show_progress_blocks and ("tqdm" in globals()) and (tqdm is not None):
        it = tqdm(it, desc="Integrating blocks")

    for b in it:
        blk = _flatten_block_tuple(blocks[b])
        nblk = len(blk)
        loss_ratio_b = None  # will default to ones if not provided

        # ----------------------------
        # Accept either:
        #  - flat 19-tuple (your "clean" elec-only blocks), OR
        #  - flattened 20-tuple that still contains L_layers inside the block, OR
        #  - fully packed 8/9-tuple (idx, omegas, s, const, piston, L_layers, n_modes, out_tuple)
        # ----------------------------
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

            # ignore L_layers_b; trust the function argument L_layers (but sanity check is useful)
            # if L_layers_b.shape[0] != N_layers: raise ValueError("Block L_layers length mismatch")

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
            # ignore L_layers_b (optional sanity check)

        elif nblk == 20:
            # flattened packed form, with L_layers explicitly present
            (idx_b, omegas_b, s_vals_b, const_prefac_b, piston_b,
             L_layers_b, n_modes_layer,
             Swplus_layers, Swminus_layers,
             S11_iface, S12_iface, S21_iface, S22_iface,
             R_back_arr,
             B_top_arr, f_top_arr,
             phi_plus_top, phi_minus_top, q_plus_top, q_minus_top) = blk
            # ignore L_layers_b (optional sanity check)

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
        Wtmp_s = z44()   # keep only if your njit signature expects it

        I_block = np.zeros(Nωb, dtype=Complex)
        Ns   = s_vals_b.shape[0]
        s_w_b=simpson_factors_uniform_best(Ns)
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


def compress_blocks_fulltop_to_elec_only(
    blocks_full,
    mech_bc_flag_topwall: int,
    top_passthrough_idx: int = 3,
):
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


def rotation_matrix_cartesian_x(theta_deg):
    theta_rad = np.deg2rad(theta_deg)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ], dtype=Complex)


def rotate_tensor_4_rank(C, theta_deg):
    R = rotation_matrix_cartesian_x(theta_deg)
    return np.einsum('ip,jq,kr,ls,pqrs', R, R, R, R, C)


def rotate_tensor_3_rank(E, theta_deg):
    R = rotation_matrix_cartesian_x(theta_deg)
    return np.einsum('ip,jq,kr,pqr', R, R, R, E)


def rotate_tensor_2_rank(Eps, theta_deg):
    R = rotation_matrix_cartesian_x(theta_deg)
    return  np.einsum('ip,jq,pq', R, R, Eps)


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