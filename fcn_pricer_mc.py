import math
from typing import Sequence, Optional
import numpy as np
from scipy.stats import qmc, norm


def price_fcn_mc(
    S0_list: Sequence[float],            # 현재 기초자산 가격들
    ref_list: Sequence[float],           # 기준 레벨(보통 발행시점 S0, 또는 100 등)
    sigma_list: Sequence[float],         # 연 변동성
    q_list: Sequence[float],             # 연 배당 수익률
    r: float,                            # 연 무위험이자율
    T: float,                            # 만기 (년)
    notional: float,                     # 원금
    coupon_rate: float,                  # 연 쿠폰율 (예: 0.08 = 연 8%)
    pay_freq: int,                       # 연 쿠폰 지급 횟수 (예: 4 = 분기)
    barrier_ratio: float,                # 낙인 배리어 비율 (예: 0.6 = 60%)
    corr: Sequence[Sequence[float]],     # 상관행렬 (n_assets x n_assets)
    n_paths: int = 100000,               # Monte Carlo 경로 수
    n_steps_per_year: int = 252,         # 1년당 시뮬레이션 스텝 수 (일 단위면 252 정도)
    seed: Optional[int] = 123,           # 난수 시드
    use_sobol: bool = True,              # Sobol 사용 여부
    sobol_scramble: bool = True,         # Sobol scramble 여부
) -> dict:
    """
    다자산 워스트오프 FCN 가격 계산 (낙인 후 업사이드는 원금 100%로 캡).

    구조:
      - 기초: n개, GBM, 상관행렬 corr
      - 낙인:
          경로 중 어느 시점에서라도 min_i (S_i / ref_i) < barrier_ratio 이면 KI = True
      - 만기 워스트비율:
          worst_T = min_i (S_i(T) / ref_i)
      - 원금 상환:
          KI = False   → 원금 100%
          KI = True:
              worst_T < 1  → 원금 * worst_T (손실)
              worst_T >= 1 → 원금 100% (업사이드는 캡)
      - 쿠폰:
          연 coupon_rate, pay_freq 회 지급 (기초와 무관하게 항상 지급)
    """

    S0 = np.asarray(S0_list, dtype=float)
    ref = np.asarray(ref_list, dtype=float)
    sigma = np.asarray(sigma_list, dtype=float)
    q = np.asarray(q_list, dtype=float)
    corr = np.asarray(corr, dtype=float)

    n_assets = len(S0)
    if not (len(ref) == len(sigma) == len(q) == n_assets):
        raise ValueError("S0_list, ref_list, sigma_list, q_list 길이는 모두 같아야 합니다.")
    if corr.shape != (n_assets, n_assets):
        raise ValueError("corr는 (n_assets x n_assets) 행렬이어야 합니다.")
    if not np.allclose(corr, corr.T, atol=1e-8):
        raise ValueError("corr는 대칭행렬이어야 합니다.")
    if not np.allclose(np.diag(corr), 1.0, atol=1e-8):
        raise ValueError("corr의 대각원소는 모두 1이어야 합니다.")
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        raise ValueError("corr가 양의 정부호가 아닙니다. (Cholesky 실패)")

    if T <= 0:
        raise ValueError("T는 양수여야 합니다.")
    if pay_freq <= 0:
        raise ValueError("pay_freq는 양수여야 합니다.")
    if barrier_ratio <= 0 or barrier_ratio >= 1:
        raise ValueError("barrier_ratio는 (0,1) 범위여야 합니다.")
    if np.any(sigma <= 0):
        raise ValueError("모든 sigma는 양수여야 합니다.")

    # 시뮬레이션 세팅
    n_steps = int(T * n_steps_per_year)
    if n_steps < 1:
        n_steps = 1
    dt = T / n_steps

    # RNG 세팅
    dim = n_assets * n_steps
    if use_sobol:
        sampler = qmc.Sobol(d=dim, scramble=sobol_scramble, seed=seed)
        U = sampler.random(n_paths)                    # (n_paths, dim)
        Z = norm.ppf(U).reshape(n_paths, n_steps, n_assets)
        rng = None
    else:
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal(size=(n_paths, n_steps, n_assets))

    # 상관 구조 반영: (n_paths, n_steps, n_assets)
    X = Z @ L.T

    # 경로 생성
    S_paths = np.empty((n_paths, n_steps + 1, n_assets), dtype=float)
    S_paths[:, 0, :] = S0

    drift = (r - q - 0.5 * sigma**2) * dt
    vol_dt = sigma * math.sqrt(dt)

    for t in range(1, n_steps + 1):
        # S_t = S_{t-1} * exp(drift + vol_dt * X_{t-1})
        S_paths[:, t, :] = S_paths[:, t - 1, :] * np.exp(drift + vol_dt * X[:, t - 1, :])

    # 기준 대비 비율
    rel_paths = S_paths / ref  # shape: (n_paths, n_steps+1, n_assets)

    # 경로 전체에서의 최소 비율 (낙인 체크)
    min_ratio_path = rel_paths.min(axis=(1, 2))          # (n_paths,)
    knocked_in = (min_ratio_path < barrier_ratio)        # (n_paths,)

    # 만기 시점 워스트오프 비율
    worst_T = rel_paths[:, -1, :].min(axis=1)            # (n_paths,)

    # 원금 상환 비율:
    #  - KI = False → 1.0
    #  - KI = True:
    #       worst_T < 1  → worst_T
    #       worst_T >= 1 → 1.0
    factor_if_KI = np.minimum(1.0, worst_T)
    factor = np.where(knocked_in, factor_if_KI, 1.0)

    principal_payoff = notional * factor
    pv_principal = math.exp(-r * T) * principal_payoff.mean()

    # 쿠폰 PV (기초와 무관한 고정 쿠폰)
    coupon_per_period = notional * coupon_rate / pay_freq
    pv_coupons = 0.0
    m_theoretical = int(round(T * pay_freq))
    for k in range(1, m_theoretical + 1):
        t_k = k / pay_freq
        if t_k <= T + 1e-8:
            pv_coupons += coupon_per_period * math.exp(-r * t_k)

    price = pv_principal + pv_coupons

    # coupon_rate = 1.0 일 때의 쿠폰 PV 계수 (쿠폰 찾기용)
    # 1회 쿠폰 = notional * (1 / pay_freq) 일 때의 PV
    unit_coupon_pv_factor = 0.0
    for k in range(1, m_theoretical + 1):
        t_k = k / pay_freq
        if t_k <= T + 1e-8:
            unit_coupon_pv_factor += (notional / pay_freq) * math.exp(-r * t_k)

    return {
        "price": price,
        "pv_principal": pv_principal,
        "pv_coupons": pv_coupons,
        "unit_coupon_pv_factor": unit_coupon_pv_factor,
    }


def price_and_delta_fcn_mc(
    S0_list: Sequence[float],
    ref_list: Sequence[float],
    sigma_list: Sequence[float],
    q_list: Sequence[float],
    r: float,
    T: float,
    notional: float,
    coupon_rate: float,
    pay_freq: int,
    barrier_ratio: float,
    corr: Sequence[Sequence[float]],
    n_paths: int = 100000,
    n_steps_per_year: int = 252,
    seed: Optional[int] = 123,
    use_sobol: bool = True,
    sobol_scramble: bool = True,
    bump_rel: float = 1e-3,
) -> dict:
    """
    FCN 가격 + 기초자산별 델타(dPrice/dS0)를 같이 계산.
    """

    S0 = np.asarray(S0_list, dtype=float)
    n_assets = len(S0)

    # 기준 가격
    base = price_fcn_mc(
        S0_list=S0_list,
        ref_list=ref_list,
        sigma_list=sigma_list,
        q_list=q_list,
        r=r,
        T=T,
        notional=notional,
        coupon_rate=coupon_rate,
        pay_freq=pay_freq,
        barrier_ratio=barrier_ratio,
        corr=corr,
        n_paths=n_paths,
        n_steps_per_year=n_steps_per_year,
        seed=seed,
        use_sobol=use_sobol,
        sobol_scramble=sobol_scramble,
    )

    deltas: list[float] = []

    for i in range(n_assets):
        S_up = S0.copy()
        S_dn = S0.copy()

        h = S0[i] * bump_rel
        if h == 0.0:
            h = bump_rel

        S_up[i] = S0[i] + h
        S_dn[i] = max(S0[i] - h, 1e-8)

        # 각 자산마다 seed를 다르게, 다만 up/down에서는 동일 seed 사용 (공통 난수)
        seed_i = None if seed is None else seed + 100 * (i + 1)

        up = price_fcn_mc(
            S0_list=S_up,
            ref_list=ref_list,
            sigma_list=sigma_list,
            q_list=q_list,
            r=r,
            T=T,
            notional=notional,
            coupon_rate=coupon_rate,
            pay_freq=pay_freq,
            barrier_ratio=barrier_ratio,
            corr=corr,
            n_paths=n_paths,
            n_steps_per_year=n_steps_per_year,
            seed=seed_i,
            use_sobol=use_sobol,
            sobol_scramble=sobol_scramble,
        )

        dn = price_fcn_mc(
            S0_list=S_dn,
            ref_list=ref_list,
            sigma_list=sigma_list,
            q_list=q_list,
            r=r,
            T=T,
            notional=notional,
            coupon_rate=coupon_rate,
            pay_freq=pay_freq,
            barrier_ratio=barrier_ratio,
            corr=corr,
            n_paths=n_paths,
            n_steps_per_year=n_steps_per_year,
            seed=seed_i,
            use_sobol=use_sobol,
            sobol_scramble=sobol_scramble,
        )

        delta_i = (up["price"] - dn["price"]) / (2.0 * h)
        deltas.append(delta_i)

    result = dict(base)
    result["delta"] = deltas
    return result


# ------------------ 테스트용 예제 ------------------ #

if __name__ == "__main__":
    # 예제 FCN 파라미터
    S0_list = [95.0, 105.0]      # 현재 기초가격
    ref_list = [100.0, 100.0]    # 기준 레벨 (발행시점, 또는 100)
    sigma_list = [0.25, 0.30]    # 연 변동성
    q_list = [0.0, 0.0]          # 배당 수익률
    r = 0.03                     # 무위험이자율
    T = 3.0                      # 3년 만기
    notional = 100_000_000       # 원금 1억
    coupon_rate = 0.08           # 연 8% 쿠폰
    pay_freq = 4                 # 분기 지급
    barrier_ratio = 0.6          # 60% 낙인

    corr = [
        [1.0, 0.5],
        [0.5, 1.0],
    ]

    result = price_and_delta_fcn_mc(
        S0_list=S0_list,
        ref_list=ref_list,
        sigma_list=sigma_list,
        q_list=q_list,
        r=r,
        T=T,
        notional=notional,
        coupon_rate=coupon_rate,
        pay_freq=pay_freq,
        barrier_ratio=barrier_ratio,
        corr=corr,
        n_paths=131072,          # Sobol 쓸 때는 2^k 권장
        n_steps_per_year=252,
        seed=123,
        use_sobol=True,
        sobol_scramble=True,
        bump_rel=1e-3,
    )

    print("=== FCN 프라이싱 (낙인 후 업사이드 캡) + 델타 예제 ===")
    print(f"FCN 가격: {result['price']:,.2f}")
    print(f"  원금 PV: {result['pv_principal']:,.2f}")
    print(f"  쿠폰 PV: {result['pv_coupons']:,.2f}")
    print(f"  unit_coupon_pv_factor (쿠폰율=1 기준): {result['unit_coupon_pv_factor']:,.2f}")
    print(f"델타 벡터 (각 기초별 dPrice/dS0): {result['delta']}")
