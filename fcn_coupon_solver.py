import math
from typing import Sequence, Optional
import numpy as np
from scipy.stats import qmc, norm


def _compute_pv_principal_and_unit_coupon_factor(
    S0_list: Sequence[float],
    ref_list: Sequence[float],
    sigma_list: Sequence[float],
    q_list: Sequence[float],
    r: float,
    T: float,
    notional: float,
    pay_freq: int,
    barrier_ratio: float,
    corr: Sequence[Sequence[float]],
    n_paths: int = 100000,
    n_steps_per_year: int = 252,
    seed: Optional[int] = 123,
    use_sobol: bool = True,
    sobol_scramble: bool = True,
) -> tuple[float, float]:
    """
    내부용:
      - pv_principal : FCN 원금부분 PV (낙인/워스트 반영)
      - unit_coupon_pv_factor : 쿠폰율 1.0일 때 쿠폰 PV
        => coupon_rate = c 일 때, 쿠폰 PV = c * unit_coupon_pv_factor
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

    dim = n_assets * n_steps
    if use_sobol:
        sampler = qmc.Sobol(d=dim, scramble=sobol_scramble, seed=seed)
        U = sampler.random(n_paths)
        Z = norm.ppf(U).reshape(n_paths, n_steps, n_assets)
        rng = None
    else:
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal(size=(n_paths, n_steps, n_assets))

    X = Z @ L.T

    S_paths = np.empty((n_paths, n_steps + 1, n_assets), dtype=float)
    S_paths[:, 0, :] = S0

    drift = (r - q - 0.5 * sigma**2) * dt
    vol_dt = sigma * math.sqrt(dt)

    for t in range(1, n_steps + 1):
        S_paths[:, t, :] = S_paths[:, t - 1, :] * np.exp(drift + vol_dt * X[:, t - 1, :])

    rel_paths = S_paths / ref

    min_ratio_path = rel_paths.min(axis=(1, 2))       # (n_paths,)
    knocked_in = (min_ratio_path < barrier_ratio)

    worst_T = rel_paths[:, -1, :].min(axis=1)         # (n_paths,)
    factor_if_KI = np.minimum(1.0, worst_T)
    factor = np.where(knocked_in, factor_if_KI, 1.0)

    principal_payoff = notional * factor
    pv_principal = math.exp(-r * T) * principal_payoff.mean()

    # 쿠폰율 1.0일 때의 쿠폰 PV 계수
    # coupon_rate = 1 일 때, 1회 쿠폰 = notional / pay_freq
    m_theoretical = int(round(T * pay_freq))
    unit_coupon_pv_factor = 0.0
    for k in range(1, m_theoretical + 1):
        t_k = k / pay_freq
        if t_k <= T + 1e-8:
            unit_coupon_pv_factor += (notional / pay_freq) * math.exp(-r * t_k)

    return pv_principal, unit_coupon_pv_factor


def price_fcn_mc(
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
) -> dict:
    """
    주어진 coupon_rate에서 FCN 가격 계산.
    Price(c) = pv_principal + c * unit_coupon_pv_factor
    """

    pv_principal, unit_factor = _compute_pv_principal_and_unit_coupon_factor(
        S0_list=S0_list,
        ref_list=ref_list,
        sigma_list=sigma_list,
        q_list=q_list,
        r=r,
        T=T,
        notional=notional,
        pay_freq=pay_freq,
        barrier_ratio=barrier_ratio,
        corr=corr,
        n_paths=n_paths,
        n_steps_per_year=n_steps_per_year,
        seed=seed,
        use_sobol=use_sobol,
        sobol_scramble=sobol_scramble,
    )

    pv_coupons = coupon_rate * unit_factor
    price = pv_principal + pv_coupons

    return {
        "price": price,
        "pv_principal": pv_principal,
        "pv_coupons": pv_coupons,
        "unit_coupon_pv_factor": unit_factor,
    }


def solve_coupon_for_target_price(
    target_price: float,
    S0_list: Sequence[float],
    ref_list: Sequence[float],
    sigma_list: Sequence[float],
    q_list: Sequence[float],
    r: float,
    T: float,
    notional: float,
    pay_freq: int,
    barrier_ratio: float,
    corr: Sequence[Sequence[float]],
    n_paths: int = 100000,
    n_steps_per_year: int = 252,
    seed: Optional[int] = 123,
    use_sobol: bool = True,
    sobol_scramble: bool = True,
) -> dict:
    """
    목표 가격(target_price)에 맞는 연 쿠폰율 coupon_rate를 역산.
      Price(c) = pv_principal + c * unit_factor
      => c = (target_price - pv_principal) / unit_factor
    """

    pv_principal, unit_factor = _compute_pv_principal_and_unit_coupon_factor(
        S0_list=S0_list,
        ref_list=ref_list,
        sigma_list=sigma_list,
        q_list=q_list,
        r=r,
        T=T,
        notional=notional,
        pay_freq=pay_freq,
        barrier_ratio=barrier_ratio,
        corr=corr,
        n_paths=n_paths,
        n_steps_per_year=n_steps_per_year,
        seed=seed,
        use_sobol=use_sobol,
        sobol_scramble=sobol_scramble,
    )

    if unit_factor <= 0.0:
        raise ValueError("unit_coupon_pv_factor가 0 이하입니다. 구조/파라미터를 확인하세요.")

    coupon_rate = (target_price - pv_principal) / unit_factor

    # 검증용 가격 재계산
    price_info = price_fcn_mc(
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

    return {
        "coupon_rate": coupon_rate,
        "target_price": target_price,
        "pv_principal": pv_principal,
        "unit_coupon_pv_factor": unit_factor,
        "price_check": price_info["price"],
        "pv_coupons_at_solution": price_info["pv_coupons"],
    }


# ------------------ 테스트용 예제 ------------------ #

if __name__ == "__main__":
    S0_list = [95.0, 105.0]
    ref_list = [100.0, 100.0]
    sigma_list = [0.25, 0.30]
    q_list = [0.0, 0.0]
    r = 0.03
    T = 3.0
    notional = 100_000_000
    pay_freq = 4
    barrier_ratio = 0.6

    corr = [
        [1.0, 0.5],
        [0.5, 1.0],
    ]

    # 1) 먼저 임의 쿠폰율로 가격 한 번 찍어보고
    test_coupon = 0.08
    price_info = price_fcn_mc(
        S0_list=S0_list,
        ref_list=ref_list,
        sigma_list=sigma_list,
        q_list=q_list,
        r=r,
        T=T,
        notional=notional,
        coupon_rate=test_coupon,
        pay_freq=pay_freq,
        barrier_ratio=barrier_ratio,
        corr=corr,
        n_paths=131072,
        n_steps_per_year=252,
        seed=123,
        use_sobol=True,
        sobol_scramble=True,
    )

    print("=== FCN 프라이싱 (쿠폰율 8%) 예제 ===")
    print(f"쿠폰율 (input)                : {test_coupon:.4f}")
    print(f"가격 (Price)                 : {price_info['price']:,.2f}")
    print(f"  원금 PV                    : {price_info['pv_principal']:,.2f}")
    print(f"  쿠폰 PV                    : {price_info['pv_coupons']:,.2f}")
    print(f"  unit_coupon_pv_factor      : {price_info['unit_coupon_pv_factor']:,.2f}")

    # 2) 방금 나온 가격을 target_price로 넣고, 그 가격을 만드는 쿠폰율을 다시 역산
    target_price = price_info["price"]

    solve_info = solve_coupon_for_target_price(
        target_price=target_price,
        S0_list=S0_list,
        ref_list=ref_list,
        sigma_list=sigma_list,
        q_list=q_list,
        r=r,
        T=T,
        notional=notional,
        pay_freq=pay_freq,
        barrier_ratio=barrier_ratio,
        corr=corr,
        n_paths=131072,
        n_steps_per_year=252,
        seed=123,
        use_sobol=True,
        sobol_scramble=True,
    )

    print("\n=== FCN 쿠폰율 역산 예제 ===")
    print(f"목표 가격 (Target Price)     : {solve_info['target_price']:,.2f}")
    print(f"역산된 쿠폰율 (coupon_rate)  : {solve_info['coupon_rate']:.6f}")
    print(f"검증용 가격 (price_check)    : {solve_info['price_check']:,.2f}")
    print(f"쿠폰 PV (solution 기준)      : {solve_info['pv_coupons_at_solution']:,.2f}")
