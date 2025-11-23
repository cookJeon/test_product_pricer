import math
from typing import Sequence, Optional
import numpy as np
from scipy.stats import qmc, norm


def _compute_pv_principal_and_unit_coupon_factor(
    S0_list: Sequence[float],            # 현재 기초자산 가격들 (1~3개)
    sigma_list: Sequence[float],         # 연 변동성들
    q_list: Sequence[float],             # 연 배당수익률들
    r: float,                            # 연 무위험이자율
    T: float,                            # 만기 (년)
    notional: float,                     # 원금
    pay_freq: int,                       # 연 쿠폰 지급 횟수 (예: 4 = 분기)
    condition_levels: Sequence[float],   # 각 자산별 쿠폰 조건 가격 (S_i >= level_i 모두 만족해야 쿠폰)
    corr: Sequence[Sequence[float]],     # 자산 간 상관행렬 (n x n)
    n_paths: int = 131072,               # Monte Carlo 경로 수 (Sobol 쓸 때는 2^k 권장)
    seed: Optional[int] = 123,           # 난수 시드
    use_sobol: bool = True,              # Sobol 사용 여부
    sobol_scramble: bool = True,         # Sobol scramble 여부
) -> tuple[float, float]:
    """
    내부용 함수.
    - 원금 PV (pv_principal)
    - 쿠폰율이 1.0일 때의 쿠폰 PV (unit_coupon_pv_factor)
      => 일반 쿠폰율 c 에 대해 쿠폰 PV = c * unit_coupon_pv_factor

    를 Monte Carlo로 계산한다.
    """

    S0 = np.asarray(S0_list, dtype=float)
    sigma = np.asarray(sigma_list, dtype=float)
    q = np.asarray(q_list, dtype=float)
    cond_levels = np.asarray(condition_levels, dtype=float)

    n_assets = len(S0)
    if not (1 <= n_assets <= 3):
        raise ValueError("기초자산 개수는 1~3개만 허용합니다.")

    if sigma.shape != (n_assets,) or q.shape != (n_assets,):
        raise ValueError("sigma_list, q_list 길이는 S0_list와 같아야 합니다.")

    if cond_levels.shape != (n_assets,):
        raise ValueError("condition_levels 길이는 S0_list와 같아야 합니다.")
    if np.any(cond_levels <= 0.0):
        raise ValueError("condition_levels(쿠폰 조건 가격)는 모두 양수여야 합니다.")

    corr = np.asarray(corr, dtype=float)
    if corr.shape != (n_assets, n_assets):
        raise ValueError("corr는 (n_assets x n_assets) 상관행렬이어야 합니다.")

    # 상관행렬 체크 + Cholesky
    if not np.allclose(corr, corr.T, atol=1e-8):
        raise ValueError("corr는 대칭행렬이어야 합니다.")
    if not np.allclose(np.diag(corr), 1.0, atol=1e-8):
        raise ValueError("corr의 대각 원소는 모두 1이어야 합니다.")
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        raise ValueError("corr가 양의 정부호가 아닙니다. (Cholesky 실패)")

    if T <= 0:
        raise ValueError("T는 양수여야 합니다.")
    if pay_freq <= 0:
        raise ValueError("pay_freq는 양수여야 합니다.")
    if np.any(sigma <= 0):
        raise ValueError("모든 sigma는 양수여야 합니다.")

    # 쿠폰 지급 시점들 (년 단위)
    coupon_times = []
    m_theoretical = int(round(T * pay_freq))
    for k in range(1, m_theoretical + 1):
        t_k = k / pay_freq
        if t_k <= T + 1e-8:
            coupon_times.append(t_k)
    coupon_times = np.array(coupon_times, dtype=float)

    # 원금 PV (항상 보장)
    df_T = math.exp(-r * T)
    pv_principal = notional * df_T

    if len(coupon_times) == 0:
        # 쿠폰 없으면 unit_factor는 0
        return pv_principal, 0.0

    # RNG 세팅 (Sobol 또는 일반 난수)
    if use_sobol:
        sampler = qmc.Sobol(d=n_assets, scramble=sobol_scramble, seed=seed)
        rng = None
    else:
        sampler = None
        rng = np.random.default_rng(seed)

    # 쿠폰율이 1.0일 때의 쿠폰 PV = notional/pay_freq * sum_k [ df_k * P(조건 만족) ]
    # 여기서는 "sum_k df_k * prob_hat_k" 를 먼저 구한 뒤, 마지막에 (notional/pay_freq)를 곱한다.
    sum_df_prob = 0.0

    for t_k in coupon_times:
        # 위험중립 로그노말 파라미터
        drift_t = (r - q - 0.5 * sigma**2) * t_k   # (n_assets,)
        vol_t = sigma * math.sqrt(t_k)             # (n_assets,)

        if use_sobol:
            U = sampler.random(n_paths)            # (n_paths, n_assets)
            Z = norm.ppf(U)
        else:
            Z = rng.standard_normal(size=(n_paths, n_assets))

        # 상관 구조 반영
        X = Z @ L.T                                # (n_paths, n_assets)

        # S(t_k) 시뮬레이션
        S_t = S0 * np.exp(drift_t + vol_t * X)     # (n_paths, n_assets)

        # 모든 자산이 각자의 조건을 만족하는지
        fulfilled = np.all(S_t >= cond_levels, axis=1)   # (n_paths,)
        prob_hat = fulfilled.mean()

        df_k = math.exp(-r * t_k)
        sum_df_prob += df_k * prob_hat

    unit_coupon_pv_factor = (notional / pay_freq) * sum_df_prob

    return pv_principal, unit_coupon_pv_factor


def price_principal_protected_conditional_coupon(
    S0_list: Sequence[float],
    sigma_list: Sequence[float],
    q_list: Sequence[float],
    r: float,
    T: float,
    notional: float,
    coupon_rate: float,
    pay_freq: int,
    condition_levels: Sequence[float],
    corr: Sequence[Sequence[float]],
    n_paths: int = 131072,
    seed: Optional[int] = 123,
    use_sobol: bool = True,
    sobol_scramble: bool = True,
) -> dict:
    """
    주어진 coupon_rate에 대한 상품 가격을 계산.
    Price = pv_principal + coupon_rate * unit_coupon_pv_factor
    """

    pv_principal, unit_factor = _compute_pv_principal_and_unit_coupon_factor(
        S0_list=S0_list,
        sigma_list=sigma_list,
        q_list=q_list,
        r=r,
        T=T,
        notional=notional,
        pay_freq=pay_freq,
        condition_levels=condition_levels,
        corr=corr,
        n_paths=n_paths,
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
    target_price: float,                # 목표 가격 (총 가격, 원 단위)
    S0_list: Sequence[float],
    sigma_list: Sequence[float],
    q_list: Sequence[float],
    r: float,
    T: float,
    notional: float,
    pay_freq: int,
    condition_levels: Sequence[float],
    corr: Sequence[Sequence[float]],
    n_paths: int = 131072,
    seed: Optional[int] = 123,
    use_sobol: bool = True,
    sobol_scramble: bool = True,
) -> dict:
    """
    목표 가격(target_price)을 넣으면,
    그 가격을 맞추기 위한 연 쿠폰율(coupon_rate)을 계산한다.

    수학적으로:
        Price(c) = pv_principal + c * unit_factor
        => c = (target_price - pv_principal) / unit_factor
    """

    pv_principal, unit_factor = _compute_pv_principal_and_unit_coupon_factor(
        S0_list=S0_list,
        sigma_list=sigma_list,
        q_list=q_list,
        r=r,
        T=T,
        notional=notional,
        pay_freq=pay_freq,
        condition_levels=condition_levels,
        corr=corr,
        n_paths=n_paths,
        seed=seed,
        use_sobol=use_sobol,
        sobol_scramble=sobol_scramble,
    )

    if unit_factor <= 0.0:
        raise ValueError("unit_coupon_pv_factor가 0 이하입니다. (쿠폰 구조/조건을 확인하세요.)")

    # target_price가 원금 PV보다 작으면, 이론적으로 쿠폰율이 0 이하가 됩니다.
    # 여기서는 단순히 음수도 허용하고, 결과를 그대로 반환합니다.
    coupon_rate = (target_price - pv_principal) / unit_factor

    # 함께 검증용으로 price도 다시 계산
    price_info = price_principal_protected_conditional_coupon(
        S0_list=S0_list,
        sigma_list=sigma_list,
        q_list=q_list,
        r=r,
        T=T,
        notional=notional,
        coupon_rate=coupon_rate,
        pay_freq=pay_freq,
        condition_levels=condition_levels,
        corr=corr,
        n_paths=n_paths,
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
    # 예제:
    # - 기초자산 2개
    # - 두 자산 모두 조건을 만족해야 쿠폰 지급
    # - 원금은 3년 뒤 100% 보장
    # - 목표 가격을 넣고, 그에 맞는 연 쿠폰율을 역산

    S0_list = [95.0, 105.0]      # 현재 가격 (예: 95, 105)
    sigma_list = [0.30, 0.25]    # 연 변동성 30%, 25%
    q_list = [0.0, 0.0]          # 배당 없음
    r = 0.03                     # 무위험이자율 3%
    T = 3.0                      # 3년 만기
    notional = 100_000_000       # 원금 1억
    pay_freq = 4                 # 분기 쿠폰 (연 4회)

    # 두 자산 모두 조건:
    #   자산1 S1 >= 100
    #   자산2 S2 >= 110
    condition_levels = [100.0, 110.0]

    # 상관행렬 (2자산, 상관 0.5)
    corr = [
        [1.0, 0.5],
        [0.5, 1.0],
    ]

    # 예를 들어, 목표 가격을 98,464,227.94원으로 둔다고 해보자.
    # (이 값은 이전에 coupon_rate=0.06일 때 나왔던 가격에 근접한 값이다.)
    target_price = 98_464_227.94

    result = solve_coupon_for_target_price(
        target_price=target_price,
        S0_list=S0_list,
        sigma_list=sigma_list,
        q_list=q_list,
        r=r,
        T=T,
        notional=notional,
        pay_freq=pay_freq,
        condition_levels=condition_levels,
        corr=corr,
        n_paths=131072,
        seed=123,
        use_sobol=True,
        sobol_scramble=True,
    )

    print("=== 원금보장 AND-조건부 쿠폰 상품 쿠폰율 역산 예제 ===")
    print(f"목표 가격 (Target Price)           : {result['target_price']:,.2f}")
    print(f"계산된 연 쿠폰율 (coupon_rate)      : {result['coupon_rate']:.6f}")
    print(f"원금 PV (PV principal)              : {result['pv_principal']:,.2f}")
    print(f"unit_coupon_pv_factor (쿠폰율=1 기준): {result['unit_coupon_pv_factor']:,.2f}")
    print(f"해당 쿠폰율에서의 가격 (검증용)     : {result['price_check']:,.2f}")
    print(f"해당 쿠폰율에서의 쿠폰 PV           : {result['pv_coupons_at_solution']:,.2f}")
