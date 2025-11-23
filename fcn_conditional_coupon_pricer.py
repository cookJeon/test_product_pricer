import math
from typing import Sequence, Optional
import numpy as np
from scipy.stats import qmc, norm


def price_principal_protected_conditional_coupon(
    S0_list: Sequence[float],            # 현재 기초자산 가격들 (1~3개)
    sigma_list: Sequence[float],         # 연 변동성들
    q_list: Sequence[float],             # 연 배당수익률들
    r: float,                            # 연 무위험이자율
    T: float,                            # 만기 (년)
    notional: float,                     # 원금
    coupon_rate: float,                  # 연 쿠폰율 (예: 0.06 = 연 6%)
    pay_freq: int,                       # 연 쿠폰 지급 횟수 (예: 4 = 분기)
    condition_levels: Sequence[float],   # 각 자산별 쿠폰 조건 가격 (S_i >= level_i 모두 만족해야 쿠폰)
    corr: Sequence[Sequence[float]],     # 자산 간 상관행렬 (n x n)
    n_paths: int = 131072,               # Monte Carlo 경로 수 (Sobol 쓸 때는 2^k 권장)
    seed: Optional[int] = 123,           # 난수 시드
    use_sobol: bool = True,              # Sobol 사용 여부
    sobol_scramble: bool = True,         # Sobol scramble 여부
) -> dict:
    """
    원금보장 + 조건부 쿠폰 상품의 현재가를 Monte Carlo로 추정.

    구조 요약:
      - 기초: 최대 3개, 로그노말 GBM, 상관행렬 corr 반영
      - 원금: 만기 T에 항상 notional 상환 (원금 보장)
      - 쿠폰: 연 coupon_rate, pay_freq 회 지급
              각 쿠폰 시점 t_k 마다
                  모든 i에 대해 S_i(t_k) >= condition_levels[i] 이면 해당 회차 쿠폰 지급
                  아니면 0

    반환:
      {
        "price": 총 가격 (원금 PV + 쿠폰 PV),
        "pv_principal": 원금 PV,
        "pv_coupons": 쿠폰 PV
      }
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
    if coupon_rate < 0:
        raise ValueError("coupon_rate는 음수가 될 수 없습니다.")
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

    # 쿠폰 1회 지급액 (조건 충족 시)
    coupon_per_period = notional * coupon_rate / pay_freq

    if coupon_rate == 0.0 or len(coupon_times) == 0:
        return {
            "price": pv_principal,
            "pv_principal": pv_principal,
            "pv_coupons": 0.0,
        }

    # RNG 세팅 (Sobol 또는 일반 난수)
    if use_sobol:
        sampler = qmc.Sobol(d=n_assets, scramble=sobol_scramble, seed=seed)
        rng = None
    else:
        sampler = None
        rng = np.random.default_rng(seed)

    # 쿠폰 PV 누적
    pv_coupons = 0.0

    # 쿠폰 시점마다 "모든 자산이 조건을 만족"할 확률을 MC로 추정
    for t_k in coupon_times:
        # 위험중립하에서 로그노말 분포 파라미터 (각 자산별)
        drift_t = (r - q - 0.5 * sigma**2) * t_k   # (n_assets,)
        vol_t = sigma * math.sqrt(t_k)             # (n_assets,)

        if use_sobol:
            U = sampler.random(n_paths)           # (n_paths, n_assets)
            Z = norm.ppf(U)                       # 표준정규로 변환
        else:
            Z = rng.standard_normal(size=(n_paths, n_assets))

        # 상관 구조 반영
        X = Z @ L.T                                # (n_paths, n_assets)

        # 각 경로에서 S(t_k) 계산
        # S(t_k) = S0 * exp(drift_t + vol_t * X)
        S_t = S0 * np.exp(drift_t + vol_t * X)     # (n_paths, n_assets)

        # 쿠폰 조건 체크: 모든 자산 i에 대해 S_t[:, i] >= cond_levels[i]
        fulfilled = np.all(S_t >= cond_levels, axis=1)   # (n_paths,)

        # 조건 만족 확률 추정
        prob_hat = fulfilled.mean()

        # 해당 회차 쿠폰 PV
        df_k = math.exp(-r * t_k)
        pv_coupons += coupon_per_period * df_k * prob_hat

    price = pv_principal + pv_coupons

    return {
        "price": price,
        "pv_principal": pv_principal,
        "pv_coupons": pv_coupons,
    }


def price_and_delta_principal_protected_conditional_coupon(
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
    bump_rel: float = 1e-3,
) -> dict:
    """
    가격 + 기초자산별 델타를 함께 계산.

    델타:
      Δ_i ≈ (P(S0_i + h) - P(S0_i - h)) / (2h),  h = S0_i * bump_rel

    - up/down 계산 시 동일 seed를 사용해서 공통 난수(Sobol) 공유
    - 원금 PV는 S0에 무관하므로, 델타는 사실상 쿠폰 부분에서만 나옴
    """

    S0 = np.asarray(S0_list, dtype=float)
    n_assets = len(S0)

    # 기준 가격
    base_result = price_principal_protected_conditional_coupon(
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

    base_price = base_result["price"]

    deltas: list[float] = []

    for i in range(n_assets):
        S_up = S0.copy()
        S_dn = S0.copy()

        h = S0[i] * bump_rel
        if h == 0.0:
            h = bump_rel

        S_up[i] = S0[i] + h
        S_dn[i] = max(S0[i] - h, 1e-8)

        # 각 자산별로 seed를 약간씩 다르게, 단 up/down에는 같은 seed 사용
        seed_i = None if seed is None else seed + 100 * (i + 1)

        up_result = price_principal_protected_conditional_coupon(
            S0_list=S_up,
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
            seed=seed_i,
            use_sobol=use_sobol,
            sobol_scramble=sobol_scramble,
        )

        dn_result = price_principal_protected_conditional_coupon(
            S0_list=S_dn,
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
            seed=seed_i,
            use_sobol=use_sobol,
            sobol_scramble=sobol_scramble,
        )

        price_up = up_result["price"]
        price_dn = dn_result["price"]

        delta_i = (price_up - price_dn) / (2.0 * h)
        deltas.append(delta_i)

    return {
        "price": base_price,
        "pv_principal": base_result["pv_principal"],
        "pv_coupons": base_result["pv_coupons"],
        "delta": deltas,
    }


# ------------------ 테스트용 예제 ------------------ #

if __name__ == "__main__":
    # 예제:
    # - 기초자산 2개
    # - 두 자산 모두 각자의 조건 가격 이상일 때만 쿠폰 지급
    # - 원금은 3년 뒤 100% 보장
    # - 델타까지 계산

    S0_list = [95.0, 105.0]      # 현재 가격 (예: 95, 105)
    sigma_list = [0.30, 0.25]    # 연 변동성 30%, 25%
    q_list = [0.0, 0.0]          # 배당 없음
    r = 0.03                     # 무위험이자율 3%
    T = 3.0                      # 3년 만기
    notional = 100_000_000       # 원금 1억
    coupon_rate = 0.06           # 연 6% 쿠폰
    pay_freq = 4                 # 분기 쿠폰 (연 4회)

    # 두 자산 모두 조건: 첫 번째는 100 이상, 두 번째는 110 이상일 때만 쿠폰 지급이라고 가정
    condition_levels = [100.0, 110.0]

    # 상관행렬 (2자산, 상관 0.5)
    corr = [
        [1.0, 0.5],
        [0.5, 1.0],
    ]

    result = price_and_delta_principal_protected_conditional_coupon(
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
        n_paths=131072,      # 2^17, Sobol에서 권장되는 형태
        seed=123,
        use_sobol=True,
        sobol_scramble=True,
        bump_rel=1e-3,
    )

    print("=== 조건부 쿠폰(두 자산 AND) 원금보장 상품 프라이싱 + 델타 예제 ===")
    print(f"총 가격 (Price)         : {result['price']:,.2f}")
    print(f"원금 PV (PV principal)  : {result['pv_principal']:,.2f}")
    print(f"쿠폰 PV (PV coupons)    : {result['pv_coupons']:,.2f}")
    print(f"델타 벡터 (각 기초별 dPrice/dS0): {result['delta']}")
