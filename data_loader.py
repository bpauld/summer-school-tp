import numpy as np


def generate_fleet(n, T, seed=42, start_time=12, arrival_center=14, departure_center=36, spread=4):
    """
    Generate a fleet of n electric vehicles over T timesteps.

    The horizon starts at `start_time` (default 12:00 noon) and covers 24 h at
    30-min resolution (T=48 slots).  Vehicles arrive around slot `arrival_center`
    and depart around slot `departure_center`, with ±`spread` slots of randomness.

    All vehicles share the same battery capacity, p_min, and p_max.
    They differ in their initial and required states of charge.

    Returns a dict:
      - start_time: float, hour at which slot 0 starts (e.g. 12.0)
      - arrival:    int array (n,), arrival timestep
      - departure:  int array (n,), departure timestep
      - soc_init:   float array (n,), initial state of charge (kWh)
      - soc_req:    float array (n,), required state of charge at departure (kWh)
      - p_min:      float array (n,), minimum charging power when charging (kW)
      - p_max:      float array (n,), maximum charging power when charging (kW)
      - capacity:   float, battery capacity (kWh)
    """
    rng = np.random.default_rng(seed)
    dt = 0.5  # 30-minute intervals

    arrival   = rng.integers(arrival_center - spread, arrival_center + spread + 1, size=n)
    departure = rng.integers(departure_center - spread, departure_center + spread + 1, size=n)

    # Shared vehicle parameters
    capacity = 300.0   # kWh
    p_min    = 30  * np.ones(n)    # kW
    p_max    = 100  * np.ones(n)   # kW 

    # Per-vehicle SoC: arrive partially charged, need a high SoC to leave
    soc_init = rng.uniform(0.10, 0.40, size=n) * capacity
    soc_req  = rng.uniform(0.70, 0.95, size=n) * capacity

    return {
        "n": n,
        "T": T,
        "dt": dt,
        "start_time": start_time,
        "arrival": arrival,
        "departure": departure,
        "soc_init": soc_init,
        "soc_req": soc_req,
        "p_min": p_min,
        "p_max": p_max,
        "capacity": capacity,
    }


def generate_prices(T, seed=0):
    """
    Generate a realistic day-ahead electricity price signal over T timesteps.

    Returns:
      - prices: float array of shape (T,), price in €/kWh
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 24, T)

    # Base price with two daily peaks (morning and evening)
    base = (
        0.10
        + 0.04 * np.sin(2 * np.pi * (t - 8) / 24)
        + 0.06 * np.exp(-((t - 8) ** 2) / 4)
        + 0.08 * np.exp(-((t - 19) ** 2) / 3)
    )
    noise = rng.normal(0, 0.005, size=T)
    prices = np.clip(base + noise, 0.05, 0.30)
    return prices


def generate_problem(n, T, P_max_factor=0.25, seed=42):
    """
    Generate a full EV charging problem instance.

    Args:
      n:             number of vehicles
      T:             number of timesteps (48 for a 24h horizon at 30-min resolution)
      P_max_factor:  P_max = P_max_factor * n * mean(p_max), controls grid tightness
      seed:          random seed

    Returns a dict with fleet data, prices, and P_max.
    """
    fleet = generate_fleet(n, T, seed=seed)
    prices = generate_prices(T, seed=seed)
    P_max = P_max_factor * n * fleet["p_max"].mean()

    return {**fleet, "prices": prices, "P_max": P_max}
