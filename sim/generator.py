"""Physics-lite vehicle simulation.

Produces the ``truth`` signal matrix that the CAN emitter puts on the bus and the
ingest path decodes back. Deliberately *not* a vehicle dynamics package: it is a
lumped-parameter model good enough that the analytics layer has real structure to
find (a range trend, a city/highway efficiency split, a thermal excursion), and
simple enough to read in one sitting.

Model summary
-------------
Longitudinal : first-order speed tracking toward a mode-dependent target, with
               stops in CITY. Acceleration is the numerical derivative.
Energy       : Wh/km = rolling + aero(v^2) + aux(v) + HVAC(|ambient-setpoint|).
               The aero term makes highway efficiency worse; the aux term makes
               *very* low speed worse. That U-shape is what makes the
               city-vs-highway comparison a real question rather than a constant.
Pack thermal : single-node lumped thermal model.
               C dT/dt = Q_joule(I^2 R) + Q_ambient - Q_coolant(pump, dT)
               A stuck pump removes the coolant term and the pack runs away -
               which is exactly the fault sequence Q4 asks about.
SOC          : coulomb counting from pack current, with a finite pack.

Everything is seeded. Same scenario + same seed => bit-identical output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.scenario import FaultKind, Mode, Scenario

RATE_HZ = 10.0
DT = 1.0 / RATE_HZ


# --- vehicle parameters (a mid-size BEV; all values are model assumptions) ---
MASS_KG = 1900.0
PACK_CAPACITY_KWH = 75.0
PACK_NOMINAL_V = 390.0
PACK_RESISTANCE_OHM = 0.055
PACK_THERMAL_CAPACITY_J_PER_K = 400_000.0  # ~400 kg pack x ~1000 J/kg/K, lumped
HEAT_TRANSFER_AMBIENT_W_PER_K = 12.0
COOLANT_GAIN_W_PER_K = 95.0
COOLANT_MAX_DT_C = 12.0
PUMP_BASELINE_PCT = 18.0   # circulation while the vehicle is on

# Road-load model: F = 0.5*rho*Cd*A*v^2 + Crr*m*g, then P = F*v.
AERO_DRAG_COEFF_N_PER_MPS2 = 0.37   # 0.5 * 1.2 kg/m3 * 0.28 Cd * 2.2 m2
ROLLING_RESISTANCE_COEFF = 0.010
DRIVETRAIN_EFFICIENCY = 0.90
AUX_BASE_W = 350.0                  # compute, lighting, electronics
MAX_HVAC_W = 8000.0
MAX_REGEN_W = 10230.0               # matches VCU_regenPower_W in the vendor DBC
REGEN_W_PER_MS2 = 22_000.0          # ~190 kW/m/s^2 available from the motors

SPEED_TARGETS = {Mode.IDLE: 0.0, Mode.CITY: 32.0, Mode.HIGHWAY: 110.0}

OVERHEAT_WARN_C = 60.0
OVERHEAT_CRITICAL_C = 75.0


@dataclass(frozen=True)
class Event:
    """A ground-truth event. The analytics layer must rediscover these."""

    ts_s: float
    kind: str
    description: str
    severity: str = "info"


@dataclass
class Simulation:
    scenario: Scenario
    t_s: np.ndarray                 # (N,) seconds since start
    truth: dict[str, np.ndarray]    # canonical path -> (N,) values
    events: list[Event]
    modes: np.ndarray               # (N,) Mode enum values, ground truth

    @property
    def n(self) -> int:
        return self.t_s.shape[0]


def _hold_last_moving(values: np.ndarray, speed: np.ndarray,
                      moving_threshold: float = 5.0) -> np.ndarray:
    """Replace stationary samples with the most recent moving sample.

    Leading stationary samples (before the vehicle first moves) take the first
    moving value, so a drive that starts parked does not report a 1300 Wh/km
    consumption figure.
    """
    moving = speed > moving_threshold
    if not moving.any():
        return values.copy()

    out = values.copy()
    last = int(np.flatnonzero(moving)[0])
    first_moving_value = out[last]
    for i in range(out.shape[0]):
        if moving[i]:
            last = i
        else:
            out[i] = out[last] if i > last else first_moving_value
    return out


def _trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Causal moving average - no look-ahead, like a real trip computer."""
    window = max(1, min(window, values.shape[0]))
    csum = np.cumsum(values)
    out = csum / np.arange(1, values.shape[0] + 1)
    out[window:] = (csum[window:] - csum[:-window]) / window
    return out


def _extra_heat(faults, t: float) -> float:
    """Additional heat injected at time t by active faults (W)."""
    heat = 0.0
    for f in faults:
        if f.kind is not FaultKind.BATTERY_OVERHEAT:
            continue
        if f.start_s <= t < f.start_s + f.duration_s:
            # ramps in over the first 20% of the fault window
            span = max(f.duration_s * 0.2, 1e-6)
            frac = min(1.0, (t - f.start_s) / span)
            heat += f.params.get("watts", 9000.0) * frac
    return heat


def _pump_available(faults, t: float) -> bool:
    for f in faults:
        if f.kind is not FaultKind.COOLANT_PUMP_FAILURE:
            continue
        if f.start_s <= t < f.start_s + f.duration_s:
            return False
    return True


def _adas_state(faults, t: float) -> int:
    """1 = engaged, 0 = disengaged. Disengagements are explicitly injected."""
    base = 1
    for f in faults:
        if f.kind is not FaultKind.ADAS_DISENGAGEMENT:
            continue
        if f.start_s <= t < f.start_s + f.duration_s:
            return 0
    return base


def generate(scenario: Scenario, rng_seed_offset: int = 0) -> Simulation:
    rng = np.random.default_rng(scenario.seed + rng_seed_offset)

    n = int(round(scenario.duration_s * RATE_HZ))
    if n < 2:
        raise ValueError("scenario too short; need at least 0.2 s")
    t = np.arange(n) / RATE_HZ

    # ---- driving mode timeline -------------------------------------------
    mode_series = np.full(n, Mode.IDLE.value, dtype=object)
    for start, end, mode in scenario.segment_bounds():
        lo, hi = int(start * RATE_HZ), min(int(end * RATE_HZ), n)
        mode_series[lo:hi] = mode.value

    # ---- speed: first-order tracking toward the mode target --------------
    target = np.array([SPEED_TARGETS[Mode(m)] for m in mode_series])
    # CITY has periodic stops, which is what makes it city-like.
    stop_mask = np.zeros(n, dtype=bool)
    if np.any(mode_series == Mode.CITY.value):
        phase = (t % 55.0)
        stop_mask = (mode_series == Mode.CITY.value) & (phase > 42.0)
        target = np.where(stop_mask, 0.0, target)

    speed_true = np.zeros(n)
    tau = 3.5  # s, drivetrain lag
    for i in range(1, n):
        brake_boost = 2.2 if stop_mask[i] else 1.0
        speed_true[i] = (speed_true[i - 1]
                         + (target[i] - speed_true[i - 1]) * DT / (tau / brake_boost))
    speed_true = np.clip(speed_true, 0.0, 220.0)

    # The model runs on the true speed; the *sensor* sees noise. Deriving
    # acceleration from the noisy value would manufacture spurious regen
    # events, which is exactly the bug this split fixes.
    accel = np.clip(np.gradient(speed_true, DT), -12.0, 12.0)
    speed = np.clip(speed_true + rng.normal(0.0, 0.12, n), 0.0, 220.0)

    # ---- pedals ----------------------------------------------------------
    accel_pedal = np.clip(np.maximum(accel, 0.0) * 9.0, 0.0, 100.0)
    # The bus carries brake *application*, not travel (see signals.yaml).
    brake_active = (accel < -1.0).astype(float)

    # ---- energy: road-load model ----------------------------------------
    speed_mps = speed_true / 3.6
    aero_force_n = AERO_DRAG_COEFF_N_PER_MPS2 * speed_mps**2
    rolling_force_n = ROLLING_RESISTANCE_COEFF * MASS_KG * 9.81
    traction_w = (aero_force_n + rolling_force_n) * speed_mps

    hvac_demand = np.abs(scenario.ambient_c - scenario.hvac_setpoint_c)
    hvac_power = np.clip(180.0 + 210.0 * hvac_demand, 0.0, MAX_HVAC_W)
    hvac_power = np.where(speed_true > 1.0, hvac_power, hvac_power * 0.55)

    aux_w = hvac_power + AUX_BASE_W
    electrical_w = traction_w / DRIVETRAIN_EFFICIENCY + aux_w

    # Extra draw to accelerate; recuperation only under genuine braking.
    transient_w = (MASS_KG * np.maximum(accel, 0.0)
                   * np.maximum(speed_mps, 0.5)) / DRIVETRAIN_EFFICIENCY
    recuperated_w = np.where(
        brake_active > 0,
        np.clip(REGEN_W_PER_MS2 * np.maximum(-accel, 0.0), 0.0, MAX_REGEN_W),
        0.0,
    )

    pack_w = electrical_w + np.clip(transient_w, 0.0, 220_000.0) - recuperated_w
    pack_w = np.where(speed_true < 0.5, aux_w, pack_w)   # parked: only aux
    pack_w = np.clip(pack_w, -MAX_REGEN_W, 400_000.0)

    # Instantaneous consumption, as a VCU would report it. Dividing aux draw by
    # a near-zero speed when parked produces absurd Wh/km, so hold the last
    # moving value across stops instead of inventing a spike.
    energy_per_km = electrical_w / np.maximum(speed_true, 5.0)
    energy_per_km = _hold_last_moving(energy_per_km, speed_true)
    energy_per_km = _trailing_mean(energy_per_km, int(30 * RATE_HZ))
    energy_per_km += rng.normal(0.0, 1.2, n)
    energy_per_km = np.clip(energy_per_km, 0.0, 1999.0)

    regen_w = np.clip(recuperated_w, 0.0, MAX_REGEN_W)

    # ---- pack: voltage, current, SOC (coulomb counting) ------------------
    current = pack_w / PACK_NOMINAL_V
    current = np.clip(current, -500.0, 500.0)
    pack_voltage = PACK_NOMINAL_V - current * PACK_RESISTANCE_OHM

    soc = np.zeros(n)
    soc[0] = scenario.initial_soc
    coulomb_per_tick = current / 3600.0 / RATE_HZ  # Ah per tick
    pack_ah = PACK_CAPACITY_KWH * 1000.0 / PACK_NOMINAL_V
    for i in range(1, n):
        soc[i] = soc[i - 1] - 100.0 * coulomb_per_tick[i - 1] / pack_ah
    soc = np.clip(soc, 0.0, 100.0)

    # ---- pack thermal ----------------------------------------------------
    temp = np.zeros(n)
    temp[0] = max(scenario.ambient_c + 4.0, 18.0)
    coolant_in = np.zeros(n)
    coolant_in[0] = scenario.ambient_c
    pump_pct = np.zeros(n)
    coolant_dt = np.zeros(n)

    for i in range(1, n):
        t_s = t[i]
        pump_ok = _pump_available(scenario.faults, t_s)
        q_joule = (current[i - 1] ** 2) * PACK_RESISTANCE_OHM
        q_extra = _extra_heat(scenario.faults, t_s)
        q_amb = HEAT_TRANSFER_AMBIENT_W_PER_K * (scenario.ambient_c - temp[i - 1])

        # Baseline circulation whenever the vehicle is live, rising with pack
        # temperature. A failed pump drops this to zero, which is a clean,
        # observable predecessor to the thermal excursion - the signature the
        # analytics layer is supposed to find for Q4.
        thermal_demand = float(np.clip((temp[i - 1] - 28.0) / 30.0 * 100.0, 0.0, 100.0))
        live = 1.0 if speed[i - 1] > 1.0 else 0.0
        duty = 0.0 if not pump_ok else max(PUMP_BASELINE_PCT * live, thermal_demand)
        pump_pct[i] = duty

        coolant_in[i] = scenario.ambient_c + 0.55 * max(temp[i - 1] - scenario.ambient_c, 0.0)
        coolant_dt[i] = float(np.clip(
            (temp[i - 1] - coolant_in[i]) * 0.4, 0.0, COOLANT_MAX_DT_C))
        q_cool = COOLANT_GAIN_W_PER_K * (duty / 100.0) * coolant_dt[i]

        dT = (q_joule + q_extra + q_amb + 0.0 - q_cool) * DT / PACK_THERMAL_CAPACITY_J_PER_K
        temp[i] = temp[i - 1] + dT

    temp += rng.normal(0.0, 0.02, n)
    cell_temp_max = np.clip(temp, -40.0, 124.0)
    cell_temp_min = np.clip(temp - (1.2 + 0.8 * np.abs(current) / 200.0), -40.0, 124.0)
    overheat = (cell_temp_max >= OVERHEAT_WARN_C).astype(float)

    # ---- derivative signals ---------------------------------------------
    range_km = np.clip(soc / 100.0 * PACK_CAPACITY_KWH * 1000.0
                       / np.maximum(energy_per_km, 60.0), 0.0, 1000.0)

    adas = np.array([_adas_state(scenario.faults, x) for x in t], dtype=float)

    # sensor drift: a plausible slow bias, the classic "why is my range wrong"
    for f in scenario.faults_of(FaultKind.SENSOR_DRIFT):
        lo, hi = int(f.start_s * RATE_HZ), min(int((f.start_s + f.duration_s) * RATE_HZ), n)
        drift = f.params.get("per_second", 0.02) * (t[lo:hi] - f.start_s)
        if f.params.get("target", 1.0) > 0:
            energy_per_km[lo:hi] = energy_per_km[lo:hi] + drift

    truth = {
        "vehicle.speed": speed,
        "vehicle.accel_long": accel,
        "chassis.steer_angle": np.clip(rng.normal(0.0, 3.0, n), -540, 540),
        "chassis.brake_active": brake_active,
        "powertrain.accel_pedal": accel_pedal,
        "powertrain.motor_torque": np.clip(MASS_KG * accel * 0.12, -500, 500),
        "powertrain.motor_rpm": np.clip(speed * 88.0, 0, 20000),
        "powertrain.gear": np.full(n, 4.0),
        "adas.state": adas,
        "adas.ldw_active": np.zeros(n),
        "adas.aeb_active": np.zeros(n),
        "thermal.ambient": np.full(n, scenario.ambient_c),
        "battery.soc": soc,
        "battery.pack_voltage": np.clip(pack_voltage, 0, 800),
        "battery.pack_current": current,
        "battery.cell_temp_max": cell_temp_max,
        "battery.cell_temp_min": cell_temp_min,
        "battery.cell_imbalance": np.clip(8.0 + rng.normal(0, 1.5, n), 0, 16383),
        "battery.overheat_warning": overheat,
        "thermal.coolant_inlet": np.clip(coolant_in, -40, 125),
        "thermal.coolant_pump": pump_pct,
        "energy.per_km": energy_per_km,
        "energy.range_estimate": range_km,
        "energy.regen_power": regen_w,
        "hvac.power": hvac_power,
        "hvac.setpoint": np.full(n, scenario.hvac_setpoint_c),
    }

    return Simulation(
        scenario=scenario, t_s=t, truth=truth,
        events=_ground_truth_events(scenario, cell_temp_max, adas, t),
        modes=mode_series,
    )


def _ground_truth_events(scenario, cell_temp_max, adas, t) -> list[Event]:
    """Events the analytics layer is expected to rediscover."""
    events: list[Event] = []

    warn_idx = np.flatnonzero(cell_temp_max >= OVERHEAT_WARN_C)
    if warn_idx.size:
        events.append(Event(
            ts_s=float(t[warn_idx[0]]), kind="battery.overheat_warning",
            description=f"pack reached the {OVERHEAT_WARN_C:.0f} C warning threshold",
            severity="warning"))

    crit_idx = np.flatnonzero(cell_temp_max >= OVERHEAT_CRITICAL_C)
    if crit_idx.size:
        events.append(Event(
            ts_s=float(t[crit_idx[0]]), kind="battery.overheat_critical",
            description=f"pack exceeded {OVERHEAT_CRITICAL_C:.0f} C", severity="critical"))

    # ADAS engagement falling edge == disengagement
    drops = np.flatnonzero((adas[:-1] > 0) & (adas[1:] == 0))
    for i in drops:
        events.append(Event(
            ts_s=float(t[i + 1]), kind="adas.disengagement",
            description="ADAS disengaged", severity="info"))

    for f in scenario.faults:
        events.append(Event(
            ts_s=f.start_s, kind=f"fault.{f.kind.value}",
            description=f"{f.kind.value} injected", severity="info"))

    return sorted(events, key=lambda e: e.ts_s)
