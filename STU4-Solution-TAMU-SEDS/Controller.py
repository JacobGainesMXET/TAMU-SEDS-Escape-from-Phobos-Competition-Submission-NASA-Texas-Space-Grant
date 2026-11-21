#############################################################################################
# DON'T CHANGE ANY OF THE BELOW. NECESSARY FOR CORRECTLY JOINING THE COMPETITION SIMULATION #
#############################################################################################

import os, sys, time, datetime, traceback
import spaceteams as st
def custom_exception_handler(exctype, value, tb):
    error_message = "".join(traceback.format_exception(exctype, value, tb))
    st.logger_fatal(error_message)
sys.excepthook = custom_exception_handler
st.connect_to_sim(sys.argv)

import numpy as np
import numpy.typing as npt


this = st.GetThisSystem()

doSpawning = st.SimGlobals.GetSimEntity().GetParam(st.VarType.bool, "DoSpawning")
if doSpawning:
    while True:
        time.sleep(1.0 / this.GetParam(st.VarType.double, "LoopFreqHz"))

phobos: st.Entity = this.GetParam(st.VarType.entityRef, "Phobos")
phobosFrame = phobos.GetBodyFixedFrame()
shuttle: st.Entity = this.GetParam(st.VarType.entityRef, "Shuttle")
shuttleFrame = shuttle.GetBodyFixedFrame()
target: st.Entity = this.GetParam(st.VarType.entityRef, "Target")

topCamera: st.Entity = this.GetParam(st.VarType.entityRef, "TopCamera")
topCameraFrame = topCamera.GetBodyFixedFrame()
bottomCamera: st.Entity = this.GetParam(st.VarType.entityRef, "BottomCamera")
bottomCameraFrame = bottomCamera.GetBodyFixedFrame()


def GetNearestDebrisLocations(N: int) -> list[npt.NDArray]:
    allDebrisInfo: list[npt.NDArray] = shuttle.GetParamArray(st.VarType.doubleM3x3, 
                                                            ["Radar", "SortedDebrisData"])
    nearest = allDebrisInfo[:min(N, len(allDebrisInfo))]
    return nearest


def PanTiltCamera(camera: st.Entity, yaw_deg: float, pitch_deg: float) -> None:
    if camera == topCamera:
        yaw_deg = np.clip(yaw_deg, -89.0, 89.0)
        pitch_deg = np.clip(pitch_deg, -50.0, 0.0)
    elif camera == bottomCamera:
        yaw_deg = np.clip(yaw_deg, -89.0, 89.0)
        pitch_deg = np.clip(pitch_deg, 0.0, 50.0)
    camera.setRotation_DCM(st.math.Euler321_to_DCM([np.deg2rad(yaw_deg), np.deg2rad(pitch_deg), 0.0]), shuttleFrame)


def SetCommand(accelCommand: npt.NDArray, angAccelCommand: npt.NDArray) -> None:
    shuttle.SetParam(st.VarType.doubleV3, ["JetSelection", "CommandedAccel"], accelCommand)
    shuttle.SetParam(st.VarType.doubleV3, ["JetSelection", "CommandedAngAccel"], angAccelCommand)


#############################################################################################
# DON'T CHANGE ANY OF THE ABOVE. NECESSARY FOR CORRECTLY JOINING THE COMPETITION SIMULATION #
#############################################################################################

#############################################################################################
# RADAR TOOLS                                                                               #
# Only needs numpy (np / npt are already imported by the boilerplate).                      #
#############################################################################################

from dataclasses import dataclass

# From the wiki: radar 1-sigma noise is 2 m on position and closest-range, 0.02 m/s on range
# rate, and 5 % on diameter. The cone uses the position noise as part of its safety margin.
RADAR_POS_SIGMA_M = 2.0

# ROUGH guess at the shuttle's bounding radius (camera mounts sit ~1-2 m from the CoM).
# Check it against the wireframe images on the wiki and tune it.
SHUTTLE_RADIUS_M = 2.5


@dataclass
class RadarContact:
    """One radar observation of one piece of debris, with the 3x3 matrix unpacked."""
    rel_pos: npt.NDArray     # m    debris center relative to the shuttle, in Phobos-frame axes
    surface_range: float     # m    shuttle -> closest point on the debris (radar r_c)
    range_rate: float        # m/s  positive = moving apart, negative = closing
    diameter: float          # m    average diameter (5 % noise)
    t_meas: float            # s    sim time of the measurement

    @property
    def center_range(self) -> float:
        return float(np.linalg.norm(self.rel_pos))

    @property
    def radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def closing(self) -> bool:
        return self.range_rate < 0.0

    def time_to_contact(self, clearance_m: float = 0.0) -> float:
        """
        Rough time until the gap to the debris surface shrinks to `clearance_m`, assuming
        the current range rate holds. Radar only gives the radial part of the relative
        velocity, so treat this as a priority hint, not a prediction. inf if not closing.
        """
        if self.range_rate >= 0.0:
            return float("inf")
        return max(self.surface_range - clearance_m, 0.0) / -self.range_rate


@dataclass
class TravelCone:
    """The cone that was used, plus the debris that intersects it (nearest first)."""
    axis: npt.NDArray        # unit vector, Phobos-frame axes, from the shuttle outward
    length_m: float
    half_angle_deg: float
    contacts: list[RadarContact]


def GetRadarContacts(max_count: int = 500) -> list[RadarContact]:
    """Read the radar once and unpack it. Call this once per loop and reuse the result."""
    contacts: list[RadarContact] = []
    for item in GetNearestDebrisLocations(max_count):   # boilerplate helper, max 500
        contacts.append(RadarContact(
            rel_pos=np.array(item[0, :], dtype=float),
            surface_range=float(item[1, 0]),
            range_rate=float(item[1, 1]),
            diameter=float(item[1, 2]),
            t_meas=float(item[2, 0]),
        ))
    return contacts


def GetDebrisWithinRange(contacts: list[RadarContact],
                         max_range_m: float,
                         reference=None,
                         measure: str = "surface") -> list[RadarContact]:
    """
    All debris within `max_range_m` of a reference, nearest first.

    reference:
        None             -> measure from the shuttle (default)
        a RadarContact   -> measure from ANOTHER piece of debris (that piece is excluded)
        a 3-vector       -> measure from any point, given relative to the shuttle in
                            Phobos-frame axes
    measure:
        "surface" -> gap to the debris surface (uses the radar's own r_c for the shuttle case)
        "center"  -> distance to the debris center
    """
    if not contacts:
        return []

    if reference is None:
        if measure == "surface":
            dist = np.array([c.surface_range for c in contacts])
        else:
            dist = np.array([c.center_range for c in contacts])
        mask = dist <= max_range_m
    else:
        if isinstance(reference, RadarContact):
            ref_pos, ref_radius = reference.rel_pos, reference.radius
        else:
            ref_pos, ref_radius = np.asarray(reference, dtype=float), 0.0

        positions = np.array([c.rel_pos for c in contacts])
        dist = np.linalg.norm(positions - ref_pos, axis=1)
        if measure == "surface":
            radii = np.array([c.radius for c in contacts])
            dist = np.maximum(dist - radii - ref_radius, 0.0)
        mask = (dist <= max_range_m) & np.array([c is not reference for c in contacts])

    idx = np.flatnonzero(mask)
    idx = idx[np.argsort(dist[idx])]
    return [contacts[i] for i in idx]


def GetDebrisInTravelCone(contacts: list[RadarContact],
                          ship_vel_phobos: npt.NDArray,
                          half_angle_deg: float = 20.0,
                          lookahead_s: float = 10.0,
                          min_length_m: float = 30.0,
                          fallback_axis: npt.NDArray | None = None,
                          min_speed_m_s: float = 0.5,
                          ship_radius_m: float = SHUTTLE_RADIUS_M,
                          safety_margin_m: float = 2.0 * RADAR_POS_SIGMA_M,
                          only_closing: bool = False) -> TravelCone:
    """
    Debris that could matter as the shuttle travels: everything that touches a cone whose
    apex is the shuttle and whose axis points along the direction of travel.

    ship_vel_phobos   shuttle velocity in Phobos-frame axes (the same `vel` the starter code
                      reads). Radar positions are already in these axes, so no conversion.
    half_angle_deg    cone half-angle. Widen it if you expect to steer / swerve a lot.
    lookahead_s       cone length = max(min_length_m, speed * lookahead_s)
    fallback_axis     direction to use when the shuttle is nearly stationary (e.g. the
                      direction to the target). With no usable direction at all, the cone
                      becomes a full sphere of radius `length`, which is the safe choice.
    ship_radius_m,    each piece of debris is inflated by its own radius + the shuttle's
    safety_margin_m   radius + this margin (default = 2 sigma of radar position noise), so
                      big debris just off-axis is still caught and noise doesn't hide a miss.
    only_closing      if True, drop debris whose range rate says it is moving away.
    """
    v = np.asarray(ship_vel_phobos, dtype=float)
    speed = float(np.linalg.norm(v))

    if speed >= min_speed_m_s:
        axis = v / speed
        half_deg = half_angle_deg
    elif fallback_axis is not None and np.linalg.norm(fallback_axis) > 1e-9:
        fa = np.asarray(fallback_axis, dtype=float)
        axis = fa / np.linalg.norm(fa)
        half_deg = half_angle_deg
    else:
        axis = np.array([1.0, 0.0, 0.0])   # arbitrary; irrelevant for a full sphere
        half_deg = 180.0

    length = max(min_length_m, speed * lookahead_s)
    cone = TravelCone(axis=axis, length_m=length, half_angle_deg=half_deg, contacts=[])
    if not contacts:
        return cone

    positions = np.array([c.rel_pos for c in contacts])               # (N, 3)
    dist = np.linalg.norm(positions, axis=1)                          # center distances
    r_eff = np.array([c.radius for c in contacts]) + ship_radius_m + safety_margin_m

    safe_dist = np.maximum(dist, 1e-9)
    along = positions @ axis                                          # distance along the axis
    theta = np.arccos(np.clip(along / safe_dist, -1.0, 1.0))          # angle off the axis
    alpha = np.arcsin(np.clip(r_eff / safe_dist, 0.0, 1.0))           # angular size of inflated debris

    inside = (theta <= np.deg2rad(half_deg) + alpha) & (along <= length + r_eff)
    inside |= dist <= r_eff                                           # already touching: always keep

    if only_closing:
        inside &= np.array([c.closing for c in contacts])

    idx = np.flatnonzero(inside)
    idx = idx[np.argsort(dist[idx])]
    cone.contacts = [contacts[i] for i in idx]
    return cone

#############################################################################################
# END RADAR TOOLS                                                                           #
#############################################################################################
# ---------------------------------------------------------------------------------------
# Example use inside the main loop (uses `vel` and `goalDirection` from the starter code):
#
#   contacts = GetRadarContacts(500)                       # read the radar ONCE per loop
#
#   nearby = GetDebrisWithinRange(contacts, 50.0)          # gap to surface <= 50 m
#   nearby_center = GetDebrisWithinRange(contacts, 50.0, measure="center")
#   neighbors = GetDebrisWithinRange(contacts, 15.0, reference=nearby[0])  # near ANOTHER debris
#
#   cone = GetDebrisInTravelCone(contacts, vel, half_angle_deg=20.0,
#                                lookahead_s=10.0, fallback_axis=goalDirection)
#   for c in cone.contacts[:5]:
#       print(c.surface_range, c.range_rate, c.time_to_contact(SHUTTLE_RADIUS_M))
#
#   st.OnScreenLogMessage(f"{len(nearby)} within 50 m, {len(cone.contacts)} in travel cone",
#                         "RadarTools", st.Severity.Info)
# ---------------------------------------------------------------------------------------


#############################################################################################
# QUALIFICATION CONTROLLER: fly toward the target, steer around radar-detected debris       #
#############################################################################################

import math
import copy
import csv
import hashlib
import json
from collections import deque
from dataclasses import asdict
from pathlib import Path

# Standalone flight controller. NumPy and spaceteams are supplied by Space Teams.
controller_directory = os.path.dirname(os.path.abspath(__file__))


# Guidance, radar tracking, and tactical avoidance

def limited(vector, maximum):
    vector = np.asarray(vector, dtype=float)
    return vector * min(1., maximum / max(float(np.linalg.norm(vector)), 1e-12))


def unit(vector, fallback=(1., 0., 0.)):
    length = float(np.linalg.norm(vector))
    return np.asarray(vector, dtype=float) / length if length > 1e-9 else np.array(fallback, dtype=float)


LAUNCH_PLAN_TIME = 20.
COAST_PROGRESS = .70
FORWARD_LOOKAHEAD = 16.
LAUNCH_LIFT_MIN = .08  # m/s², bracketed from broadside jet-allocation checks
LAUNCH_LIFT_MAX = .16


def rear_thrust(request, nose, goal, velocity):
    """Translate along the current nose only when it adds useful forward impulse."""
    magnitude = np.linalg.norm(request)
    if (magnitude <= 1e-9 or nose @ goal <= 0. or nose @ velocity < 0.
            or nose @ request < magnitude * math.cos(math.radians(35.))):
        return np.zeros(3)
    return magnitude * nose


# A small forward component lets the installed allocator sustain stronger +Z
# thrust with almost all propellant going through the bottom jets.
BOTTOM_LIFT_BIAS_DEG = .05


def low_tilt_target(launch_axes, goal, nose_bias_deg=1.):
    """Bring the shuttle's +Z axis toward the departure corridor with a small tilt."""
    local = launch_axes @ goal
    # Slight nose-up bias leaves a small positive-X translation component.
    # On the installed asymmetric jet layout that improves vertical authority.
    horizontal = local[:2] - np.array([math.sin(math.radians(nose_bias_deg)), 0.])
    length = np.linalg.norm(horizontal)
    if length < 1e-10:
        return launch_axes.copy()
    angle = min(math.atan2(length, local[2]), math.radians(7.5))
    # Rows of the rotation give the desired body axes in launch coordinates.
    x, y = horizontal / length
    skew = np.array([[0., 0., -x], [0., 0., -y], [x, y, 0.]])
    return (np.eye(3) + math.sin(angle)*skew + (1. - math.cos(angle))*(skew @ skew)) @ launch_axes


def departure_burn_direction(r, v, gravity, now, model):
    """Direction of the best priced straight burn followed by ballistic entry."""
    # For arrival after T, a burn lasting b adds a*b*(T-b/2) metres.
    # Aim at the near edge of the capture sphere along the required impulse.
    times = np.arange(8., 601., 4.)
    offset = r - times[:, None] * v - .5 * times[:, None]**2 * gravity
    travel = np.maximum(0., np.linalg.norm(offset, axis=1) - (model.success_radius - 8.))
    authority = model.planning_accel
    discriminant = times**2 - 2. * travel / authority
    feasible = discriminant >= 0.
    if not np.any(feasible):
        return unit(r)
    times, travel = times[feasible], travel[feasible]
    impulse = 2. * travel / (times + np.sqrt(discriminant[feasible]))
    value = model.nominal_time / (now + times) - model.fuel_per_dv * impulse / model.nominal_fuel
    return unit(offset[feasible][np.argmax(value)], unit(r))


def nonbraking(acceleration, velocity, goal):
    """Preserve forward progress and allow corrections perpendicular to the flight path."""
    axis = unit(goal)
    command = acceleration - min(0., float(acceleration @ axis)) * axis
    if velocity @ axis > 0. and velocity @ velocity > 1e-8:
        command = command - min(0., float(command @ velocity)) / float(velocity @ velocity) * velocity
    return command


@dataclass
class MissionModel:
    # Verified against the local scoring system; not written to the simulator.
    nominal_time: float = 120.
    nominal_fuel: float = 75.
    max_damage: float = 209200.
    success_radius: float = 50.
    flyby_offset: float = 30.           # aim off the ship centre, still inside its success sphere
    planning_accel: float = .09        # below the measured .096 minimum pure-axis authority
    avoidance_accel: float = .09
    fuel_per_dv: float = 5.6           # kg/(m/s); ideal is mass/(80*9.8), updated from flight
    coast_speed: float | None = None  # initial burn cutoff, never an exact speed-holding target
    cruise_burns: bool = False        # timed course trims and score-qualified speed restoration
    cruise_speed_band: float = .3     # review a finite burn at this closing-speed deficit
    cruise_coast_seconds: float = 8.  # minimum coast between optional speed-restoration burns
    reburn_score_margin: float = .25
    position_sigma: float = 2.
    process_accel_sigma: float = .015   # tuning allowance for unmodelled relative acceleration
    uncertainty_sigma: float = 2.5
    ship_radius: float = 2.5           # existing geometry estimate; still needs hull validation
    clearance: float = 3.
    horizon: float = 30.
    track_timeout: float = 12.
    radar_timeout: float = 3.
    avoidance_turn_rate: float = math.radians(.25)  # conservative allocated-attitude forecast
    avoidance_turn_delay: float = 3.              # allow time for damped attitude response
    arrival_target: float | None = None  # absolute mission seconds, optional time-directed cruise
    score_cruise: bool = False
    low_tilt: bool = False
    bottom_lift: bool = False

    def intervention_padding(self, sigma):
        """Extra nominal-path padding beyond both hulls and the fixed clearance."""
        # A very uncertain velocity can spread one return over tens of metres.
        # Continue tracking that envelope. Spend impulse when its estimated
        # path also comes within a radar position-error band of the buffer.
        return np.minimum(self.uncertainty_sigma * sigma, self.uncertainty_sigma * self.position_sigma)

    def score(self, elapsed, fuel, damage=0.):
        time_score = self.nominal_time / elapsed if elapsed > 0. else 0.
        return 50. + 100. * (time_score - max(fuel / self.nominal_fuel, 0.) - 2. * damage / self.max_damage)


def learn_burn_model(model, fuel_delta, delivered_dv, angular_delta, ideal_cost, on_seconds=0.):
    """Calibrate the score estimate on low-rotation burn windows, never on coasting.

    Include actual allocated propellant, not the commanded acceleration magnitude.
    """
    if delivered_dv > .03 and angular_delta < .01 and fuel_delta > 0.:
        if on_seconds > 0. and not .03 < delivered_dv / on_seconds < .25:
            return
        measured = float(np.clip(fuel_delta / delivered_dv, ideal_cost, 3. * ideal_cost))
        model.fuel_per_dv = .9 * model.fuel_per_dv + .1 * measured
        # Use most of observed burn authority instead of unnecessarily throttling a
        # good burn. Keep 5% reserve and reject impact-sized acceleration estimates.
        if on_seconds > .2 and .03 < delivered_dv / on_seconds < .25:
            available = min(.16, .95 * delivered_dv / on_seconds)
            model.planning_accel = .8 * model.planning_accel + .2 * available


class StateHistory:
    """Own-ship truth history, used to put each OLD radar sample into world coordinates."""
    def __init__(self):
        self.samples = deque()

    def add(self, time, position, velocity):
        if self.samples and time <= self.samples[-1][0]:
            return
        self.samples.append((time, np.array(position), np.array(velocity)))
        while len(self.samples) > 2 and self.samples[1][0] < time - 20.:
            self.samples.popleft()

    def at(self, time):
        if not self.samples or time > self.samples[-1][0] + .2:
            return None
        first_time, _, first_velocity = self.samples[0]
        if time < first_time - .5:
            # The initial radar sweep is stamped zero. The control loop can
            # start a fraction of a second later while the shuttle is on its pad.
            if not (0. <= time < first_time <= 1. and np.linalg.norm(first_velocity) <= .1):
                return None
        times = [sample[0] for sample in self.samples]
        index = int(np.searchsorted(times, time))
        if index == 0 or index == len(times):
            stamp, position, velocity = self.samples[min(index, len(times) - 1)]
            return position + velocity * (time - stamp), velocity.copy()
        t0, p0, v0 = self.samples[index - 1]
        t1, p1, v1 = self.samples[index]
        weight = (time - t0) / (t1 - t0)
        return p0 + weight * (p1 - p0), v0 + weight * (v1 - v0)


def predict_state(state, covariance, dt, gravity, process_sigma):
    transition = np.eye(6)
    transition[:3, 3:] = np.eye(3) * dt
    predicted = transition @ state
    predicted[:3] += .5 * gravity * dt * dt
    predicted[3:] += gravity * dt
    noise_map = np.vstack((np.eye(3) * .5 * dt * dt, np.eye(3) * dt))
    uncertainty = transition @ covariance @ transition.T + process_sigma**2 * noise_map @ noise_map.T
    return predicted, uncertainty


def measurement_update(state, covariance, matrix, value, noise):
    innovation = matrix @ covariance @ matrix.T + noise
    gain = np.linalg.solve(innovation, matrix @ covariance).T
    updated = state + gain @ (value - matrix @ state)
    identity = np.eye(6) - gain @ matrix
    # Joseph form preserves positive covariance despite very accurate radial velocities.
    uncertainty = identity @ covariance @ identity.T + gain @ noise @ gain.T
    return updated, .5 * (uncertainty + uncertainty.T)


@dataclass
class DebrisTrack:
    number: int
    time: float
    state: np.ndarray
    covariance: np.ndarray
    radius: float
    hits: int = 1
    reliable_hits: int | None = None
    last_ray: np.ndarray | None = None

    def __post_init__(self):
        if self.reliable_hits is None:
            self.reliable_hits = self.hits

    def predict(self, time, gravity, model):
        return predict_state(self.state, self.covariance, max(0., time - self.time),
                             gravity, model.process_accel_sigma)


class RadarTracker:
    """Gated one-to-one association: radar array indices are never object identities.

    A position/velocity Kalman filter uses 3-D position history plus the independent
    radial-velocity measurement. A single range rate is NOT treated as a 3-D velocity.
    """
    def __init__(self, model):
        self.model = model
        self.tracks = {}
        self.seen = {}
        self.next_number = 1
        self.last_fresh = -math.inf
        self.stats = {}

    def update(self, contacts, now, history, gravity):
        self.tracks = {key: track for key, track in self.tracks.items()
                       if now - track.time <= self.model.track_timeout}
        self.seen = {key: stamp for key, stamp in self.seen.items() if now - stamp <= 16.}
        self.stats = dict(new=0, matched=0, created=0, rejected=0, ambiguous=0)
        observations = []
        for contact in contacts:
            values = np.r_[contact.rel_pos, contact.surface_range, contact.range_rate,
                           contact.diameter, contact.t_meas]
            if (not np.all(np.isfinite(values)) or contact.diameter <= 0.
                    or contact.t_meas < 0. or contact.t_meas > now + .2
                    or now - contact.t_meas > self.model.track_timeout):
                self.stats['rejected'] += 1
                continue
            key = tuple(values)
            if key in self.seen:
                continue
            self.seen[key] = contact.t_meas
            ship_state = history.at(contact.t_meas)
            distance = float(np.linalg.norm(contact.rel_pos))
            if ship_state is None or distance < 1e-3:
                self.stats['rejected'] += 1
                continue
            # Independent centre/surface observations should agree within their noise.
            tolerance = 10. + .15 * contact.diameter
            if abs(distance - .5 * contact.diameter - contact.surface_range) > tolerance:
                self.stats['rejected'] += 1
                continue
            ship_position, ship_velocity = ship_state
            observations.append((contact, ship_position + contact.rel_pos, ship_velocity,
                                 contact.rel_pos / distance))
            self.last_fresh = max(self.last_fresh, contact.t_meas)
        self.stats['new'] = len(observations)
        # Associate against the tracks that existed BEFORE this batch. Otherwise the
        # 500 simultaneous initial measurements could merge unrelated nearby rocks.
        tracks = list(self.tracks.values())
        pairs = []
        ambiguous = set()
        if tracks:
            states = np.array([track.state for track in tracks])
            covariances = np.array([track.covariance for track in tracks])
            timestamps = np.array([track.time for track in tracks])
            diameters = np.array([2. * track.radius for track in tracks])
        for index, (contact, position, ship_velocity, ray) in enumerate(observations):
            candidates = []
            if tracks:
                dt = np.maximum(0., contact.t_meas - timestamps)
                predicted_position = states[:, :3] + states[:, 3:] * dt[:, None] + .5 * gravity * dt[:, None]**2
                position_variance = (np.diagonal(covariances[:, :3, :3], axis1=1, axis2=2)
                                     + 2. * dt[:, None] * np.diagonal(covariances[:, :3, 3:], axis1=1, axis2=2)
                                     + dt[:, None]**2 * np.diagonal(covariances[:, 3:, 3:], axis1=1, axis2=2)
                                     + .25 * self.model.process_accel_sigma**2 * dt[:, None]**4)
                scores = np.sum((position - predicted_position)**2 /
                                np.maximum(position_variance + self.model.position_sigma**2, 1e-6), axis=1)
                radial_errors = contact.range_rate - (states[:, 3:] + gravity * dt[:, None] - ship_velocity) @ ray
                radial_variances = (np.einsum('i,nij,j->n', ray, covariances[:, 3:, 3:], ray)
                                    + self.model.process_accel_sigma**2 * dt**2 + .25)
                valid = ((contact.t_meas > timestamps + 1e-6) & (scores <= 16.)
                         & (np.abs(contact.diameter - diameters) <= np.maximum(1., .25 * np.maximum(diameters, contact.diameter)))
                         & (radial_errors**2 <= 16. * radial_variances))
                for track_index in np.flatnonzero(valid):
                    score = scores[track_index] + radial_errors[track_index]**2 / radial_variances[track_index]
                    candidates.append((float(score), index, tracks[track_index].number))
            candidates.sort()
            if len(candidates) > 1 and candidates[1][0] < candidates[0][0] + 2.:
                ambiguous.add(index)
            pairs.extend(candidates)
        assigned = {}
        used_tracks = set()
        for score, index, number in sorted(pairs):
            if index not in assigned and number not in used_tracks:
                assigned[index] = number
                used_tracks.add(number)
        for index, (contact, position, ship_velocity, ray) in enumerate(observations):
            if index in assigned:
                track = self.tracks[assigned[index]]
                state, covariance = track.predict(contact.t_meas, gravity, self.model)
                track.hits += 1
                self.stats['matched'] += 1
                if index in ambiguous:
                    covariance[3:, 3:] += np.eye(3) * .5**2
                    self.stats['ambiguous'] += 1
                    track.reliable_hits = 0
                else:
                    track.reliable_hits += 1
            else:
                # Unknown transverse velocity gets a broad prior, not a fictitious
                # radar-derived vector. Subsequent position observations resolve it.
                # Undo only own radial motion. A Doppler return provides no
                # evidence that the asteroid shares our transverse velocity.
                state = np.r_[position, (ship_velocity @ ray + contact.range_rate) * ray]
                covariance = np.diag([1e6] * 3 + [16.] * 3)
                track = DebrisTrack(self.next_number, contact.t_meas, state, covariance, .5 * contact.diameter)
                self.tracks[track.number] = track
                self.next_number += 1
                self.stats['created'] += 1
            position_matrix = np.column_stack((np.eye(3), np.zeros((3, 3))))
            state, covariance = measurement_update(state, covariance, position_matrix, position,
                                                   np.eye(3) * self.model.position_sigma**2)
            velocity_matrix = np.r_[np.zeros(3), ray][None, :]
            # Bearing noise couples transverse motion into the radial observation.
            bearing_noise = 2. / max(contact.center_range, 2.) * np.linalg.norm(state[3:] - ship_velocity)
            radial_noise = .08**2 + bearing_noise**2
            if index not in ambiguous:
                # The simulator independently perturbs surface range and 3D position.
                # Linearize the range at the measured ray. Diameter uncertainty is
                # part of this measurement's noise, especially for the larger rocks.
                range_matrix = np.r_[ray, np.zeros(3)][None, :]
                centre_range = contact.surface_range + .5 * contact.diameter
                range_var = self.model.position_sigma**2 + (.025 * contact.diameter)**2
                ship_origin = position - contact.rel_pos
                state, covariance = measurement_update(
                    state, covariance, range_matrix, np.array([ray @ ship_origin + centre_range]),
                    np.array([[range_var]]))
                state, covariance = measurement_update(
                    state, covariance, velocity_matrix,
                    np.array([contact.range_rate + ship_velocity @ ray]), np.array([[radial_noise]]))
            track.state, track.covariance, track.time = state, covariance, contact.t_meas
            track.last_ray = ray
            track.radius = .8 * track.radius + .2 * (.5 * contact.diameter)
        return [item[0] for item in observations]

    def ready(self, now):
        confirmed = sum(track.hits >= 2 for track in self.tracks.values())
        return bool(self.tracks) and confirmed >= .6 * len(self.tracks) and now - self.last_fresh < self.model.radar_timeout

    def fresh(self, now):
        """Initial observations can protect launch while transverse tracks mature."""
        return bool(self.tracks) and now - self.last_fresh < self.model.radar_timeout


def travel_time(distance, speed, acceleration):
    if distance <= 0.:
        return 0.
    if abs(acceleration) < 1e-9:
        return distance / speed if speed > 1e-6 else math.inf
    discriminant = speed * speed + 2. * acceleration * distance
    if discriminant < 0.:
        return math.inf
    roots = [(-speed + sign * math.sqrt(discriminant)) / acceleration for sign in (-1., 1.)]
    return min((root for root in roots if root >= 0.), default=math.inf)


@dataclass
class GuidanceDecision:
    acceleration: np.ndarray
    mode: str
    eta: float
    advantage: float = 0.
    burn_seconds: float = 0.
    coast_entry: float = math.inf


def ballistic_entry(relative_position, relative_velocity, gravity, radius, horizon):
    """First predicted entry into a target sphere under constant external acceleration.

    Uses trajectory direction, not nose direction. Positive relative_velocity is
    ship minus target velocity; target acceleration is assumed negligible.
    """
    r, v, g = map(lambda value: np.asarray(value, dtype=float),
                  (relative_position, relative_velocity, gravity))
    if not np.all(np.isfinite(np.r_[r, v, g])) or radius <= 0. or horizon <= 0.:
        return math.inf
    if r @ r <= radius * radius:
        return 0.
    coefficients = np.array([.25 * (g @ g), v @ g, v @ v - r @ g,
                             -2. * (r @ v), r @ r - radius * radius])
    # Drop only exact leading zeros; a small gravity term may matter over a long coast.
    coefficients = np.trim_zeros(coefficients, 'f')
    if len(coefficients) < 2:
        return math.inf
    roots = np.roots(coefficients)
    candidates = [float(root.real) for root in roots
                  if abs(root.imag) < 1e-6 and 0. <= root.real <= horizon]
    return min(candidates, default=math.inf)


def finite_burn_entry(relative_position, relative_velocity, gravity, acceleration, duration,
                      radius, horizon=600.):
    """Arrival during a finite burn or its subsequent gravity-affected coast."""
    r, v, g, a = map(lambda value: np.asarray(value, dtype=float),
                     (relative_position, relative_velocity, gravity, acceleration))
    if not math.isfinite(duration) or not math.isfinite(horizon) or horizon <= 0.:
        return math.inf
    if np.linalg.norm(r) <= radius:
        return 0.
    duration = min(max(float(duration), 0.), horizon)
    during = ballistic_entry(r, v, g + a, radius, duration)
    if math.isfinite(during):
        return during
    after_r = r - v * duration - .5 * (g + a) * duration**2
    after_v = v + (g + a) * duration
    return duration + ballistic_entry(after_r, after_v, g, radius, horizon - duration)


@dataclass
class ScheduledBurn:
    acceleration: np.ndarray
    start: float
    initial_velocity: np.ndarray
    gravity: np.ndarray
    duration: float
    mode: str
    advantage: float = 0.

    def remaining(self, now, velocity, gravity):
        elapsed = max(0., now - self.start)
        magnitude = float(np.linalg.norm(self.acceleration))
        delivered = np.asarray(velocity) - self.initial_velocity - .5 * (self.gravity + gravity) * elapsed
        delivered = float(delivered @ unit(self.acceleration))
        impulse_left = max(0., magnitude * self.duration - delivered)
        return max(0., min(self.duration - elapsed, impulse_left / max(magnitude, 1e-9)))


class AccelerationGuidance:
    """Launch to cruise speed, coast, and schedule finite corrections when needed.

    Acceleration here predicts delivered motion. Controller.py sends a full linear
    command when this nominal path is accepted. This is not a speed-holding servo.
    """
    def __init__(self, model):
        self.model = model
        self.offset = None
        self.coast_latched = False
        self.ballistic_coasting = False
        self.burn = None
        self.last_burn_end = -math.inf
        self.coast_started = None
        self.burn_count = 0
        self.reburn_advantage = 0.
        self.launch_distance = None
        self.distance_coast = False
        self.transit_reason = 'speed cutoff'
        self.coast_arrival = math.inf

    def observe_distance(self, distance):
        if self.launch_distance is None:
            self.launch_distance = distance
        if not self.model.score_cruise:
            self.distance_coast |= distance <= (1. - COAST_PROGRESS) * self.launch_distance

    def finish_burn(self, now):
        if self.burn is not None:
            self.last_burn_end = now
            self.coast_started = now
        self.burn = None

    def departure_horizon(self, distance, closing_speed):
        # Carry the extended forecast only as far as the remaining launch area.
        remaining = distance - .90 * self.launch_distance
        return float(np.clip(remaining / max(.1, closing_speed), self.model.horizon, 1.5 * self.model.horizon))

    def remaining_burn(self, now, velocity, gravity):
        return self.burn.remaining(now, velocity, gravity) if self.burn is not None else 0.

    def _start_burn(self, acceleration, duration, mode, now, velocity, gravity, advantage=0.):
        self.burn = ScheduledBurn(np.array(acceleration), now, np.array(velocity), np.array(gravity),
                                  duration, mode, advantage)
        self.burn_count += 1
        self.coast_started = None

    def _burn_decision(self, r, v, g, now, eta):
        remaining = self.remaining_burn(now, v, g)
        if remaining <= 1e-6:
            self.finish_burn(now)
            return None
        burn = self.burn
        return GuidanceDecision(burn.acceleration.copy(), burn.mode, eta,
                                advantage=burn.advantage, burn_seconds=remaining)

    def _speed_burn(self, r, v, g, now, fuel, coast_entry, radius, nose):
        model = self.model
        if nose is not None and unit(nose) @ unit(v) < math.cos(math.radians(8.)):
            # An optional top-up must not incur a large turn or become a sidestep.
            self.finish_burn(now)
            return None
        if self.burn is not None:
            if self.burn.mode == 'REBOOST' and float(v @ unit(r)) < model.coast_speed:
                remaining = self.remaining_burn(now, v, g)
                eta = finite_burn_entry(r, v, g, self.burn.acceleration, remaining, radius)
                if math.isfinite(eta):
                    decision = self._burn_decision(r, v, g, now, eta)
                    if decision is not None:
                        return decision
            self.finish_burn(now)
        if self.coast_started is None:
            self.coast_started = now
        closing = float(v @ unit(r))
        if (coast_entry <= 20. or closing > model.coast_speed - model.cruise_speed_band
                or now - self.coast_started < model.cruise_coast_seconds
                or now - self.last_burn_end < model.cruise_coast_seconds):
            return None
        direction = unit(v)
        authority = min(model.planning_accel, model.avoidance_accel)
        net_forward = float((authority * direction + g) @ unit(r))
        if net_forward <= .001:
            return None
        to_target_speed = (model.coast_speed - closing) / net_forward
        durations = sorted({min(duration, to_target_speed, 8.) for duration in (2., 4., 8.)})
        best = None
        self.reburn_advantage = -math.inf
        for duration in durations:
            if duration < .05:
                continue
            acceleration = authority * direction
            eta = finite_burn_entry(r, v, g, acceleration, duration, radius)
            if not math.isfinite(eta):
                continue
            burn_fuel = model.fuel_per_dv * authority * min(duration, eta)
            advantage = model.score(now + eta, fuel + burn_fuel) - model.score(now + coast_entry, fuel)
            self.reburn_advantage = max(self.reburn_advantage, advantage)
            if advantage > model.reburn_score_margin and (best is None or advantage > best[0]):
                best = (advantage, acceleration, duration, eta)
        if best is None:
            return None
        advantage, acceleration, duration, eta = best
        self._start_burn(acceleration, duration, 'REBOOST', now, v, g, advantage)
        return self._burn_decision(r, v, g, now, eta)

    def _course_burn(self, r, v, g, now, delta_v, eta, radius):
        delta_v = nonbraking(delta_v, v, r)
        if self.burn is not None:
            if self.burn.mode == 'TRIM_BURN' and self.burn.acceleration @ delta_v > 0.:
                decision = self._burn_decision(r, v, g, now, eta)
                if decision is not None:
                    return decision
            self.finish_burn(now)
        # Allow a short observation gap after a completed trim, unless arrival is
        # close. Geometric entry hysteresis supplies the normal correction deadband.
        if now - self.last_burn_end < 1.5 and eta > 15.:
            return GuidanceDecision(np.zeros(3), 'TRIM_WAIT', eta)
        authority = min(self.model.planning_accel, self.model.avoidance_accel)
        if authority <= 0. or np.linalg.norm(delta_v) < 1e-9:
            return GuidanceDecision(np.zeros(3), 'TRIM_WAIT', eta)
        acceleration = authority * unit(delta_v)
        candidates = []
        for fraction in (.25, .5, .75, 1.):
            duration = float(np.clip(fraction * np.linalg.norm(delta_v) / authority, .05, 8.))
            entry = finite_burn_entry(r, v, g, acceleration, duration, radius)
            candidates.append((duration, entry))
        safe = [(duration, entry) for duration, entry in candidates if math.isfinite(entry)]
        duration, entry = min(safe) if safe else candidates[-1]
        self._start_burn(acceleration, duration, 'TRIM_BURN', now, v, g)
        return self._burn_decision(r, v, g, now, entry if math.isfinite(entry) else eta)

    def _transit_cutoff(self, r, v, g, now, fuel, entry, radius):
        """Stop the main burn as soon as coasting can meet the arrival window."""
        model = self.model
        direction = unit(r)
        closing, gravity = float(v @ direction), float(g @ direction)
        distance = max(0., float(np.linalg.norm(r)) - radius)
        radial_eta = travel_time(distance, closing, gravity)
        # A lateral miss still needs a course trim. Use radial timing to decide
        # whether more forward speed is worth buying independently of that trim.
        self.coast_arrival = now + entry
        eta = entry if math.isfinite(entry) else radial_eta
        if closing <= 0. or not math.isfinite(eta):
            self.transit_reason = 'acquire closing speed'
            return False
        if now + eta <= model.arrival_target + (2. if self.coast_latched else 0.):
            self.transit_reason = 'on schedule'
            return True
        # Reserve one m/s of corrective impulse plus one telemetry interval
        # of propellant. The measured price includes allocated translation jets.
        if fuel + model.fuel_per_dv + 1. >= model.nominal_fuel:
            self.transit_reason = 'maneuver reserve'
            return True
        # If even full forward thrust cannot meet the deadline, stop when one
        # extra kg buys less than one second. For a ballistic radial intercept
        # d = v*t + g*t^2/2 gives -dt/dv = t/(v + g*t).
        earliest = travel_time(distance, closing, gravity + model.planning_accel)
        entry_speed = closing + gravity * radial_eta
        if (now + earliest > model.arrival_target and math.isfinite(radial_eta)
                and entry_speed > 0. and radial_eta < entry_speed * model.fuel_per_dv):
            self.transit_reason = 'low time saving'
            return True
        self.transit_reason = 'build closing speed'
        return False

    def _score_cutoff(self, r, v, g, now, fuel, entry, radius):
        """Price the next two seconds of forward impulse through target entry."""
        model = self.model
        axis = unit(r)
        self.coast_arrival = now + entry
        # If an unpowered flyby misses laterally, price forward motion on the
        # radial projection. The 3-D capture test still requires a course trim.
        if not math.isfinite(entry):
            r = np.linalg.norm(r) * axis
            v = (v @ axis) * axis
            g = (g @ axis) * axis
            entry = travel_time(np.linalg.norm(r) - radius, v @ axis, g @ axis)
        if not math.isfinite(entry) or entry > 600. or v @ axis <= 0.:
            self.transit_reason = 'acquire closing speed'
            return False
        if fuel + model.fuel_per_dv + 1. >= model.nominal_fuel:
            self.transit_reason = 'maneuver reserve'
            return True
        duration = min(2., entry)
        accel = model.planning_accel * axis
        powered = finite_burn_entry(r, v, g, accel, duration, radius)
        extra_fuel = model.fuel_per_dv * model.planning_accel * min(duration, powered)
        self.reburn_advantage = (model.score(now + powered, fuel + extra_fuel)
                                 - model.score(now + entry, fuel)) if math.isfinite(powered) else -math.inf
        coast = self.reburn_advantage <= (.05 if self.coast_latched else 0.)
        self.transit_reason = 'score favors coast' if coast else 'score favors burn'
        return coast

    def decide(self, relative_position, relative_velocity, gravity, elapsed, fuel, reference, *, nose=None):
        model = self.model
        timed_transit = model.cruise_burns and model.arrival_target is not None
        adaptive = timed_transit or model.score_cruise
        cruise = adaptive or (model.cruise_burns and model.coast_speed is not None)
        distance_now = np.linalg.norm(relative_position)
        self.observe_distance(distance_now)
        if distance_now <= model.success_radius:
            self.finish_burn(elapsed)
            self.coast_arrival, self.transit_reason = elapsed, 'arrived'
            return GuidanceDecision(np.zeros(3), 'ARRIVED', 0.)
        if self.distance_coast:
            self.finish_burn(elapsed)
            entry = ballistic_entry(relative_position, relative_velocity, gravity, model.success_radius, 600.)
            self.coast_arrival, self.transit_reason = elapsed + entry, 'distance cutoff'
            return GuidanceDecision(np.zeros(3), 'DISTANCE_COAST', entry, coast_entry=entry)
        approaching = float(relative_position @ relative_velocity) > 0.
        # Remember reaching the cutoff so gravity slowing a safe coast below it
        # does not repeatedly restart the main burn. A changed trajectory still
        # has to pass the arrival test below, or receive a course correction.
        if not adaptive:
            if not approaching or model.coast_speed is None:
                self.coast_latched = False
            else:
                speed = float(relative_velocity @ unit(relative_position)) if cruise else np.linalg.norm(relative_velocity)
                if speed >= model.coast_speed - (.05 if cruise else 0.):
                    self.coast_latched = True
        # Enter with 8 m radial margin; retain coast with 4 m margin and a larger
        # prediction window to avoid thrust chatter at the edge of the capture test.
        # At the speed cutoff, allow a long coast: the old 90 s limit alone could
        # otherwise force continued acceleration even on an aligned 8 m/s path.
        capture_radius = model.success_radius - (4. if self.ballistic_coasting else 8.)
        horizon = 600. if adaptive or self.coast_latched else (120. if self.ballistic_coasting else 90.)
        coast_entry = ballistic_entry(relative_position, relative_velocity, gravity,
                                       capture_radius, horizon)
        if model.score_cruise:
            self.coast_latched = self._score_cutoff(relative_position, relative_velocity, gravity,
                                                    elapsed, fuel, coast_entry, capture_radius)
        elif timed_transit:
            self.coast_latched = self._transit_cutoff(relative_position, relative_velocity, gravity,
                                                     elapsed, fuel, coast_entry, capture_radius)
        self.ballistic_coasting = (math.isfinite(coast_entry) and approaching
                                  and (not cruise or self.coast_latched or coast_entry <= 8.))
        if self.ballistic_coasting:
            if cruise and not adaptive and coast_entry > 8.:
                decision = self._speed_burn(relative_position, relative_velocity, gravity,
                                             elapsed, fuel, coast_entry, capture_radius, nose)
                if decision is not None:
                    return decision
            else:
                self.finish_burn(elapsed)
            return GuidanceDecision(np.zeros(3), 'FINAL_COAST' if coast_entry <= 8. else 'CLEAR_COAST',
                                    coast_entry, advantage=self.reburn_advantage, coast_entry=coast_entry)
        if self.offset is None and cruise:
            self.offset = np.zeros(3)
        if self.offset is None:
            axis = unit(relative_position)
            side = reference - float(reference @ axis) * axis
            if np.linalg.norm(side) < 1e-6:
                side = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))])
            self.offset = unit(side) * min(model.flyby_offset, .6 * model.success_radius)
        aim = relative_position + self.offset
        direction = unit(aim)
        aim_radius = max(2., model.success_radius - np.linalg.norm(self.offset) - 2.)
        distance = max(0., float(np.linalg.norm(aim)) - aim_radius)
        speed = float(relative_velocity @ direction)
        g_along = float(gravity @ direction)
        coast_eta = travel_time(distance, speed, g_along)
        powered_eta = travel_time(distance, speed, g_along + model.planning_accel)
        lateral_velocity = relative_velocity - speed * direction
        lateral_gravity = gravity - g_along * direction
        # If gravity would stop forward progress short of the target, or arrival
        # is beyond the bounded forecast, more forward impulse is still needed.
        if not math.isfinite(coast_eta) or coast_eta > 600.:
            self.coast_latched = False
        eta = coast_eta if self.coast_latched else powered_eta
        eta = float(np.clip(eta, 1., 600.))
        acceleration = np.zeros(3) if self.coast_latched else model.planning_accel * direction
        # Correct a substantial lateral arrival miss without subtracting forward speed.
        # After the cutoff, use the full coast ETA and steer toward the existing
        # flyby aim point. Do not keep adding forward thrust to fix a lateral miss.
        correction_eta = eta if self.coast_latched else min(eta, 120.)
        lateral_miss = lateral_velocity * correction_eta + .5 * lateral_gravity * correction_eta**2
        if cruise and self.coast_latched:
            delta_v = -(lateral_velocity + .5 * lateral_gravity * correction_eta)
            # Bank forward while correcting the coast intercept. A lateral jet
            # impulse would not be executable by the rear-thrust flight policy.
            delta_v += direction * (np.linalg.norm(delta_v) / math.tan(math.radians(30.)))
            return self._course_burn(relative_position, relative_velocity, gravity, elapsed,
                                     delta_v, eta, capture_radius)
        if cruise:
            self.finish_burn(elapsed)
            self.coast_started = None
        if self.coast_latched or np.linalg.norm(lateral_miss) > aim_radius:
            settle = .25 * correction_eta
            if cruise and not model.score_cruise:
                to_cutoff = max(0., distance_now - (1. - COAST_PROGRESS) * self.launch_distance)
                remaining_burn_time = travel_time(to_cutoff, speed, model.planning_accel + g_along)
                settle = min(settle, .6 * remaining_burn_time)
            correction = -(lateral_velocity + .5 * lateral_gravity * correction_eta) / max(3., settle)
            acceleration += limited(correction, (.58 if cruise else .4) * model.planning_accel)
        # All unpowered coasts are qualified by the 3D entry test above.
        mode = 'CORRECT' if self.coast_latched else 'ACCELERATE'
        acceleration = limited(acceleration, model.planning_accel)
        if cruise and mode == 'ACCELERATE':
            acceleration = nonbraking(acceleration, relative_velocity, relative_position)
        if cruise and elapsed < LAUNCH_PLAN_TIME and mode == 'ACCELERATE':
            acceleration = model.planning_accel * unit(relative_position)
        return GuidanceDecision(acceleration, mode, eta)


def alignment_strength(angle):
    """Smoothly increase alignment priority between 5 and 60 degrees of nose error.

    Bounded at both ends: there is no singular 1/angle gain near correct alignment.
    """
    fraction = float(np.clip((angle - math.radians(5.)) / math.radians(55.), 0., 1.))
    return fraction * fraction * (3. - 2. * fraction)


def pointing_tolerance(distance, avoiding=False):
    """Attitude tolerance only: never relax the geometric arrival/collision tests."""
    near_fraction = float(np.clip((500. - distance) / 400., 0., 1.))
    degrees = 8. + 10. * near_fraction
    return math.radians(max(degrees, 12. if avoiding else 8.))


def angular_command(desired_alpha, omega_body, linear_on, actuator_period, limit, *, align=True,
                    idle_alignment=False):
    """Use gentle spin damping when translation is off, without a minimum torque pulse.

    The installed allocator treats solved activity as on-time. In its unsaturated
    regime an activity is divided by its tick duration to obtain effective acceleration.
    Attenuating idle torque by that period is a feedback-gain calibration, not a
    change to the allocator. It prevents tiny errors from demanding full-tick firings.
    During translation, shared allocation already limits the response.
    """
    if linear_on and align:
        return limited(desired_alpha, limit)
    if idle_alignment and align:
        return limited(desired_alpha, limit) * actuator_period
    if np.linalg.norm(omega_body) <= math.radians(.15):
        return np.zeros(3)
    damping = limited(-np.asarray(omega_body) / 2., limit)
    return damping if linear_on else damping * actuator_period


@dataclass
class Encounter:
    clearance: float = math.inf
    time: float = math.inf
    track: int = -1
    miss_distance: float = math.inf
    uncertainty: float = 0.
    age: float = 0.
    closest_time: float = math.inf
    physical_time: float = math.inf
    hull_gap: float = math.inf       # smallest modeled hull-to-hull gap across all tracks
    intervention_clearance: float | None = None  # nominal path after allowing for later radar reviews


def burn_displacement(times, acceleration, burn_seconds=8., initial_direction=None,
                      turn_rate=math.radians(.25), turn_delay=3.):
    """Finite burn followed by coast, optionally with a delayed change of heading.

    The heading blend approximates a damped turn, rather than instantaneous sideways
    thrust. Integrate its linear acceleration ramp analytically, including burns
    that finish before the turn does. This is a forecast, not a thruster model.
    """
    times = np.asarray(times, dtype=float)
    acceleration = np.asarray(acceleration, dtype=float)
    powered = np.minimum(times, burn_seconds)
    impulse_time = .5 * powered**2 + powered * (times - powered)
    instant = impulse_time[:, None] * acceleration
    magnitude = float(np.linalg.norm(acceleration))
    if initial_direction is None or magnitude < 1e-9:
        return instant, instant
    initial = magnitude * unit(initial_direction)
    angle = math.acos(float(np.clip(initial @ acceleration / magnitude**2, -1., 1.)))
    if angle < 1e-6:
        return instant, instant
    turn_time = angle / max(turn_rate, 1e-6)
    after_delay = np.maximum(powered - turn_delay, 0.)
    after_turn = np.maximum(after_delay - turn_time, 0.)
    ramp_position = (after_delay**3 - after_turn**3) / (6. * turn_time)
    ramp_velocity = (after_delay**2 - after_turn**2) / (2. * turn_time)
    ramp = ramp_position + (times - powered) * ramp_velocity
    delayed = impulse_time[:, None] * initial + ramp[:, None] * (acceleration - initial)
    return delayed, instant


class PredictionScene:
    """Vectorized finite-burn/then-coast paths, including propagated uncertainty.

    Half-second sampling gets a relative-speed margin to cover between-sample misses.
    Unconfirmed tracks use an 8 s horizon while their transverse velocity is unknown.
    """
    def __init__(self, tracks, now, ship_position, ship_velocity, gravity, model, horizon=None,
                 departure=False, terminal=False, lift_axis=None):
        self.model = model
        self.lift_axis = lift_axis
        self.tracks = list(tracks)
        self.departure = departure
        self.moving_departure = (model.low_tilt and departure and lift_axis is not None
                                 and ship_velocity @ lift_axis > .5)
        self.launch_lift = departure and now < LAUNCH_PLAN_TIME
        lookahead = 1.5 * model.horizon if departure else model.horizon
        self.times = np.arange(0., min(lookahead, horizon if horizon is not None else lookahead) + .25, .5)
        states, covariances = [], []
        for track in self.tracks:
            state, covariance = track.predict(now, gravity, model)
            states.append(state)
            covariances.append(covariance)
        if not states:
            self.position = np.empty((0, 3))
            return
        states, self.covariances = np.array(states), np.array(covariances)
        self.position = states[:, :3] - ship_position
        self.velocity = states[:, 3:] - ship_velocity
        self.radii = np.array([track.radius * 1.15 + model.ship_radius + model.clearance for track in self.tracks])
        self.valid = self.times[None, :] <= np.array([
            min(lookahead, 8. + 8. * (track.hits - 1)) for track in self.tracks])[:, None]
        self.ages = np.array([now - track.time for track in self.tracks])
        if terminal or self.moving_departure:
            # Eight seconds on a new track can leave less than one radar
            # revisit plus the time needed to clear a shuttle radius. Look
            # through the second-hit horizon for closing contacts. Keep that
            # horizon until track expiry; uncertainty grows with sample age.
            closing = np.sum(self.position * self.velocity, axis=1) < 0.
            first = np.array([track.hits == 1 for track in self.tracks])
            extend = first & closing
            self.valid |= extend[:, None] & (self.times <= 16.)
        t = self.times[None, :, None, None]
        self.future_covariance = (self.covariances[:, None, :3, :3]
                                  + t * (self.covariances[:, None, :3, 3:] + self.covariances[:, None, 3:, :3])
                                  + t**2 * self.covariances[:, None, 3:, 3:])
        # Broad-phase bound covers EVERY candidate, including its largest possible
        # displacement and the largest covariance eigenvalue (Gershgorin bound).
        # Far safe rocks need not be reevaluated for each lateral direction.
        time = self.times[None, :]
        powered = np.minimum(time, FORWARD_LOOKAHEAD)
        maximum_accel = max(model.planning_accel, model.avoidance_accel)
        if model.score_cruise or model.low_tilt:
            maximum_accel = max(maximum_accel, .25)
        if self.launch_lift:
            maximum_accel = max(maximum_accel, LAUNCH_LIFT_MAX)
        maximum_displacement = maximum_accel * (.5 * powered**2 + powered * (time - powered))
        maximum_variance = np.max(np.sum(np.abs(self.future_covariance), axis=3), axis=2)
        maximum_variance += .25 * model.process_accel_sigma**2 * time**4
        if departure:
            # Extend only well-observed tracks. A distant, diffuse forecast must
            # not obscure an actionable near-field encounter.
            reliable = np.array([track.reliable_hits >= 3 for track in self.tracks])[:, None]
            self.valid &= ((time <= model.horizon) |
                           (reliable & (maximum_variance <= np.maximum(15., self.radii[:, None])**2)))
        maximum_margin = model.uncertainty_sigma * np.sqrt(np.maximum(maximum_variance, 0.))
        maximum_margin += .25 * (np.linalg.norm(self.velocity, axis=1)[:, None] + maximum_accel * (powered + .25))
        unforced_distance = np.linalg.norm(self.position[:, None, :] + self.velocity[:, None, :] * time[:, :, None], axis=2)
        relevant = np.any((unforced_distance <= self.radii[:, None] + maximum_margin + maximum_displacement) & self.valid, axis=1)
        self.tracks = [track for track, keep in zip(self.tracks, relevant) if keep]
        for name in ('position', 'velocity', 'radii', 'valid', 'ages', 'future_covariance'):
            setattr(self, name, getattr(self, name)[relevant])
        self.rays = np.array([tr.last_ray if tr.last_ray is not None else np.zeros(3)
                              for tr in self.tracks]).reshape(-1, 3)
        ray_variance = np.einsum('ni,ntij,nj->nt', self.rays, self.future_covariance, self.rays)
        self.ray_sigma = np.sqrt(np.maximum(ray_variance, 0.) + .25 * model.process_accel_sigma**2 * time**4)

    def assess(self, acceleration, burn_seconds=8., initial_direction=None, *, vector_escape=False):
        if not len(self.position):
            return Encounter()
        if not 0. <= burn_seconds <= FORWARD_LOOKAHEAD:
            raise ValueError('Burn exceeds the forward prediction horizon')
        time = self.times
        powered = np.minimum(time, burn_seconds)
        if self.model.low_tilt:
            # The jet allocator can change the translation vector without slewing the hull.
            initial_direction = None
        delayed, instant = burn_displacement(time, acceleration, burn_seconds, initial_direction,
                                             self.model.avoidance_turn_rate, self.model.avoidance_turn_delay)
        magnitude = np.linalg.norm(acceleration)
        speed_bound = magnitude
        bounds = None
        if self.model.low_tilt:
            # The installed array delivered .085–.145 m/s² within 15° of +Z in
            # sampled allocations. Widen that interval for prediction and use
            # the full escape envelope for oblique or unknown directions.
            near_top = (self.lift_axis is not None
                        and acceleration @ self.lift_axis > magnitude * math.cos(math.radians(15.)))
            bounds = (.07, .18) if near_top else (.05, .25)
        elif self.launch_lift and initial_direction is None:
            bounds = (LAUNCH_LIFT_MIN, LAUNCH_LIFT_MAX)
        elif vector_escape:
            bounds = (.05, .25)
        if bounds is not None and magnitude > 1e-9:
            delayed = instant * min(1., bounds[0] / magnitude)
            speed_bound = max(magnitude, bounds[1])
            instant = instant * (speed_bound / magnitude)
        separation = self.position[:, None, :] + self.velocity[:, None, :] * time[None, :, None]
        separation -= delayed[None, :, :]
        # Check the swept corridor between an immediate turn and the delayed turn.
        # Either timing can be unsafe; do not select a dodge that relies on instant
        # attitude response or misses a different rock halfway through the turn.
        sweep = instant - delayed
        # Doppler constrains motion along the last measured sightline even
        # when transverse velocity is poorly known. Separation along this
        # plane can rule out a collision for the *entire* burn envelope.
        radial = np.einsum('nti,ni->nt', separation, self.rays)
        radial_end = radial - self.rays @ sweep.T
        radial_gap = np.maximum(np.minimum(radial, radial_end), np.minimum(-radial, -radial_end))
        fraction = np.clip(np.einsum('nti,ti->nt', separation, sweep)
                           / np.maximum(np.sum(sweep**2, axis=1), 1e-12)[None, :], 0., 1.)
        separation -= fraction[:, :, None] * sweep[None, :, :]
        distance = np.linalg.norm(separation, axis=2)
        normal = separation / np.maximum(distance[:, :, None], 1e-9)
        variance = np.einsum('nti,ntij,ntj->nt', normal, self.future_covariance, normal)
        variance += .25 * self.model.process_accel_sigma**2 * time[None, :]**4
        uncertainty = self.model.uncertainty_sigma * np.sqrt(np.maximum(variance, 0.))
        # Only motion toward the separating plane can close the gap between
        # samples. A fast tangential flyby need not pay its full speed here.
        burn_speed = speed_bound * (powered + .25)
        if self.model.bottom_lift:
            relative_speed = np.abs(np.einsum('nti,ni->nt', normal, self.velocity))
        else:
            relative_speed = np.linalg.norm(self.velocity, axis=1)[:, None]
        sample_margin = .25 * (relative_speed + burn_speed)
        ray_margin = .25 * (np.abs(np.sum(self.rays * self.velocity, axis=1))[:, None] + burn_speed)
        directional_gap = distance - sample_margin
        doppler_gap = radial_gap - ray_margin
        doppler_gap[~np.any(self.rays, axis=1)] = -math.inf
        clearance = np.maximum(directional_gap - uncertainty,
                               doppler_gap - self.model.uncertainty_sigma * self.ray_sigma) - self.radii[:, None]
        clearance[~self.valid] = math.inf
        intervention = None
        if self.model.bottom_lift:
            # Compute per rock and time before reducing. A distant diffuse
            # return must never mask a different rock about to cross the hull.
            directional = directional_gap - self.model.intervention_padding(np.sqrt(np.maximum(variance, 0.)))
            doppler = doppler_gap - self.model.intervention_padding(self.ray_sigma)
            watched = np.maximum(directional, doppler) - self.radii[:, None]
            intervention = float(np.min(np.where(self.valid, watched, math.inf)))
        row, column = np.unravel_index(np.argmin(clearance), clearance.shape)
        dangerous = clearance < 0.
        first_time = float(time[np.flatnonzero(np.any(dangerous, axis=0))[0]]) if np.any(dangerous) else math.inf
        hull_gap = distance - self.radii[:, None] + self.model.clearance - sample_margin
        hull_gap[~self.valid] = math.inf
        impacts = np.flatnonzero(np.any(hull_gap <= 0., axis=0))
        return Encounter(float(clearance[row, column]), first_time, self.tracks[row].number,
                         float(distance[row, column]), float(uncertainty[row, column]), float(self.ages[row]), float(time[column]),
                         float(time[impacts[0]]) if len(impacts) else math.inf, float(np.min(hull_gap)), intervention)


@dataclass
class AvoidanceDecision:
    acceleration: np.ndarray
    mode: str
    encounter: Encounter
    baseline: Encounter
    urgent: bool = False
    candidates: int = 1
    forward_burn: bool = False
    burn_seconds: float = 8.
    strategy: str = 'GUIDANCE'
    launch_vector: bool = False
    vector_escape: bool = False


class AvoidancePlanner:
    def __init__(self, model):
        self.model = model
        self.active = False
        self.previous = np.zeros(3)
        self.safe_since = None
        self.forward_burn = False
        self.burn_until = -math.inf
        self.short_burn_end = -math.inf

    def _launch(self, scene, nominal, velocity, axis, duration, direct, allow_forward):
        # The existing launch allocator can lift along a world-frame vector
        # even while the vehicle is broadside. Use that same burn for a small
        # deflection. Compare the coast on the same moving-debris timeline.
        coast = scene.assess(np.zeros(3))
        options = [(np.zeros(3), 0., coast)]
        if allow_forward:
            options.append((nominal, duration, direct))
            side = unit(np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))]))
            other = np.cross(axis, side)
            wide_escape = self.model.low_tilt and scene.moving_departure
            angles = (6., 12., 20., 35., 50., 65., 75., 88.) if wide_escape else (6., 12., 20.)
            for degrees in angles:
                # Preserve a shallow path when one clears the uncertainty margin.
                # These are translation angles. The hull keeps its launch reference.
                if degrees == 35. and any(hit.clearance >= 0. for _, _, hit in options):
                    break
                if degrees == 88. and (not self.model.bottom_lift or direct.hull_gap > self.model.clearance
                                       or any(hit.clearance >= 0. for _, _, hit in options)):
                    break
                tangent = math.tan(math.radians(degrees))
                for azimuth in np.arange(8) * math.pi / 4.:
                    tilt = math.cos(azimuth) * side + math.sin(azimuth) * other
                    accel = np.linalg.norm(nominal) * unit(axis + tangent * tilt)
                    if velocity @ axis > 0. and accel @ velocity < 0.:
                        continue
                    options.append((accel, duration, scene.assess(accel, duration)))

        safe = [item for item in options if item[2].clearance >= 0.]
        if safe:
            # Prefer the shallowest clear departure to sitting on a blocked pad.
            pool = safe
        else:
            # First protect the closest physical pass, even if another track
            # dominates the statistical envelope. A sub-metre change is below
            # the radar position noise. Keep the existing heading on near ties.
            gap = max(hit.hull_gap for _, _, hit in options)
            if not (self.model.low_tilt and scene.moving_departure):
                gap = min(gap, self.model.clearance)
            minimum_gap = max(0., gap - .25) if gap >= 0. else gap - .25
            pool = [item for item in options if item[2].hull_gap >= minimum_gap]
            margin = max(item[2].clearance for item in pool)
            pool = [item for item in pool if item[2].clearance >= margin - .5]
        def preference(item):
            accel, seconds, hit = item
            previous_axis = unit(self.previous, axis) if self.active else axis
            aim = unit(accel)
            # While the margin is unresolved, keep the selected side on near
            # ties. A shorter remaining launch burn alone need not undo the turn.
            alignment = round(float(aim @ axis), 6)
            continuity = round(float(aim @ previous_axis), 6) if self.active else alignment
            return (seconds > 0., alignment if safe else continuity,
                    continuity if safe else alignment, hit.clearance)
        accel, seconds, hit = max(pool, key=preference)
        self.active, self.forward_burn = True, False  # rear-only burns are rechecked at handoff
        self.previous = accel.copy()
        strategy = 'LAUNCH_COAST' if not seconds else ('LAUNCH_STRAIGHT' if np.array_equal(accel, nominal)
                                                       else 'LAUNCH_TILT')
        if hit.clearance < 0.:
            strategy += '_LIMITED_CLEARANCE'
        return AvoidanceDecision(accel, 'AVOID', hit, direct, hit.time < 6., len(options),
                                 bool(seconds), seconds, strategy, bool(seconds))

    def choose(self, scene, nominal, velocity, goal, now, *, nose=None, allow_forward=True,
               nominal_burn_seconds=8., rear_guidance=False, launch=False, coast_only=False, plan_on_coast=False,
               transit_burn=False, route_uncertain=False):
        nominal = nonbraking(nominal, velocity, goal)
        if coast_only:
            nominal = np.zeros(3)
        elif launch:
            nominal_burn_seconds = min(FORWARD_LOOKAHEAD, max(0., LAUNCH_PLAN_TIME - now))
        baseline = (scene.assess(nominal, nominal_burn_seconds, initial_direction=nose)
                    if rear_guidance else scene.assess(nominal, nominal_burn_seconds))
        goal_axis = unit(goal)
        if self.model.bottom_lift and (baseline.intervention_clearance is not None
                                      and baseline.intervention_clearance >= 0.):
            # Release as soon as the nominal burn or coast clears. Keep the
            # full uncertainty envelope when testing an avoidance maneuver.
            self.active = self.forward_burn = False
            self.safe_since = None
            strategy = 'MONITOR' if baseline.clearance < 0. else 'GUIDANCE'
            return AvoidanceDecision(nominal, 'GUIDANCE', baseline, baseline, strategy=strategy)
        if launch and not coast_only and baseline.clearance < 0. and np.linalg.norm(nominal) > 0.:
            return self._launch(scene, nominal, velocity, goal_axis, nominal_burn_seconds,
                                baseline, allow_forward)
        if launch or coast_only:
            self.active = self.forward_burn = False
            return AvoidanceDecision(nominal, 'GUIDANCE', baseline, baseline,
                                     strategy='DISTANCE_COAST' if coast_only else 'LAUNCH')
        # A marginal near-term burn or an unresolved crossing farther along the
        # route justifies a handoff pause. Check the coast itself before using it.
        if (plan_on_coast and velocity @ goal_axis > .5
                and (route_uncertain or baseline.clearance < 2. * self.model.clearance)):
            coast = scene.assess(np.zeros(3))
            if coast.clearance >= 0.:
                self.active = self.forward_burn = False
                return AvoidanceDecision(np.zeros(3), 'PLAN_COAST', coast, baseline, strategy='PLAN_COAST')
        if baseline.clearance >= 0.:
            if not self.active:
                return AvoidanceDecision(nominal, 'GUIDANCE', baseline, baseline)
            if baseline.clearance > 3.:
                self.safe_since = now if self.safe_since is None else self.safe_since
                # Keep the retention timer without spending impulse to finish
                # an escape after the nominal trajectory is already safe.
                if now - self.safe_since >= 1.5:
                    self.active = False
                return AvoidanceDecision(nominal, 'GUIDANCE', baseline, baseline)
            else:
                self.safe_since = None
        else:
            self.safe_since = None
        coast = scene.assess(np.zeros(3))
        if coast.clearance >= 0. and velocity @ goal_axis > .5:
            # A blocked eight-second burn can still leave room for a brief,
            # straight impulse. Protect the coast corridor over the entire
            # prediction horizon, including a high delivered thrust estimate.
            if (transit_burn and allow_forward and not scene.departure and nose is not None
                    and nose @ unit(nominal) >= math.cos(math.radians(5.))
                    and nose @ goal_axis > 0. and nose @ velocity > 0.):
                remaining = self.short_burn_end - now
                if remaining <= 0. and now - baseline.age > self.short_burn_end:
                    remaining = 2.
                swept = .25 * remaining * self.model.horizon
                if (remaining > 0. and baseline.age <= self.model.radar_timeout
                        and coast.clearance >= self.model.clearance + swept):
                    accel = np.linalg.norm(nominal) * nose
                    hit = scene.assess(accel, remaining, initial_direction=nose)
                    if hit.clearance >= self.model.clearance:
                        self.previous, self.forward_burn, self.active = accel, True, True
                        self.burn_until = self.short_burn_end = now + remaining
                        return AvoidanceDecision(accel.copy(), 'AVOID', hit, baseline, False, 3,
                                                 True, remaining, 'SHORT_FORWARD')
            self.active, self.forward_burn = True, False
            self.previous = np.zeros(3)
            return AvoidanceDecision(self.previous.copy(), 'AVOID', coast, baseline,
                                     baseline.time < 6., 2, strategy='COAST')
        # Keep an accepted short forward burn stable, but never carry it past its
        # forecast duration without checking the new state and radar observations.
        nose_forward = allow_forward and (nose is None or unit(nose) @ goal_axis >= math.cos(math.radians(35.)))
        if self.active and self.forward_burn and now < self.burn_until and nose_forward:
            remaining = min(FORWARD_LOOKAHEAD, self.burn_until - now)
            encounter = scene.assess(self.previous, remaining, initial_direction=nose)
            if encounter.clearance >= 0.:
                return AvoidanceDecision(self.previous.copy(), 'AVOID', encounter, baseline,
                                         baseline.time < 6., 2, True, remaining, 'FORWARD')

        # Preserve an existing forward course, including the guidance flyby offset.
        # Pointing every boost at the target centre can create a second correction
        # merely to restore the already acceptable arrival trajectory.
        course = unit(velocity) if np.linalg.norm(velocity) > .5 else goal_axis
        if np.linalg.norm(nominal) > .01 and unit(nominal) @ goal_axis > math.cos(math.radians(20.)):
            course = unit(nominal)
        if course @ goal_axis < math.cos(math.radians(20.)):
            course = goal_axis
        course_side = unit(np.cross(course, np.eye(3)[np.argmin(np.abs(course))]))
        course_other = np.cross(course, course_side)
        directions = [course] if np.linalg.norm(course - goal_axis) < 1e-9 else [course, goal_axis]
        for degrees in (5., 12., 20.):
            angle = math.radians(degrees)
            for azimuth in np.arange(8) * math.pi / 4.:
                lateral = math.cos(azimuth) * course_side + math.sin(azimuth) * course_other
                direction = math.cos(angle) * course + math.sin(angle) * lateral
                if direction @ goal_axis >= math.cos(math.radians(20.)) - 1e-12:
                    directions.append(direction)
        evaluated = []
        count = 1
        # Keep the best executable path even if every uncertainty envelope overlaps.
        # Prefer avoiding predicted hull contact, then greater margin at closest pass.
        def fallback_rank(hit):
            if getattr(scene, 'departure', False):
                return (min(hit.hull_gap, self.model.clearance), hit.clearance, hit.physical_time)
            return (hit.physical_time, hit.clearance)
        best_rank = fallback_rank(coast)
        fallback = (np.zeros(3), 0., coast)
        # A rear-thrust maneuver is meaningful only while the nose already faces
        # generally toward the target. Otherwise coast while aligning the nose.
        if nose_forward:
            for direction in directions:
                acceleration = min(self.model.planning_accel, self.model.avoidance_accel) * direction
                if velocity @ goal_axis > 0. and (acceleration @ velocity < 0. or
                                                  (nose is not None and nose @ velocity < 0.)):
                    continue
                for duration in (2., 4., 8., FORWARD_LOOKAHEAD):
                    encounter = scene.assess(acceleration, duration, initial_direction=nose)
                    rank = fallback_rank(encounter)
                    if rank > best_rank:
                        best_rank, fallback = rank, (acceleration, duration, encounter)
                    count += 1
                    if encounter.clearance < 0.:
                        continue
                    # Minimize impulse and subsequent lateral cleanup. All these
                    # planned directions retain cos(20 degrees), about 94%, or more
                    # of their acceleration toward the target.
                    lateral = acceleration - float(acceleration @ course) * course
                    cost = duration * (np.linalg.norm(acceleration) + 2. * np.linalg.norm(lateral))
                    if self.active:
                        cost += .5 * min(duration, 2.) * np.linalg.norm(acceleration - self.previous)
                    tilted = direction @ goal_axis < math.cos(math.radians(2.))
                    evaluated.append(((tilted, cost), acceleration, duration, encounter))
        if evaluated:
            _, acceleration, duration, encounter = min(evaluated, key=lambda item: item[0])
            self.active, self.forward_burn = True, True
            self.previous = acceleration.copy()
            self.burn_until = now + duration
            return AvoidanceDecision(acceleration, 'AVOID', encounter, baseline,
                                     baseline.time < 6., count, True, duration, 'FORWARD')

        count += 1
        if coast.clearance >= 0.:
            self.active, self.forward_burn = True, False
            self.previous = np.zeros(3)
            return AvoidanceDecision(self.previous.copy(), 'AVOID', coast, baseline,
                                     baseline.time < 6., count, strategy='COAST')

        acceleration, duration, encounter = fallback
        if (self.model.score_cruise and allow_forward and now >= LAUNCH_PLAN_TIME
                and coast.hull_gap < self.model.clearance and encounter.hull_gap < self.model.clearance):
            # Test direct jet allocation only when coast and rear-only choices
            # still threaten the hull. Rank the worst pass across all tracks.
            best = None
            for degrees in (25., 45.):
                tangent = math.tan(math.radians(degrees))
                for side in (course_side, -course_side, course_other, -course_other):
                    axis = unit(course + tangent * side)
                    if axis @ goal_axis <= 0. or axis @ velocity < 0.:
                        continue
                    accel = self.model.avoidance_accel * axis
                    for seconds in (8., FORWARD_LOOKAHEAD):
                        hit = scene.assess(accel, seconds, vector_escape=True)
                        count += 1
                        if hit.hull_gap < encounter.hull_gap + 1. or hit.age > self.model.radar_timeout:
                            continue
                        rank = (hit.clearance >= 0., min(hit.hull_gap, self.model.clearance),
                                hit.clearance, float(axis @ unit(self.previous, goal_axis)), -seconds)
                        if best is None or rank > best[0]:
                            best = (rank, accel, seconds, hit)
            if best is not None:
                _, accel, seconds, hit = best
                self.active, self.forward_burn = True, False
                self.previous = accel.copy()
                return AvoidanceDecision(accel, 'AVOID', hit, baseline, hit.time < 6., count,
                                         True, seconds, 'VECTOR_ESCAPE', vector_escape=True)
        self.active, self.forward_burn = True, duration > 0.
        self.previous = acceleration.copy()
        self.burn_until = now + duration
        return AvoidanceDecision(acceleration, 'AVOID', encounter, baseline,
                                 encounter.time < 6., count, self.forward_burn, duration,
                                 'FORWARD_LIMITED_CLEARANCE' if duration else 'COAST_LIMITED_CLEARANCE')


class PulseController:
    """Feedback impulse debt prevents the allocator from turning tiny requests into
    continuous firing. Desired and measured velocity changes use one common frame.
    """
    def __init__(self, start, stop, min_on, command_floor):
        self.start, self.stop, self.min_on, self.command_floor = start, stop, min_on, command_floor
        self.debt = np.zeros(3)
        self.previous_desired = np.zeros(3)
        self.active = False
        self.changed_at = -math.inf
        self.switches = 0

    def reset(self):
        self.debt[:] = 0.
        self.previous_desired[:] = 0.
        self.active = False

    def command(self, desired, measured_delta, dt, now, urgent=False):
        self.debt = limited(self.debt + self.previous_desired * dt - measured_delta, 4. * self.start)
        self.previous_desired = np.array(desired)
        magnitude = float(np.linalg.norm(desired))
        direction = unit(desired)
        aligned_debt = float(self.debt @ direction)
        before = self.active
        if magnitude < 1e-6:
            self.active = False
            self.debt *= math.exp(-dt / .5)
        elif urgent or (not self.active and aligned_debt >= self.start):
            self.active = True
        elif self.active and aligned_debt <= self.stop and now - self.changed_at >= self.min_on:
            self.active = False
        if before != self.active:
            self.changed_at = now
            self.switches += 1
        if not self.active:
            return np.zeros(3)
        return direction * max(magnitude, self.command_floor)


class AccelerationAttitude:
    """Slew the reference toward sustained thrust; hold the reference during coasting."""
    def __init__(self, forward):
        self.forward = unit(forward)

    def update(self, acceleration, dt, pointing_error=None, priority=1.):
        if np.linalg.norm(acceleration) < .005:
            return self.forward.copy()
        desired = unit(acceleration)
        cosine = float(np.clip(self.forward @ desired, -1., 1.))
        angle = math.acos(cosine)
        strength = alignment_strength(pointing_error) * float(np.clip(priority, 0., 1.)) if pointing_error is not None else 0.
        step = min(angle, math.radians(2. + 8. * strength) * dt)
        if angle > 1e-8:
            tangent = desired - cosine * self.forward
            if np.linalg.norm(tangent) < 1e-8:
                tangent = np.cross(self.forward, np.eye(3)[np.argmin(np.abs(self.forward))])
            self.forward = unit(math.cos(step) * self.forward + math.sin(step) * unit(tangent))
        return self.forward.copy()


# Strategic route planning

STEP = 3.0
HORIZON = 120.
TIMES = np.arange(0., HORIZON + STEP, STEP)


@dataclass
class Maneuver:
    name: str
    start: float
    duration: float
    side: np.ndarray
    angle: float = 0.

    @property
    def rear_burn(self):
        return self.name in ('steer', 'advance')

    def set_aim(self, guidance):
        if self.name in ('steer', 'flyby'):
            # Recover toward the same side of the arrival sphere as the early
            # deflection, avoiding a fuel-consuming correction across the route.
            guidance.offset = self.side * min(guidance.model.flyby_offset,
                                               .6 * guidance.model.success_radius)

    def apply(self, nominal, now, goal, model):
        age = now - self.start
        if not 0. <= age < self.duration or nominal.mode in ('FINAL_COAST', 'DISTANCE_COAST', 'ARRIVED'):
            return nominal
        if self.name == 'delay' and nominal.mode == 'ACCELERATE':
            return GuidanceDecision(np.zeros(3), 'ROUTE_COAST', nominal.eta)
        if self.name == 'advance' and nominal.mode == 'CLEAR_COAST':
            nominal = GuidanceDecision(min(model.planning_accel, model.avoidance_accel) * goal, 'ACCELERATE', nominal.eta,
                                       burn_seconds=min(8., self.duration - age))
        if self.angle and nominal.mode == 'ACCELERATE':
            # Follow the selected offset heading during this leg. The ordinary
            # flyby law takes over after release and its cleanup is in the score.
            axis = goal
            side = self.side - (self.side @ axis) * axis
            # Share the existing forward burn. Ease into and out of the heading
            # bias at a rate consistent with the allocated shallow-turn response.
            desired = unit(axis + math.tan(math.radians(self.angle)) * side)
            base = unit(nominal.acceleration)
            difference = math.degrees(math.acos(float(np.clip(base @ desired, -1., 1.))))
            slew = min(.5 * (age + .25), .5 * (self.duration - age), 16.)
            weight = min(1., slew / max(1e-8, difference))
            axis = unit((1. - weight) * base + weight * desired)
            return GuidanceDecision(min(np.linalg.norm(nominal.acceleration), model.avoidance_accel) * axis, nominal.mode, nominal.eta,
                                    burn_seconds=nominal.burn_seconds)
        return nominal


@dataclass
class Forecast:
    positions: np.ndarray            # target-relative displacement since plan start
    eta: float
    fuel: float                      # projected total at arrival
    score: float
    burns: int
    seconds_on: float
    delta_v: float


def forecast(guidance, r, v, gravity, reference, now, fuel, action, nose=None):
    # The rollout owns its latches, trim impulse budget and learned model snapshot.
    guide = copy.deepcopy(guidance)
    model = guide.model
    aim_applied = False
    positions = [np.zeros(3)]
    remaining, speed = r.copy(), v.copy()
    heading = unit(r) if nose is None else nose.copy()
    elapsed = dv = on_time = 0.
    burns, was_on = 0, False
    # Sample risk on the first 120 s, but price the flight through arrival.
    while elapsed < 360.:
        end = elapsed + STEP
        while end - elapsed > 1e-7:
            clock = now + elapsed
            if action is not None and clock >= action.start and not aim_applied:
                action.set_aim(guide)
                aim_applied = True
            decision = guide.decide(remaining, speed, gravity, clock, fuel + model.fuel_per_dv * dv, reference,
                                    nose=heading if clock < LAUNCH_PLAN_TIME else None)
            if action is not None:
                decision = action.apply(decision, clock, unit(remaining), model)
            dt = end - elapsed
            if clock < LAUNCH_PLAN_TIME - 1e-7:
                dt = min(dt, LAUNCH_PLAN_TIME - clock)
            if action is not None and clock < action.start - 1e-7:
                dt = min(dt, action.start - clock)
            if decision.burn_seconds > 1e-6:
                dt = min(dt, decision.burn_seconds)
            if action is not None and clock < action.start + action.duration - 1e-7:
                dt = min(dt, action.start + action.duration - clock)
            accel = decision.acceleration
            goal = unit(remaining)
            route_burn = (action is not None and action.rear_burn and decision.mode == 'ACCELERATE'
                          and action.start <= clock < action.start + action.duration)
            if np.any(accel) and not model.low_tilt:
                desired = unit(accel)
                error = math.acos(float(np.clip(heading @ desired, -1., 1.)))
                turn_time = dt
                rate = math.radians(3.)
                if route_burn:
                    turn_time = max(0., clock + dt - max(clock, action.start + model.avoidance_turn_delay))
                    rate = model.avoidance_turn_rate
                fraction = min(1., turn_time * rate / max(error, 1e-8))
                end_heading = unit(heading + fraction * (desired - heading))
                axis = unit(heading + end_heading)
                if clock >= LAUNCH_PLAN_TIME:
                    accel = rear_thrust(accel, axis, goal, speed)
                heading = end_heading
            step = speed * dt + .5 * (accel + gravity) * dt**2
            cutoff_radius = (1. - COAST_PROGRESS) * guide.launch_distance
            if not model.score_cruise and not guide.distance_coast and np.linalg.norm(remaining - step) <= cutoff_radius:
                crossing = ballistic_entry(remaining, speed, accel + gravity, cutoff_radius, dt)
                if math.isfinite(crossing):
                    dt = crossing
                    step = speed * dt + .5 * (accel + gravity) * dt**2
                    guide.observe_distance(cutoff_radius)
            # Segment/sphere entry interpolates arrival even on a fast flyby.
            square = float(step @ step)
            along = float(remaining @ step)
            disc = along**2 - square * (remaining @ remaining - model.success_radius**2)
            fraction = ((along - math.sqrt(disc)) / square if disc >= 0. and square > 1e-12 else math.inf)
            arrived = np.linalg.norm(remaining) <= model.success_radius or 0. <= fraction <= 1.
            if arrived:
                dt *= float(np.clip(fraction, 0., 1.)) if np.linalg.norm(remaining) > model.success_radius else 0.
                step = speed * dt + .5 * (accel + gravity) * dt**2
            power = float(np.linalg.norm(accel))
            on = power > 1e-8
            burns += int(on and not was_on)
            was_on = on
            dv += power * dt
            on_time += dt if on else 0.
            elapsed += dt
            remaining -= step
            speed += (gravity + accel) * dt
            if arrived:
                if len(positions) < len(TIMES):
                    positions.append(r - remaining)
                total = fuel + dv * model.fuel_per_dv
                return Forecast(np.array(positions), elapsed, total, model.score(now + elapsed, total),
                                burns, on_time, dv)
        if len(positions) < len(TIMES):
            positions.append(r - remaining)
        elif guide.distance_coast and not math.isfinite(decision.eta):
            break
    return Forecast(np.array(positions), math.inf, math.inf, -math.inf, burns, on_time, dv)


@dataclass
class Risk:
    clearance: float = math.inf
    miss: float = math.inf
    t: float = math.inf
    sigma: float = math.inf
    track: int = -1
    threats: int = 0
    threat_ids: frozenset = frozenset()
    unresolved: frozenset = frozenset()
    approaching_at_horizon: frozenset = frozenset()
    intervention_clearance: float | None = None

    @property
    def needs_detour(self):
        return (self.clearance if self.intervention_clearance is None else self.intervention_clearance) < 0.

    def clears(self, direct):
        # Moving a predicted encounter just beyond its credible time horizon
        # cannot by itself demonstrate an improvement.
        return (self.clearance >= 0. and not self.approaching_at_horizon.intersection(direct.threat_ids)
                and not self.unresolved.intersection(direct.threat_ids)
                and not self.unresolved.difference(direct.unresolved))


class RadarField:
    def __init__(self, tracks, now, origin, target_velocity, gravity, model):
        self.model = model
        tracks = [k for k in tracks if k.hits >= 2 and 0. <= now - k.time <= model.track_timeout
                  and k.reliable_hits >= 2]
        self.ids = np.array([k.number for k in tracks])
        if not tracks:
            return
        state = np.array([k.state for k in tracks])
        age = np.array([now - k.time for k in tracks])[:, None]
        dt = age + TIMES
        vel = state[:, 3:] - target_velocity
        self.xyz = ((state[:, None, :3] - origin) + state[:, None, 3:] * age[:, :, None]
                    + vel[:, None, :] * TIMES[:, None] + .5 * gravity * dt[:, :, None]**2)
        self.radius = np.array([k.radius * 1.15 + model.ship_radius + model.clearance for k in tracks])
        p = np.array([k.covariance for k in tracks])
        # Upper bound on the largest eigenvalue of each propagated 3x3 block.
        # Reuse the existing process acceleration with independent 4 s intervals
        # at long range (constant unknown acceleration within the first interval).
        t = dt[:, :, None, None]
        cov = p[:, None, :3, :3] + t * (p[:, None, 3:, :3] + p[:, None, :3, 3:]) + t*t*p[:, None, 3:, 3:]
        process_var = model.process_accel_sigma**2 * np.minimum(.25 * dt**4, 4./3. * dt**3)
        self.cov = cov
        self.process_var = process_var
        self.sigma = np.sqrt(np.maximum(0., np.max(np.sum(np.abs(cov), axis=-1), axis=-1) + process_var))

    def assess(self, path):
        if not len(self.ids) or len(path.positions) < 2:
            return Risk()
        n = len(path.positions)
        xyz = self.xyz[:, :n]
        # The last sample can be a fraction of a step when the shuttle arrives.
        if path.eta < TIMES[n-1]:
            xyz = xyz.copy()
            f = np.clip((path.eta - TIMES[n-2]) / STEP, 0., 1.)
            xyz[:, -1] = xyz[:, -2] + f * (xyz[:, -1] - xyz[:, -2])
        sep = xyz - path.positions
        delta = np.diff(sep, axis=1)
        approaching = frozenset()
        if path.eta > HORIZON:
            closing = np.sum(sep[:, -1] * delta[:, -1], axis=1) < 0.
            approaching = frozenset(self.ids[closing])
        frac = np.clip(-np.sum(sep[:, :-1]*delta, axis=2) /
                       np.maximum(1e-12, np.sum(delta*delta, axis=2)), 0., 1.)
        closest = sep[:, :-1] + frac[:, :, None]*delta
        miss = np.linalg.norm(closest, axis=2)
        # For a spherical obstacle the plane perpendicular to closest separation
        # bounds the whole segment. Project the track covariance onto that normal.
        # Check both time endpoints and retain the broad-axis confidence gate.
        bound = np.maximum(self.sigma[:, 1:n], self.sigma[:, :n-1])
        # A very diffuse forecast cannot justify fuel expenditure on a specific
        # side of a rock. The tactical layer still sees these observations.
        diffuse = bound > np.maximum(25., self.radius[:, None])
        unresolved = frozenset(self.ids[np.any(diffuse & (miss <= self.radius[:, None] + 25.), axis=1)])
        upper = miss - self.radius[:, None] - .125 * max(self.model.planning_accel, self.model.avoidance_accel) * STEP**2
        upper[diffuse] = math.inf
        least_upper = np.min(upper)
        if not math.isfinite(least_upper):
            return Risk(unresolved=unresolved, approaching_at_horizon=approaching)
        # Project covariance only for segments that could be a threat or the
        # global closest clearance. The broad-axis sigma bounds every projection.
        lower = upper - self.model.uncertainty_sigma * bound
        rows, cols = np.nonzero(lower <= max(0., least_upper))
        distances = miss[rows, cols]
        normal = closest[rows, cols] / np.maximum(distances[:, None], 1e-8)
        def projected(times):
            return (np.einsum('ni,nij,nj->n', normal, self.cov[rows, times], normal)
                    + self.process_var[rows, times])
        sigma = np.sqrt(np.maximum(0., np.maximum(projected(cols), projected(cols + 1))))
        sigma = np.where(distances < 1e-8, bound[rows, cols], sigma)
        clearance = upper[rows, cols] - self.model.uncertainty_sigma * sigma
        intervention = None
        if self.model.bottom_lift:
            intervention = float(np.min(upper[rows, cols] - self.model.intervention_padding(sigma)))
        threats = frozenset(self.ids[rows[clearance < 0.]])
        index = int(np.argmin(clearance))
        i, j = rows[index], cols[index]
        time = TIMES[j] + frac[i, j] * min(STEP, path.eta - TIMES[j])
        return Risk(float(clearance[index]), float(miss[i,j]), float(time), float(sigma[index]),
                    int(self.ids[i]), len(threats), threats, unresolved, approaching, intervention)

    def improves_close_pass(self, candidate, direct):
        """Allow early progress on one close pass whose full uncertainty margin is still infeasible."""
        if (direct.threats != 1 or candidate.track != direct.track
                or candidate.threat_ids != direct.threat_ids or candidate.unresolved
                or direct.track in candidate.approaching_at_horizon):
            return False
        radius = self.radius[self.ids == direct.track][0]
        return (direct.miss <= radius and candidate.miss >= direct.miss + 3.
                and candidate.clearance >= direct.clearance + 3.)


class Search:
    """Evaluate a few alternatives per control update on one frozen radar field."""
    def __init__(self, choices, field, direct_risk, guidance, r, v, g, reference, now, fuel, nose):
        self.choices = list(choices)
        self.field = field
        self.direct_risk = direct_risk
        self.args = (copy.deepcopy(guidance), r.copy(), v.copy(), g.copy(), reference.copy(), now, fuel)
        self.nose = nose.copy()
        self.start = now
        self.best = None
        self.rank = ()

    def advance(self):
        count = min(2, len(self.choices))
        for _ in range(count):
            action = self.choices.pop(0)
            path = forecast(*self.args, action, self.nose)
            straight = action.name in ('delay', 'advance') and not action.angle
            if not math.isfinite(path.score):
                continue
            risk = self.field.assess(path)
            clear = risk.clears(self.direct_risk)
            if not clear and not self.field.improves_close_pass(risk, self.direct_risk):
                continue
            if self.field.model.score_cruise:
                rank = (clear, 0. if clear else risk.clearance, path.score, straight)
            else:
                rank = (clear, straight if clear else False, 0. if clear else risk.clearance, path.score)
            if rank > self.rank:
                self.best, self.rank = action, rank
        return count


class RoutePlanner:
    def __init__(self):
        self.action = None
        self.pending = None
        self.next_review = 0.
        self.reason = 'acquiring'
        self.risk = self.direct_risk = Risk()
        self.path = None
        self.extra_fuel = 0.
        self.track_count = self.candidates = 0
        self.search = None
        self.cruise_reviewed = False
        self.cruise_reason = 'launch default'
        self.review_time = None

    def interrupt(self, now, reason):
        self.action = self.path = None
        self.pending = None
        self.search = None
        self.reason = reason
        self.risk = self.direct_risk = Risk()
        self.extra_fuel = 0.
        self.track_count = self.candidates = 0
        self.next_review = now + 1.
        self.review_time = None

    def command(self, nominal, guidance, tracks, now, origin, r, v, target_velocity, gravity,
                reference, nose, fuel, known_empty=False):
        goal = unit(r)
        if self.action is not None and now >= self.action.start + self.action.duration:
            self.action = None
            self.next_review = now
        winner = self.pending
        if self.search is not None:
            if now - self.search.start <= 4.:
                self.candidates += self.search.advance()
                if self.search.choices:
                    return nominal
                winner = self.search.best
            self.search = None
            self.next_review = now if winner is not None else now + 1.
            self.reason = 'no safe detour' if winner is None else 'rechecking route'
        if now >= self.next_review:
            self.next_review = now + 2.
            self.review_time = now
            field = RadarField(tracks, now, origin, target_velocity, gravity, guidance.model)
            self.track_count = len(field.ids)
            if not self.track_count and not known_empty:
                self.interrupt(now, 'acquiring tracks')
                return nominal
            def evaluate(action):
                path = forecast(guidance, r, v, gravity, reference, now, fuel, action, nose)
                return path, field.assess(path)
            direct, risk = evaluate(None)
            # Price the cruise speed once during early ascent. Use the slower
            # option only if it has a clear observed corridor and a score margin.
            if (not guidance.model.score_cruise and guidance.model.arrival_target is None and not self.cruise_reviewed
                    and now >= 4. and nominal.mode == 'ACCELERATE'
                    and guidance.model.coast_speed == 10. and v @ goal < 4.):
                self.cruise_reviewed = True
                economy = copy.deepcopy(guidance)
                economy.model.coast_speed = 9.
                trial = forecast(economy, r, v, gravity, reference, now, fuel, None, nose)
                trial_risk = field.assess(trial)
                if (math.isfinite(trial.eta) and trial_risk.clears(risk)
                        and (risk.clearance < 0. or trial.score > direct.score + 1.)):
                    guidance.model.coast_speed = 9.
                    direct, risk = trial, trial_risk
                    self.cruise_reason = '9 m/s clear and preferred'
                else:
                    self.cruise_reason = 'retain 10 m/s'
            # The two launch burns are physically identical before the lower
            # cutoff. A newly resolved crossing can change which coast is safe.
            # Recover the faster schedule only when its whole observed path clears.
            if (self.cruise_reviewed and guidance.model.coast_speed == 9. and not guidance.coast_latched
                    and self.action is None and nominal.mode == 'ACCELERATE' and 8. <= v @ goal < 8.95
                    and risk.clearance < 0.):
                faster = copy.deepcopy(guidance)
                faster.model.coast_speed = 10.
                trial = forecast(faster, r, v, gravity, reference, now, fuel, None, nose)
                trial_risk = field.assess(trial)
                if math.isfinite(trial.eta) and trial_risk.clears(risk):
                    guidance.model.coast_speed = 10.
                    direct, risk = trial, trial_risk
                    self.cruise_reason = '10 m/s clears updated crossing'
            self.direct_risk = risk
            self.candidates = 1
            old = self.action
            self.action, self.path, self.risk, self.extra_fuel = None, direct, risk, 0.
            self.reason = 'direct clear'
            if risk.needs_detour:
                self.reason = 'no safe detour'
            elif risk.unresolved or risk.clearance < 0.:
                self.reason = 'monitoring uncertain tracks'
            if winner is not None and risk.needs_detour:
                # Several updates elapsed during the search. Re-anchor the winning
                # leg and validate it on the current state and radar before use.
                winner.start = max(now, LAUNCH_PLAN_TIME)
                path, current = evaluate(winner)
                self.pending = None
                clear = current.clears(risk)
                if (clear or field.improves_close_pass(current, risk)) and math.isfinite(path.eta):
                    self.path, self.risk = path, current
                    if now < LAUNCH_PLAN_TIME:
                        self.pending = winner
                        self.next_review = min(self.next_review, LAUNCH_PLAN_TIME)
                        self.reason = 'planned for 20 s'
                    else:
                        self.action = winner
                        self.reason = 'predicted intersection' if clear else 'improving close pass'
                        winner.set_aim(guidance)
            elif not risk.needs_detour:
                self.pending = None
            if (old is not None and now < old.start + old.duration
                    and (not guidance.model.bottom_lift or risk.needs_detour)):
                path, current = evaluate(old)
                self.candidates += 1
                if (math.isfinite(path.eta) and (current.clears(risk) or field.improves_close_pass(current, risk))
                        and (risk.clearance < 3. or direct.score < path.score + .5)):
                    self.action, self.path, self.risk = old, path, current
                    self.reason = 'retain safe route' if current.clears(risk) else 'improving close pass'
            if (self.action is None and winner is None and risk.needs_detour and risk.t > 8.
                    and nominal.mode in ('ACCELERATE', 'CLEAR_COAST') and math.isfinite(direct.eta)):
                side = unit(np.cross(goal, np.eye(3)[np.argmin(np.abs(goal))]))
                other = np.cross(goal, side)
                coast = nominal.mode == 'CLEAR_COAST'
                choices = []
                for duration in (3., 6.):
                    if coast and nose @ goal >= math.cos(math.radians(20.)):
                        choices.append(Maneuver('advance', now, duration, side))
                    elif not coast and min(v @ goal, (v + gravity*duration) @ goal) > .5:
                        choices.append(Maneuver('delay', now, duration, side))
                if not coast:
                    choices.extend(Maneuver('flyby', now, direct.eta, lateral)
                                   for lateral in (side, -side, other, -other))
                if not coast and nose @ goal >= math.cos(math.radians(35.)):
                    choices.extend(Maneuver('steer', now, duration, lateral, angle)
                                   for lateral in (side, -side, other, -other)
                                   for duration, angle in ((30., 6.), (80., 14.)))
                if coast and nose @ goal >= math.cos(math.radians(20.)):
                    choices.extend(Maneuver('advance', now, 16., lateral, 6.)
                                   for lateral in (side, -side, other, -other))
                for choice in choices:
                    choice.start = max(now, LAUNCH_PLAN_TIME)
                self.search = Search(choices, field, risk, guidance, r, v, gravity, reference, now, fuel, nose)
                self.reason = 'checking alternatives'
            self.extra_fuel = self.path.fuel - direct.fuel if math.isfinite(direct.fuel) else 0.
        if self.action is None or now >= self.action.start + self.action.duration:
            return nominal
        return self.action.apply(nominal, now, goal, guidance.model)


# Flight telemetry

class FlightRecorder:
    fields = ('t', 'mode', 'no_debris_test', 'radar_observation_practice', 'radar_avoidance_enabled',
              'distance', 'speed', 'closing_speed', 'fuel', 'damage', 'score_if_arrived',
              'eta', 'full_thrust', 'bottom_lift_burn', 'thrust_from_lift_axis_deg',
              'coast_speed', 'transit_reason', 'coast_arrival_s',
              'speed_cutoff_reached', 'velocity_error_deg',
              'guidance_burn_s', 'guidance_burn_plans', 'reburn_advantage',
              'route', 'route_reason', 'cruise_reason', 'launch_burn', 'clear_departure', 'turn_rate_limit_deg_s',
              'pending_route', 'progress', 'distance_coast',
              'route_tracks', 'route_candidates', 'route_eta_s',
              'route_review_t', 'route_ms', 'route_unresolved',
              'route_fuel_kg', 'route_score', 'route_delta_fuel_kg', 'route_burns', 'route_burn_s',
              'route_dv', 'route_threats', 'route_track', 'route_cpa_s', 'route_miss_m', 'route_sigma_m',
              'route_direct_clearance_m', 'route_intervention_clearance_m', 'route_clearance_m',
              'route_selected_miss_m', 'route_selected_sigma_m',
              'route_selected_cpa_s', 'route_angle_deg',
              'route_leg_left_s', 'route_side_x', 'route_side_y', 'route_side_z',
              'flyby_offset_x', 'flyby_offset_y', 'flyby_offset_z',
              'avoidance_strategy', 'avoidance_burn_s', 'forward_avoidance', 'body_lateral_command',
              'nominal_mode', 'forward_command',
              'delivered_forward_accel', 'thrust_power', 'braking', 'angular_command_norm',
              'nose_error_deg', 'alignment_priority', 'angular_accel_limit', 'angular_brake_limit',
              'nose_tolerance_deg', 'quiet_coast', 'coast_entry_s',
              'tracks', 'confirmed', 'radar_age',
              'rejected', 'ambiguous', 'threat_id', 'ttc', 'cpa_time', 'predicted_miss',
              'clearance', 'uncertainty', 'threat_age', 'baseline_clearance', 'intervention_clearance',
              'hull_gap', 'baseline_hull_gap', 'candidates',
              'attitude_error_deg', 'hull_rotation_deg', 'spin_deg_s', 'linear_on', 'angular_on', 'linear_switches',
              'angular_switches', 'coast_fraction', 'fuel_per_dv', 'planning_accel', 'planner_ms',
              'x', 'y', 'z', 'vx', 'vy', 'vz', 'gx', 'gy', 'gz',
              'nose_x', 'nose_y', 'nose_z',
              'target_x', 'target_y', 'target_z', 'target_vx', 'target_vy', 'target_vz',
              'desired_ax', 'desired_ay', 'desired_az', 'command_ax', 'command_ay', 'command_az')

    def __init__(self, root, model):
        root = Path(root)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        self.directory = root / 'diagnostics' / stamp
        self.directory.mkdir(parents=True, exist_ok=False)
        self.telemetry_file = (self.directory / 'telemetry.csv').open('w', newline='', encoding='utf-8')
        self.radar_file = (self.directory / 'radar.csv').open('w', newline='', encoding='utf-8')
        self.telemetry = csv.DictWriter(self.telemetry_file, fieldnames=self.fields)
        self.telemetry.writeheader()
        self.radar = csv.writer(self.radar_file)
        self.radar.writerow(('received_t', 'measurement_t', 'rel_x', 'rel_y', 'rel_z',
                             'surface_range', 'range_rate', 'diameter'))
        source = Path(__file__)
        hashes = {source.name: hashlib.sha256(source.read_bytes()).hexdigest()} if source.is_file() else {}
        (self.directory / 'model.json').write_text(
            json.dumps(dict(model=asdict(model), source_sha256=hashes,
                            units='metres, seconds, kilograms; Phobos axes',
                            evidence='Flight telemetry; predictions are estimates, not debris truth'), indent=2),
            encoding='utf-8')

    def observations(self, now, contacts):
        for contact in contacts:
            self.radar.writerow((now, contact.t_meas, *contact.rel_pos,
                                 contact.surface_range, contact.range_rate, contact.diameter))

    def record(self, values):
        self.telemetry.writerow({key: values.get(key, '') for key in self.fields})
        self.telemetry_file.flush()
        self.radar_file.flush()

    def close(self):
        for stream in (self.telemetry_file, self.radar_file):
            try:
                stream.close()
            except OSError:
                pass                       # diagnostics must not stop control on a full disk


# Main forward burns on clear paths and radar-directed departure burns.
ATTITUDE_GAIN = 0.15
MAX_TURN_RAD_S = math.radians(3.0)
MAX_ANG_ACCEL = 0.01
ANG_RATE_TIME_S = 1.5
ATTITUDE_TOL_RAD = math.radians(1.5)
ANG_RATE_TOL_RAD_S = math.radians(0.15)
RADAR_COUNT = 500
PLANNING_INTERVAL_S = 0.25
STATUS_INTERVAL_S = 1.0
ACTUATOR_PERIOD_S = 1.0 / 60.0  # installed JetSelectionActuator LoopFreqHz; idle torque calibration


def LimitVector(vector: npt.NDArray, limit: float) -> npt.NDArray:
    magnitude = float(np.linalg.norm(vector))
    return vector * min(1.0, limit / max(magnitude, 1e-12))


def MakeTargetAttitude(forward: npt.NDArray, roll_reference: npt.NDArray) -> npt.NDArray:
    """Rows are perpendicular target body axes expressed in Phobos coordinates.

    Carry the previous desired y-axis forward to avoid unnecessary roll during a dodge.
    """
    x_axis = forward / np.linalg.norm(forward)
    y_axis = roll_reference - float(roll_reference @ x_axis) * x_axis
    if np.linalg.norm(y_axis) < 1e-6:
        reference = np.eye(3)[np.argmin(np.abs(x_axis))]
        y_axis = reference - float(reference @ x_axis) * x_axis
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    return np.vstack((x_axis, y_axis, z_axis))


def RotationVector(rotation: npt.NDArray) -> npt.NDArray:
    """Shortest axis-angle vector of an active rotation matrix, including 180 degrees.

    Working with a matrix makes this independent of quaternion sign conventions.
    """
    sine_axis = 0.5 * np.array([rotation[2, 1] - rotation[1, 2],
                                rotation[0, 2] - rotation[2, 0],
                                rotation[1, 0] - rotation[0, 1]])
    sine = float(np.linalg.norm(sine_axis))
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.atan2(sine, cosine)
    if cosine > 0.0 and sine < 1e-8:
        return sine_axis
    if cosine < 0.0 and sine < 1e-5:
        # The skew part vanishes at 180 degrees. Its eigenvector with eigenvalue 1
        # still gives the rotation axis; choose a deterministic sign at exactly pi.
        values, axes = np.linalg.eigh(0.5 * (rotation + rotation.T))
        axis = axes[:, np.argmax(values)]
        if sine > 1e-10:
            if float(axis @ sine_axis) < 0.0:
                axis = -axis
        elif axis[np.argmax(np.abs(axis))] < 0.0:
            axis = -axis
        return angle * axis
    return (angle / sine) * sine_axis


def ComputeAttitudeAcceleration(current_axes, target_axes, angular_velocity, accel_limit,
                                *, gain_scale=1.0, rate_limit=None, brake_limit=None, tolerance=None):
    """Return body angular acceleration and attitude error in radians.

    Each input matrix has body axes in its ROWS, expressed in Phobos coordinates.
    current_axes @ target_axes.T is the desired ACTIVE rotation in current body axes.
    """
    error = RotationVector(current_axes @ target_axes.T)
    angle = float(np.linalg.norm(error))
    tolerance = ATTITUDE_TOL_RAD if tolerance is None else tolerance
    if angle <= tolerance and np.linalg.norm(angular_velocity) <= ANG_RATE_TOL_RAD_S:
        return np.zeros(3), angle
    desired_rate = ATTITUDE_GAIN * gain_scale * error * max(0.0, 1.0 - tolerance / max(angle, 1e-12))
    desired_rate = LimitVector(desired_rate, MAX_TURN_RAD_S if rate_limit is None else rate_limit)
    acceleration = (desired_rate - angular_velocity) / ANG_RATE_TIME_S
    # Preserve braking authority as pointing error shrinks while the shuttle is
    # still rotating. Weakening damping here would cause overshoot and fuel reversal.
    if brake_limit is not None and float(acceleration @ angular_velocity) < 0.0:
        accel_limit = max(accel_limit, brake_limit)
    return LimitVector(acceleration, accel_limit), angle


# Competition scenes use the bottom jets with live radar protection. Observation
# only is an explicit comparison mode, independent of the nominal thrust profile.
test_mode_params = ("NoDebrisTestMode", "RadarObservationPractice", "BottomLiftTestMode", "LowTiltTestMode")
straight_flight = this.HasParam("StraightFlight") and this.GetParam(st.VarType.bool, "StraightFlight")
no_debris_test_mode = (this.GetParam(st.VarType.bool, "NoDebrisTestMode")
                       if this.HasParam("NoDebrisTestMode") else False)
radar_observation_practice = (this.HasParam("RadarObservationPractice")
                              and this.GetParam(st.VarType.bool, "RadarObservationPractice"))
obstacle_free_guidance = straight_flight or no_debris_test_mode or radar_observation_practice
if radar_observation_practice:
    st.logger_info("Radar observation practice active. Guidance follows the clear-lane trajectory.")
elif no_debris_test_mode:
    st.OnScreenLogMessage("No-debris TEST mode: guidance enabled without radar observations",
                          "Guidance", st.Severity.Info)
elif straight_flight:
    st.logger_info("Straight-flight strategy active. Radar is observation only.")

# Read only own-ship properties and permitted target/planet state. Never read debris truth.
linear_limit = shuttle.GetParam(st.VarType.double, ["Thrusters", "MaxAccel_m_s2"])
angular_limit = shuttle.GetParam(st.VarType.double, ["Thrusters", "MaxAngAccel_rad_s2"])
if not (np.isfinite(linear_limit) and linear_limit > 0.0
        and np.isfinite(angular_limit) and angular_limit > 0.0):
    SetCommand(np.zeros(3), np.zeros(3))
    raise ValueError("Invalid shuttle acceleration limits")
bottom_lift = (this.GetParam(st.VarType.bool, "BottomLiftTestMode")
               if this.HasParam("BottomLiftTestMode") else
               straight_flight or not any(this.HasParam(p) for p in test_mode_params))
model = MissionModel(planning_accel=min(.12, .5 * linear_limit),
                     avoidance_accel=min(.09, linear_limit),
                     cruise_burns=True,
                     score_cruise=True,
                     bottom_lift=bottom_lift,
                     low_tilt=bottom_lift or (this.HasParam("LowTiltTestMode") and this.GetParam(st.VarType.bool, "LowTiltTestMode")),
                     fuel_per_dv=1.3 * shuttle.getMass() / (80.0 * 9.8))
if model.low_tilt:
    profile = "Bottom-lift" if bottom_lift else "Low-tilt"
    st.OnScreenLogMessage(f"{profile} mode active. Hull angles are measured from the launch pose.",
                          "Guidance", st.Severity.Info)
try:
    phobos_mu = phobos.GetParam(st.VarType.double, ["Dynamics", "GravitationalParameter_km3_s2"]) * 1e9
except Exception:
    phobos_mu = 0.0                    # coasting observations still estimate total external acceleration
history = StateHistory()
tracker = RadarTracker(model)
guidance = AccelerationGuidance(model)
routes = RoutePlanner()
avoidance = AvoidancePlanner(model)
linear_switches = 0
angular_switches = 0
attitude_reference = None
launch_axes = None
roll_reference = None
previous_time = None
previous_velocity = None
previous_omega_world = None
previous_linear_command = np.zeros(3)
previous_angular_command = np.zeros(3)
external_residual = np.zeros(3)
lift_residual = np.zeros(3)
bottom_burn = False
lift_bias = math.radians(BOTTOM_LIFT_BIAS_DEG)
next_plan = -math.inf
next_status = -math.inf
last_fault_log = -math.inf
last_radar_fault_log = -math.inf
command = np.zeros(3)
mode = "ACQUIRE"
plan = None
avoidance_burn_until = -math.inf
nominal = None
mission_complete = False
quiet_coast = False
alignment_engaged = True
coast_seconds = 0.0
total_seconds = 0.0
fuel_used = 0.0
damage = 0.0
fuel_observer_fuel = 0.0
fuel_observer_dv = 0.0
fuel_observer_on_seconds = 0.0
fuel_observer_rotation = 0.0
planner_ms = 0.0
route_ms = 0.0
recorder = None
try:
    recorder = FlightRecorder(controller_directory, model)
    st.OnScreenLogMessage(f"Flight diagnostics: {recorder.directory}", "Guidance", st.Severity.Info)
except Exception as error:
    st.OnScreenLogMessage(f"Flight recording unavailable: {error}", "Guidance", st.Severity.Warning)

t0_sim = st.SimGlobals.GetSimEntity().GetParam(st.VarType.timestamp, ["Time", "StartDateTAI"])
t0 = t0_sim.as_datetime()
exit_flag = False
while not exit_flag:
    time.sleep(1.0 / this.GetParam(st.VarType.double, "LoopFreqHz"))
    now = (st.SimGlobals.SimClock.GetTimeNow().as_datetime() - t0).total_seconds()
    dt = 1.0 / this.GetParam(st.VarType.double, "LoopFreqHz") if previous_time is None else now - previous_time
    if dt <= 0.0:
        continue
    previous_time = now
    state = shuttle.getLocVelAcc()
    loc = np.asarray(state.loc_WRT_ExprIn(phobosFrame), dtype=float)
    vel = np.asarray(state.vel_WRT_ExprIn(phobosFrame), dtype=float)
    target_state = target.getLocVelAcc()
    target_loc = np.asarray(target_state.loc_WRT_ExprIn(phobosFrame), dtype=float)
    target_vel = np.asarray(target_state.vel_WRT_ExprIn(phobosFrame), dtype=float)
    current_axes = np.vstack([
        st.frames.FramedVector(axis, shuttleFrame).ExprIn(phobosFrame) for axis in np.eye(3)
    ])
    omega_body = np.asarray(shuttle.getAngVelocity().WRT(phobosFrame).ExprIn(shuttleFrame), dtype=float)
    if not np.all(np.isfinite(np.r_[loc, vel, target_loc, target_vel, current_axes.ravel(), omega_body])):
        SetCommand(np.zeros(3), np.zeros(3))
        previous_velocity = previous_omega_world = None
        previous_linear_command = previous_angular_command = np.zeros(3)
        if now - last_fault_log > 2.0:
            st.OnScreenLogMessage("Invalid vehicle state: commands cleared", "Guidance", st.Severity.Warning)
            last_fault_log = now
        continue
    omega_world = current_axes.T @ omega_body
    history.add(now, loc, vel)
    central_gravity = -phobos_mu * loc / max(float(np.linalg.norm(loc)), 1.0)**3
    observed_dv = np.zeros(3) if previous_velocity is None else vel - previous_velocity
    observed_dw = np.zeros(3) if previous_omega_world is None else omega_world - previous_omega_world
    # Learn Mars/other external acceleration only during coast intervals. Reject impact
    # impulses; do not mistake commanded thrust for gravity or filter the truth states.
    if previous_velocity is not None and np.linalg.norm(previous_linear_command) < 1e-9 and dt < .5:
        residual = observed_dv / dt - central_gravity
        if np.linalg.norm(residual - external_residual) < .04:
            external_residual += -math.expm1(-dt / 2.0) * (residual - external_residual)
            if bottom_lift and now < LAUNCH_PLAN_TIME:
                lift_residual = external_residual.copy()
    gravity = central_gravity + external_residual
    if (bottom_burn and now < LAUNCH_PLAN_TIME and previous_velocity is not None and 0. < dt < .5
            and np.linalg.norm(vel - target_vel) > .25 and np.linalg.norm(previous_linear_command) > .01):
        # The actuator's linear impulse follows the command. Its unknown
        # magnitude drops out of the transverse gravity measurement.
        axis = previous_linear_command / np.linalg.norm(previous_linear_command)
        residual = observed_dv / dt - central_gravity - lift_residual
        transverse = residual - (residual @ axis) * axis
        if np.linalg.norm(transverse) < .02:
            lift_residual += -math.expm1(-dt / 3.) * transverse
    measured_thrust_dv = observed_dv - gravity * dt if previous_velocity is not None else np.zeros(3)
    if np.linalg.norm(previous_linear_command) > 0.0:
        fuel_observer_dv += float(np.linalg.norm(measured_thrust_dv))
        fuel_observer_on_seconds += dt
    fuel_observer_rotation += float(np.linalg.norm(observed_dw))
    previous_velocity, previous_omega_world = vel.copy(), omega_world.copy()
    total_seconds += dt
    if np.linalg.norm(previous_linear_command) == 0.0 and np.linalg.norm(previous_angular_command) == 0.0:
        coast_seconds += dt
    relative_position = target_loc - loc
    relative_velocity = vel - target_vel
    distance = float(np.linalg.norm(relative_position))
    goal = relative_position / max(distance, 1e-9)
    was_distance_coast = guidance.distance_coast
    guidance.observe_distance(distance)
    if guidance.distance_coast and not was_distance_coast:
        next_plan = now
        next_status = now
    if launch_axes is None:
        launch_axes = current_axes.copy()
        roll_reference = current_axes[1].copy()
        if model.low_tilt:
            pad_target_axes = low_tilt_target(launch_axes, goal)
        else:
            attitude_reference = AccelerationAttitude(current_axes[0])
    thrust_axis = current_axes[2] if model.low_tilt else current_axes[0]
    if bottom_lift:
        lift_body = np.array([math.sin(lift_bias), 0., math.cos(lift_bias)])
        lift_direction = current_axes.T @ lift_body
    # Success is position-only. Latch entry so passing the target does not trigger a
    # costly reversal/flip after the scorer has already accepted the run.
    if not mission_complete and distance <= model.success_radius:
        mission_complete = True
        next_status = now  # Record measured time and fuel on the first observed entry.
    if now >= next_plan and not mission_complete:
        started = time.perf_counter()
        quiet_coast = False
        bottom_burn = False
        try:
            observations = []
            if radar_observation_practice or not no_debris_test_mode:
                try:
                    contacts = GetRadarContacts(RADAR_COUNT)
                    observations = tracker.update(contacts, now, history, gravity)
                except Exception as error:
                    if not obstacle_free_guidance:
                        raise
                    if now - last_radar_fault_log > 2.0:
                        st.logger_warn(f"Radar observation unavailable ({error})")
                        last_radar_fault_log = now
            if recorder is not None:
                try:
                    recorder.observations(now, observations)
                except Exception as error:
                    st.OnScreenLogMessage(f"Flight recording stopped: {error}", "Guidance", st.Severity.Warning)
                    recorder.close()
                    recorder = None
            nominal = guidance.decide(relative_position, relative_velocity, gravity, now,
                                       fuel_used, roll_reference, nose=thrust_axis)
            guidance_tracks = () if obstacle_free_guidance else tracker.tracks.values()
            guidance_radar_ready = obstacle_free_guidance or tracker.fresh(now)
            route_started = time.perf_counter()
            handoff = LAUNCH_PLAN_TIME <= now < LAUNCH_PLAN_TIME + 2.
            if (guidance_radar_ready and not avoidance.active
                    and (handoff or np.linalg.norm(omega_body) <= MAX_TURN_RAD_S)
                    and nominal.mode in ("ACCELERATE", "CLEAR_COAST")):
                try:
                    nominal = routes.command(nominal, guidance, guidance_tracks, now, loc,
                                             relative_position, relative_velocity, target_vel, gravity,
                                             roll_reference, thrust_axis, fuel_used, known_empty=obstacle_free_guidance)
                except Exception as error:
                    # Tactical assessment below can still use the current radar tracks.
                    routes.interrupt(now, "route forecast unavailable")
                    if now - last_fault_log > 2.:
                        st.OnScreenLogMessage(f"Route forecast unavailable: {error}", "Guidance", st.Severity.Warning)
                        last_fault_log = now
            else:
                routes.interrupt(now, "nominal guidance or tactical priority")
            route_ms = 1000. * (time.perf_counter() - route_started)
            if not guidance_radar_ready:
                guidance.finish_burn(now)
                stale = now - tracker.last_fresh > model.radar_timeout
                nominal.acceleration = np.zeros(3)
                nominal.mode = "SENSOR_COAST" if stale else "ACQUIRE"
                nominal.burn_seconds, nominal.advantage = 0., 0.
            launch = now < LAUNCH_PLAN_TIME and nominal.mode == "ACCELERATE"
            departure = now < LAUNCH_PLAN_TIME or distance > .90 * guidance.launch_distance
            horizon = guidance.departure_horizon(distance, relative_velocity @ goal) if departure else model.horizon
            scene = PredictionScene(guidance_tracks, now, loc, vel, gravity, model,
                                    horizon=min(horizon, nominal.eta + 1.0), departure=departure,
                                    lift_axis=current_axes[2] if model.low_tilt else None,
                                    terminal=model.score_cruise and distance <= .30 * guidance.launch_distance)
            if bottom_lift and nominal.mode == "ACCELERATE" and routes.action is None:
                # Propose the efficient burn first so radar checks the actual
                # thrust vector. Only an executed detour can override this leg.
                demand = np.linalg.norm(nominal.acceleration)
                if (demand > 0.
                        and lift_direction @ nominal.acceleration >= demand * math.cos(math.radians(8.))
                        and (relative_velocity @ goal <= 0. or lift_direction @ relative_velocity >= 0.)):
                    if now < LAUNCH_PLAN_TIME:
                        aim = departure_burn_direction(relative_position, relative_velocity,
                                                       central_gravity + lift_residual, now, model)
                        pad_target_axes = low_tilt_target(launch_axes, aim, BOTTOM_LIFT_BIAS_DEG)
                    nominal.acceleration = demand * lift_direction
                    bottom_burn = True
            # Pad lift is expressed in world axes and already works during the
            # initial turn. Rear-only burns need the stricter attitude limit.
            turn_limit = (2. if launch else 1.) * MAX_TURN_RAD_S
            plan = avoidance.choose(scene, nominal.acceleration, relative_velocity, goal, now,
                                    nose=thrust_axis, allow_forward=(nominal.mode not in ("ACQUIRE", "SENSOR_COAST")
                                    and np.linalg.norm(omega_body) <= turn_limit),
                                    nominal_burn_seconds=nominal.burn_seconds or 8.,
                                    rear_guidance=not launch and np.linalg.norm(nominal.acceleration) > 0.,
                                    transit_burn=(nominal.mode == "ACCELERATE" and
                                                  guidance.transit_reason == "score favors burn"),
                                    launch=launch, coast_only=guidance.distance_coast,
                                    plan_on_coast=handoff and not bottom_lift and routes.action is None and nominal.mode == "ACCELERATE",
                                    route_uncertain=(routes.path is None or bool(routes.direct_risk.unresolved)
                                                     or routes.direct_risk.clearance < 2. * model.clearance))
            if plan.mode == "AVOID":
                bottom_burn = False
                guidance.finish_burn(now)  # avoidance may preempt any scheduled nominal burn
                routes.interrupt(now, "tactical override")
            avoidance_burn_until = now + plan.burn_seconds if plan.forward_burn else -math.inf
            command = plan.acceleration.copy()
            mode = nominal.mode if plan.mode == "GUIDANCE" else plan.mode
            quiet_coast = (nominal.mode in ("CLEAR_COAST", "FINAL_COAST", "ROUTE_COAST")
                           and guidance_radar_ready and np.linalg.norm(command) < 1e-9
                           and plan.encounter.clearance >= 3.)
            if quiet_coast:
                mode = nominal.mode
        except Exception as error:
            bottom_burn = False
            guidance.finish_burn(now)
            routes.interrupt(now, "planning fault")
            command = np.zeros(3)
            mode = "SENSOR_COAST"
            plan = None
            if now - last_fault_log > 2.0:
                st.OnScreenLogMessage(f"Guidance degraded: {error}", "Guidance", st.Severity.Warning)
                last_fault_log = now
        planner_ms = 1000.0 * (time.perf_counter() - started)
        next_plan = now + PLANNING_INTERVAL_S
        if now < LAUNCH_PLAN_TIME and plan is not None and plan.mode == "AVOID":
            next_plan = min(next_plan, LAUNCH_PLAN_TIME)
    if guidance.distance_coast:
        guidance.finish_burn(now)
        command = np.zeros(3)
        mode, quiet_coast = "DISTANCE_COAST", True
    if mission_complete:
        guidance.finish_burn(now)
        command = np.zeros(3)
        mode = "ARRIVED"
    guidance_radar_ready = obstacle_free_guidance or tracker.fresh(now)
    forward_ready = guidance_radar_ready and np.linalg.norm(omega_body) <= MAX_TURN_RAD_S
    timed_burn = mode in ("TRIM_BURN", "REBOOST") and not mission_complete
    if timed_burn and (not forward_ready or guidance.remaining_burn(now, relative_velocity, gravity) <= 1e-6):
        # Check both elapsed time and observed delivered impulse EVERY control tick,
        # not just at the slower guidance-update rate. Never leave a stale burn on.
        guidance.finish_burn(now)
        command = np.zeros(3)
        mode, timed_burn = "TRIM_WAIT", False
    angled_launch = (plan is not None and plan.launch_vector and mode == "AVOID" and now < LAUNCH_PLAN_TIME
                     and now < avoidance_burn_until and guidance_radar_ready
                     and np.linalg.norm(omega_body) <= 2. * MAX_TURN_RAD_S)
    forward_avoidance = (plan is not None and plan.forward_burn and not plan.launch_vector and mode == "AVOID"
                         and now < avoidance_burn_until and not mission_complete
                         and forward_ready)
    vector_escape = forward_avoidance and plan.vector_escape
    full_thrust = (mode == "ACCELERATE" or timed_burn or forward_avoidance or angled_launch) and not mission_complete
    route_rear = (mode == "ACCELERATE" and routes.action is not None and routes.action.rear_burn)
    launch_burn = (mode == "ACCELERATE" and now < LAUNCH_PLAN_TIME) or angled_launch
    rear_burn = full_thrust and not launch_burn
    # Lift away from the pad before asking for a faster pitch. Evaluate the
    # same measured corridor in both scenarios, including radar freshness.
    clear_departure = (launch_burn and mode == "ACCELERATE" and relative_velocity @ goal > .1
                       and guidance_radar_ready and plan is not None
                       and plan.baseline.clearance >= 2. * model.clearance)
    pointing_axis = command / max(np.linalg.norm(command), 1e-12) if full_thrust else goal
    nose_error = math.acos(float(np.clip(current_axes[0] @ pointing_axis, -1., 1.)))
    if model.low_tilt:
        # Translate in the radar-cleared direction with the whole jet array. The
        # fixed hull reference is independent of both guidance and avoidance.
        route_live = not route_rear or now < routes.action.start + routes.action.duration
        spin_ok = np.linalg.norm(omega_body) <= (2. if launch_burn else 1.) * MAX_TURN_RAD_S
        permit = guidance_radar_ready and spin_ok
        linear_world = limited(command, linear_limit) if full_thrust and route_live and permit else np.zeros(3)
        if bottom_burn and mode == "ACCELERATE" and np.any(linear_world):
            # Follow body +Z during the gentle slew between radar updates.
            linear_world = np.linalg.norm(linear_world) * lift_direction
        rotation_error = RotationVector(current_axes @ pad_target_axes.T)
        attitude_error = float(np.linalg.norm(rotation_error))
        alignment_limit = braking_limit = min(angular_limit, .012)
        rate_limit = math.radians(1.2)
        alignment_priority = 0.
        desired_rate = limited(rotation_error * (.10 / .75), rate_limit)
        alpha_command = limited(.75 * (desired_rate - omega_body), alignment_limit)
        if not np.any(linear_world):
            alpha_command *= ACTUATOR_PERIOD_S
        if mission_complete or (attitude_error < math.radians(.15) and np.linalg.norm(omega_body) < math.radians(.004)):
            alpha_command = np.zeros(3)
    else:
        nose_tolerance = math.radians(2.) if rear_burn else pointing_tolerance(distance, False)
        if nose_error <= nose_tolerance:
            alignment_engaged = False
        elif nose_error > nose_tolerance + math.radians(1. if rear_burn else 4.):
            alignment_engaged = True
        # A new contact can arrive mid-slew. If spin currently rules out an angled
        # escape, damp it first so the next radar decision regains that option.
        recovery = np.linalg.norm(omega_body) > 2. * MAX_TURN_RAD_S
        arrest_launch_turn = now < LAUNCH_PLAN_TIME and mode == "AVOID" and recovery
        # Use a coasting handoff or a blocked forward corridor to finish pointing.
        align_nose = (alignment_engaged and not quiet_coast and not mission_complete and not vector_escape
                      and not arrest_launch_turn
                      and (full_thrust or mode in ("PLAN_COAST", "AVOID")))
        phase_priority = 1. if align_nose else 0.
        alignment_priority = alignment_strength(nose_error) * phase_priority
        reference_accel = (command if full_thrust else goal * model.planning_accel) if align_nose else np.zeros(3)
        desired_forward = attitude_reference.update(reference_accel, dt, pointing_error=nose_error,
                                                     priority=phase_priority)
        target_axes = MakeTargetAttitude(desired_forward, roll_reference)
        roll_reference = target_axes[1].copy()
        # At large errors the old slow slew spent extra propellant combining pitch
        # with broadside lift. Taper the rate with pointing error and keep damping
        # authority as the nose approaches the burn direction.
        max_turn_accel = .05 if clear_departure else .03
        rate_limit = math.radians(3. + (8. if clear_departure else 2.) * alignment_priority)
        alignment_limit = min(angular_limit, .03 if recovery else
                              (.005 + (max_turn_accel - .005) * alignment_priority
                               if full_thrust else (.008 if align_nose else .005)))
        braking_limit = min(angular_limit, .03 if recovery or full_thrust else (.008 if align_nose else .005))
        alpha_body, attitude_error = ComputeAttitudeAcceleration(
            current_axes, target_axes, omega_body, alignment_limit,
            gain_scale=1.0 + 2.0 * alignment_priority if full_thrust else .75,
            rate_limit=rate_limit, brake_limit=braking_limit,
            # Drive into the acceptance cone, then disengage above. Using the same
            # tolerance for both would approach its boundary asymptotically forever.
            tolerance=.5 * nose_tolerance)
        if mission_complete:
            alpha_body = np.zeros(3)
        linear_world = np.zeros(3)
        if vector_escape:
            linear_world = command.copy()
        elif launch_burn:
            # The pad starts broadside. Preserve outward lift while aligning.
            if guidance_radar_ready:
                # Once lift is established on a clear corridor, let the allocator
                # give more authority to rotation while sustaining outward thrust.
                lift_request = min(.12, linear_limit) if clear_departure else linear_limit
                linear_world = (pointing_axis if angled_launch else goal) * lift_request
        elif rear_burn and forward_ready:
            axis = current_axes[0]
            route_live = not route_rear or now < routes.action.start + routes.action.duration
            if route_live:
                linear_world = rear_thrust(command, axis, goal, relative_velocity)
        linear_world = limited(linear_world, linear_limit)
        alpha_command = angular_command(alpha_body, omega_body, np.linalg.norm(linear_world) > 0.,
                                         ACTUATOR_PERIOD_S, max(alignment_limit, braking_limit), align=align_nose,
                                         idle_alignment=align_nose)
        if mission_complete or guidance.distance_coast or (quiet_coast and not recovery):
            alpha_command = np.zeros(3)
    angular_world = current_axes.T @ alpha_command
    if bottom_burn and mode == "ACCELERATE":
        # Preserve the allocator request even if the live frame advances during planning.
        linear_body = np.linalg.norm(linear_world) * lift_body
    else:
        linear_body = st.frames.FramedVector(linear_world, phobosFrame).ExprIn(shuttleFrame)
    SetCommand(linear_body, current_axes @ angular_world)
    linear_switches += int((np.linalg.norm(linear_world) > 0.) != (np.linalg.norm(previous_linear_command) > 0.))
    angular_switches += int((np.linalg.norm(angular_world) > 0.) != (np.linalg.norm(previous_angular_command) > 0.))
    previous_linear_command = linear_world.copy()
    previous_angular_command = angular_world.copy()
    if now >= next_status:
        fuel_used = shuttle.GetParam(st.VarType.double, ["Stats", "FuelUsed"])
        damage = shuttle.GetParam(st.VarType.double, ["Stats", "Damage"])
        ideal_fuel_per_dv = shuttle.getMass() / (80.0 * 9.8)
        learn_burn_model(model, fuel_used - fuel_observer_fuel, fuel_observer_dv,
                        fuel_observer_rotation, ideal_fuel_per_dv, fuel_observer_on_seconds)
        model.planning_accel = min(model.planning_accel, .5 * linear_limit)
        fuel_observer_fuel, fuel_observer_dv, fuel_observer_rotation = fuel_used, 0.0, 0.0
        fuel_observer_on_seconds = 0.0
        encounter = plan.encounter if plan is not None else None
        hull_rotation_deg = math.degrees(np.linalg.norm(RotationVector(current_axes @ launch_axes.T)))
        command_magnitude = np.linalg.norm(linear_world)
        lift_angle = (math.degrees(math.acos(float(np.clip(current_axes[2] @ linear_world / command_magnitude, -1., 1.))))
                      if command_magnitude > 0. else None)
        st.OnScreenLogMessage(
            f"t={now:6.1f}s {mode:12s} dist={distance:7.1f}m speed={np.linalg.norm(vel):4.1f}m/s "
            f"fuel={fuel_used:6.1f}kg damage={damage:.0f}J tracks={len(tracker.tracks)} "
            f"clearance={encounter.clearance if encounter else math.inf:6.1f}m "
            f"ttc={encounter.time if encounter else math.inf:4.1f}s spin={math.degrees(np.linalg.norm(omega_body)):4.2f}deg/s "
            f"nose={math.degrees(nose_error):4.1f}deg hull={hull_rotation_deg:.1f}deg "
            f"coast={coast_seconds / max(total_seconds, .01):.0%} plan={planner_ms:.0f}ms",
            "Guidance", st.Severity.Info)
        if recorder is not None:
            values = dict(t=now, mode=mode, distance=distance, speed=np.linalg.norm(vel), closing_speed=relative_velocity @ goal,
                          no_debris_test=int(no_debris_test_mode),
                          radar_observation_practice=int(radar_observation_practice),
                          radar_avoidance_enabled=int(not obstacle_free_guidance),
                          bottom_lift_burn=int(bottom_burn and mode == "ACCELERATE" and command_magnitude > 0.),
                          thrust_from_lift_axis_deg=lift_angle,
                          fuel=fuel_used, damage=damage, score_if_arrived=model.score(now, fuel_used, damage),
                          eta=nominal.eta if nominal else math.inf,
                          full_thrust=int(full_thrust), coast_speed=model.coast_speed,
                          transit_reason="arrived" if mission_complete else guidance.transit_reason,
                          coast_arrival_s=None if mission_complete else guidance.coast_arrival,
                          route=routes.action.name if routes.action else "direct", route_reason=routes.reason,
                          cruise_reason=guidance.transit_reason if model.score_cruise else routes.cruise_reason,
                          launch_burn=int(launch_burn), clear_departure=int(clear_departure),
                          turn_rate_limit_deg_s=math.degrees(rate_limit),
                          pending_route=routes.pending.name if routes.pending is not None else '',
                          progress=1. - distance / max(guidance.launch_distance, 1e-9), distance_coast=int(guidance.distance_coast),
                          route_review_t=routes.review_time, route_ms=route_ms, route_unresolved=len(routes.risk.unresolved),
                          route_tracks=routes.track_count, route_candidates=routes.candidates,
                          route_eta_s=routes.path.eta if routes.path else None,
                          route_fuel_kg=routes.path.fuel if routes.path else None,
                          route_score=routes.path.score if routes.path else None,
                          route_delta_fuel_kg=routes.extra_fuel, route_burns=routes.path.burns if routes.path else None,
                          route_burn_s=routes.path.seconds_on if routes.path else None,
                          route_dv=routes.path.delta_v if routes.path else None,
                          route_threats=routes.direct_risk.threats, route_track=routes.direct_risk.track,
                          route_cpa_s=routes.direct_risk.t, route_miss_m=routes.direct_risk.miss,
                          route_sigma_m=routes.direct_risk.sigma, route_clearance_m=routes.risk.clearance,
                          route_direct_clearance_m=routes.direct_risk.clearance, route_selected_miss_m=routes.risk.miss,
                          route_intervention_clearance_m=routes.direct_risk.intervention_clearance,
                          route_selected_sigma_m=routes.risk.sigma, route_selected_cpa_s=routes.risk.t,
                          route_angle_deg=routes.action.angle if routes.action else 0.,
                          route_leg_left_s=max(0., routes.action.start + routes.action.duration - now) if routes.action else 0.,
                          guidance_burn_s=guidance.remaining_burn(now, relative_velocity, gravity),
                          guidance_burn_plans=guidance.burn_count,
                          reburn_advantage=(guidance.reburn_advantage
                                            if guidance.transit_reason in ("score favors burn", "score favors coast")
                                            else None),
                          avoidance_strategy=plan.strategy if plan else "GUIDANCE",
                          avoidance_burn_s=max(0., avoidance_burn_until - now) if forward_avoidance or angled_launch else 0.,
                          forward_avoidance=int(forward_avoidance),
                          body_lateral_command=float(np.linalg.norm(np.asarray(linear_body)[1:])),
                          speed_cutoff_reached=int(guidance.coast_latched),
                          velocity_error_deg=(math.degrees(math.acos(float(np.clip(
                              relative_velocity @ goal / np.linalg.norm(relative_velocity), -1., 1.))))
                              if np.linalg.norm(relative_velocity) > 1e-6 else math.nan),
                          nominal_mode=nominal.mode if nominal else "ACQUIRE", forward_command=float(linear_world @ goal),
                          delivered_forward_accel=float(measured_thrust_dv @ goal / dt),
                          thrust_power=float(linear_world @ relative_velocity),
                          braking=int(float(linear_world @ relative_velocity) < -.001),
                          angular_command_norm=float(np.linalg.norm(alpha_command)),
                          nose_error_deg=math.degrees(nose_error), alignment_priority=alignment_priority,
                          hull_rotation_deg=hull_rotation_deg,
                          nose_tolerance_deg=None if model.low_tilt else math.degrees(nose_tolerance),
                          quiet_coast=int(quiet_coast),
                          coast_entry_s=nominal.coast_entry if nominal else math.inf,
                          angular_accel_limit=alignment_limit, angular_brake_limit=braking_limit,
                          tracks=len(tracker.tracks), confirmed=sum(track.hits >= 2 for track in tracker.tracks.values()),
                          radar_age=now - tracker.last_fresh, rejected=tracker.stats.get('rejected', 0),
                          ambiguous=tracker.stats.get('ambiguous', 0), threat_id=encounter.track if encounter else -1,
                          ttc=encounter.time if encounter else math.inf, cpa_time=encounter.closest_time if encounter else math.inf,
                          predicted_miss=encounter.miss_distance if encounter else math.inf,
                          clearance=encounter.clearance if encounter else math.inf, uncertainty=encounter.uncertainty if encounter else 0.,
                          threat_age=encounter.age if encounter else 0., baseline_clearance=plan.baseline.clearance if plan else math.inf,
                          intervention_clearance=plan.baseline.intervention_clearance if plan else None,
                          hull_gap=encounter.hull_gap if encounter else math.inf,
                          baseline_hull_gap=plan.baseline.hull_gap if plan else math.inf,
                          candidates=plan.candidates if plan else 0, attitude_error_deg=math.degrees(attitude_error),
                          spin_deg_s=math.degrees(np.linalg.norm(omega_body)), linear_on=int(np.linalg.norm(linear_world) > 0.),
                          angular_on=int(np.linalg.norm(angular_world) > 0.), linear_switches=linear_switches,
                          angular_switches=angular_switches, coast_fraction=coast_seconds / max(total_seconds, .01),
                          fuel_per_dv=model.fuel_per_dv, planning_accel=model.planning_accel, planner_ms=planner_ms)
            for names, vector in ((('x','y','z'), loc), (('vx','vy','vz'), vel), (('gx','gy','gz'), gravity),
                                  (('nose_x','nose_y','nose_z'), current_axes[0]),
                                  (('route_side_x','route_side_y','route_side_z'),
                                   routes.action.side if routes.action else np.zeros(3)),
                                  (('flyby_offset_x','flyby_offset_y','flyby_offset_z'),
                                   guidance.offset if guidance.offset is not None else np.zeros(3)),
                                  (('target_x','target_y','target_z'), target_loc),
                                  (('target_vx','target_vy','target_vz'), target_vel),
                                  (('desired_ax','desired_ay','desired_az'), command),
                                  (('command_ax','command_ay','command_az'), linear_world)):
                values.update(zip(names, vector))
            try:
                recorder.record(values)
            except Exception as error:
                st.OnScreenLogMessage(f"Flight recording stopped: {error}", "Guidance", st.Severity.Warning)
                recorder.close()
                recorder = None
        next_status = now + STATUS_INTERVAL_S

if recorder is not None:
    recorder.close()
st.leave_sim()
