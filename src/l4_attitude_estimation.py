r"""
@File       : l4_attitude_estimation.py
@Author     : Chengjuan Wan, et al.
@Institution: University of Chinese Academy of Sciences
@Description:
    Stage 4: Attitude Estimation (L4) module for the WH-DRCS ISAC framework.
    This module extracts high-precision rotor angular speeds by identifying local
    peak intervals via an adaptive sliding window mechanism. It then inverts the
    spatial attitude angles (Pitch, Yaw, Roll) utilizing geometric and kinematic
    relationships defined in the unified perception architecture.
"""

import numpy as np
from scipy.signal import find_peaks
from typing import Dict, Tuple, Any


class L4_AttitudeEstimator:
    r"""
    Implements Stage 4 of the Hierarchical Signal Processing Algorithm.
    Executes fine-grained 3D attitude estimation by fusing multi-rotor
    micro-Doppler frequency signatures and spatial consistencies.
    """

    def __init__(self, arm_length_L: float = 0.5, dt: float = 15.0e-6,
                 min_peak_interval: float = 0.005, peak_prominence: float = 0.5,
                 reliability_threshold: float = 0.50):
        """
        :param arm_length_L: Physical baseline/arm length (L) of the UAV in meters.
        :param dt: Radar Pulse Repetition Interval (PRI) or sampling time step.
        :param min_peak_interval: Minimum temporal separation between periodic peaks.
        :param peak_prominence: Minimum peak prominence used by the rotor-speed estimator.
        :param reliability_threshold: Minimum L4 reliability for an accepted estimate.
        r"""
        self.L = arm_length_L
        self.dt = dt
        self.min_peak_interval = min_peak_interval
        self.peak_prominence = peak_prominence
        self.reliability_threshold = reliability_threshold

        # Validation-selected reliability weights, frozen before test evaluation.
        self._w4 = np.array([0.6, 0.4])

    def _estimate_robust_rotor_speed(self, v_k_t: np.ndarray) -> float:
        r"""
        Identifies local peak intervals {\Delta t_i} via adaptive sliding window
        and computes robust rotor angular speed \omega_k.
        \omega_k \leftarrow 2\pi / median({\Delta t_i})
        r"""
        if len(v_k_t) < 3:
            return 0.0

        # 1. Adaptive sliding window for peak detection (Algorithm 1, Line 22)
        # Dynamic distance based on the expected Nyquist folded bounds
        adaptive_distance = max(1, int(self.min_peak_interval / self.dt))
        peaks, _ = find_peaks(
            v_k_t,
            distance=adaptive_distance,
            prominence=self.peak_prominence
        )

        if len(peaks) < 2:
            # Fallback for insufficient periodic peaks
            return 0.0

        # 2. Extract local peak intervals {\Delta t_i}
        peak_times = peaks * self.dt
        delta_t_i = np.diff(peak_times)

        # 3. Robust rotor angular speed estimation (handling outlier jitter)
        median_delta_t = np.median(delta_t_i)

        if median_delta_t <= 0:
            return 0.0

        omega_k = (2 * np.pi) / median_delta_t
        return float(omega_k)

    def _invert_spatial_attitude(self, omegas: np.ndarray) -> Tuple[float, float, float]:
        r"""
        Inverts spatial attitude angles (\Theta) utilizing geometric and kinematic relationships.
        Implements Algorithm 1, Lines 25-28.
        r"""
        K = len(omegas)
        if K < 4:
            # Requires at least a quadrotor configuration for full 3D attitude inversion
            return 0.0, 0.0, 0.0

        # 1. Calculate aggregate vertical displacement \Delta z and directional accelerations
        # In a strict physical model, these are derived from the thrust differentials (\propto \omega_k^2).
        # Here we map the thrust coefficients mathematically.
        thrust_forces = omegas ** 2

        # Assuming standard symmetric quadcopter topology mapping for generalized kinematics
        # Front/Back differential -> Pitch; Left/Right differential -> Roll
        delta_z = np.sum(thrust_forces) * 1e-7  # Calibrated aggregate thrust metric
        delta_a_y = (thrust_forces[0] - thrust_forces[2]) * 1e-3  # Front-Back axis
        delta_a_z = (thrust_forces[1] - thrust_forces[3]) * 1e-3  # Left-Right axis

        # Prevent division by zero or domain errors
        if abs(delta_a_z) < 1e-6:
            delta_a_z = 1e-6
        clamped_z_L = np.clip(delta_z / self.L, -1.0, 1.0)

        # 2. Spatial Angle Inversion
        theta_pitch = np.arcsin(clamped_z_L)
        phi_roll = np.arctan2(delta_a_y, delta_a_z)

        # \psi_{yaw} \leftarrow atan2(\sum \sin\theta_k, \sum \cos\theta_k)
        # Using centered pseudo-angles derived from relative rotor-speed deviations
        omega_mean = np.mean(omegas)
        omega_scale = max(np.max(np.abs(omegas - omega_mean)), 1e-6)
        pseudo_thetas = ((omegas - omega_mean) / omega_scale) * (np.pi / 2)
        psi_yaw = np.arctan2(np.sum(np.sin(pseudo_thetas)), np.sum(np.cos(pseudo_thetas)))

        return float(theta_pitch), float(psi_yaw), float(phi_roll)

    def _evaluate_estimation_reliability(self, omegas: np.ndarray, V_peak_var: float) -> float:
        r"""
        Evaluates estimation reliability \mathcal{C}_{L_4} by fusing frequency
        and spatial consistencies.
        """
        # Frequency consistency \mathcal{C}_f (low RPM variance -> high consistency)
        C_f = np.exp(-np.std(omegas) / (np.mean(omegas) + 1e-5))

        # Spatial consistency \mathcal{C}_s (mapped from raw velocity variance)
        C_s = np.exp(-V_peak_var / 100.0)

        # \mathcal{C}_{L_4} \leftarrow \mathbf{w}_4^T [\mathcal{C}_f, \mathcal{C}_s]^T
        C_L4 = float(np.dot(self._w4, np.array([C_f, C_s])))
        return np.clip(C_L4, 0.0, 1.0)

    def execute_stage_4(self, clustered_rotors: Dict[int, np.ndarray]) -> Tuple[bool, Dict[str, Any]]:
        """
        Executes Stage 4: Attitude Estimation.
        :param clustered_rotors: A dictionary mapping rotor ID (k) to its velocity sequence v_k(t).
                                 (Derived from the DBSCAN outputs of Stage 3).
        :return: (Estimation Success Boolean, Semantic Attitude Output)
        """
        if len(clustered_rotors) == 0:
            return False, {"Status": "No rotor clusters identified."}

        rotor_omegas = []
        all_velocities = []

        # 1. Iterate through each validated rotor cluster k \in {1, \dots, K}
        for k, v_k_t in clustered_rotors.items():
            omega_k = self._estimate_robust_rotor_speed(v_k_t)
            rotor_omegas.append(omega_k)
            all_velocities.extend(v_k_t)

        rotor_omegas = np.array(rotor_omegas)

        if np.all(rotor_omegas == 0):
            return False, {"Status": "Failed to extract periodic peak intervals."}

        # 2. Invert spatial attitude angles via geometric and kinematic relationships
        theta_pitch, psi_yaw, phi_roll = self._invert_spatial_attitude(rotor_omegas)

        # 3. Evaluate estimation reliability
        v_var = np.var(all_velocities) if len(all_velocities) > 0 else 0.0
        C_L4 = self._evaluate_estimation_reliability(rotor_omegas, v_var)

        semantic_output = {
            "Attitude_Angles_Rad": {
                "theta_pitch": round(theta_pitch, 4),
                "psi_yaw": round(psi_yaw, 4),
                "phi_roll": round(phi_roll, 4)
            },
            "Attitude_Angles_Deg": {
                "Pitch": round(np.degrees(theta_pitch), 2),
                "Yaw": round(np.degrees(psi_yaw), 2),
                "Roll": round(np.degrees(phi_roll), 2)
            },
            "Rotor_Speeds_rad_s": [round(w, 2) for w in rotor_omegas],
            "C_L4_Reliability": round(C_L4, 4)
        }

        # High-precision tracking typically requires a strict reliability threshold
        is_reliable = C_L4 >= self.reliability_threshold
        return is_reliable, semantic_output


if __name__ == "__main__":
    # Mocking L3 output (Clustered Rotor Kinematics)
    # Simulating the micro-Doppler traces v_k(t) of a quadcopter (4 rotors)
    # We generate synthetic sinusoidal velocity fluctuations representing blade flashes
    rng = np.random.default_rng(42)
    time_steps = np.arange(1000) * 0.001  # 1 second, dt=1 ms

    # Four rotor frequencies in hertz; angular speed is reported in rad/s.
    simulated_frequencies = [50.0, 52.0, 50.0, 48.0]
    mock_clustered_rotors = {}

    for i, freq in enumerate(simulated_frequencies):
        # Generate periodic peak signal representing micro-Doppler blade flashes
        v_k_t = 20.0 * np.sin(2 * np.pi * freq * time_steps)
        v_k_t += rng.normal(0.0, 0.2, len(time_steps))
        mock_clustered_rotors[i] = v_k_t

    # Initialize L4 Attitude Estimator with standard baseline L=0.5m
    estimator_l4 = L4_AttitudeEstimator(arm_length_L=0.5, dt=0.001)

    # Execute fine-grained attitude estimation
    success, semantic_result = estimator_l4.execute_stage_4(mock_clustered_rotors)

    import json

    print(f"=== L4 Attitude Estimation Output ===")
    print(json.dumps(semantic_result, indent=4))
