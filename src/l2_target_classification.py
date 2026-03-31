"""
@File       : l2_target_classification.py
@Author     : Chengjuan Wan, et al.
@Institution: University of Chinese Academy of Sciences
@Description:
    Stage 2: Target Classification (L2) module for the WH-DRCS ISAC framework.
    This module discriminates UAVs from generic moving scatterers (e.g., birds,
    kites, or airborne debris) by evaluating the Nyquist-Folded Micro-Doppler
    Extraction (Mechanism 1). It computes trajectory autocorrelation entropy,
    boundary truncation ratio, and employs a linear discriminant function to
    map the kinematic features into a semantic target category.
"""

import numpy as np
from scipy.linalg import norm
from typing import Dict, Tuple, Optional, Any


class L2_TargetClassifier:
    """
    Implements Stage 2 of the Hierarchical Signal Processing Algorithm.
    Evaluates the discriminant function \mathcal{C}_{L_2} using Nyquist
    folding boundary conditions and trajectory entropy.
    """

    def __init__(self, v_max: float, delta_v: float = 2.0, gamma_uav: float = 0.65):
        """
        :param v_max: Frame-specific maximum unambiguous velocity (v_{max}) in m/s.
                      Determined dynamically by the WH-DRCS dimension M.
        :param delta_v: Velocity boundary interval width (\Delta v) in m/s.
        :param gamma_uav: Discriminant threshold (\gamma_{UAV}) for UAV classification.
        """
        self.v_max = v_max
        self.delta_v = delta_v
        self.gamma_uav = gamma_uav

        # Optimal separating hyperplane parameters (\mathbf{w}_2 and b_2)
        # Pre-trained via Support Vector Machine (SVM) on empirical ISAC-PLM datasets
        # (Differentiating Drones vs. Birds/Kites/Shells)
        self._w2 = np.array([0.45, 0.35, 0.20])
        self._b2 = -0.15

    def _mechanism_1_nyquist_folding_ratio(self, velocity_seq: np.ndarray) -> float:
        """
        Mechanism 1: Nyquist-Folded Micro-Doppler Extraction.
        Calculates the discrete boundary accumulation ratio \rho_{micro}.
        \rho_{micro} = \frac{1}{T} \sum_{t=0}^{T} \mathbb{I}_{\mathcal{B}}(|v_{obs}(t)|)
        where \mathcal{B} = [v_{max} - \Delta v, v_{max}].
        """
        if len(velocity_seq) == 0:
            return 0.0

        abs_v = np.abs(velocity_seq)
        # Define the boundary interval \mathcal{B}
        lower_bound = self.v_max - self.delta_v
        upper_bound = self.v_max

        # Indicator function \mathbb{I}_{\mathcal{B}}(\cdot)
        boundary_condition = (abs_v >= lower_bound) & (abs_v <= upper_bound)

        rho_micro = np.sum(boundary_condition) / len(velocity_seq)
        return float(rho_micro)

    def _compute_trajectory_autocorrelation_entropy(self, seq: np.ndarray) -> float:
        """
        Computes the Trajectory Autocorrelation Entropy (\mathcal{H}_A).
        Utilizes the Wiener-Khinchin theorem logic to evaluate the periodic
        micro-motions (e.g., rotor blades vs. random bird wing flapping).
        """
        if len(seq) < 2:
            return 1.0

        # Zero-mean sequence for unbiased autocorrelation
        seq_zm = seq - np.mean(seq)

        # Compute discrete autocorrelation R_v(\tau)
        autocorr = np.correlate(seq_zm, seq_zm, mode='full')
        # Retain only positive lags
        autocorr = autocorr[len(autocorr) // 2:]

        # Normalize to power probability distribution
        power_dist = np.abs(autocorr) / (np.sum(np.abs(autocorr)) + 1e-12)
        power_dist = power_dist[power_dist > 0]  # Avoid log(0)

        shannon_entropy = -np.sum(power_dist * np.log(power_dist))
        max_entropy = np.log(len(power_dist))

        # Map to [0, 1] where highly periodic signals (low entropy) score higher
        return 1.0 - (shannon_entropy / max_entropy) if max_entropy > 0 else 1.0

    def _compute_kinematic_smoothness(self, sequence: np.ndarray) -> float:
        """
        Computes the differential smoothness S_r'.
        Uses first-order derivative variance to evaluate trajectory stability.
        """
        if len(sequence) < 2:
            return 1.0

        # Gradient variance indicates kinematic jitter
        grad_var = np.var(np.gradient(sequence))

        # Exponential mapping to map jitter into a [0, 1] confidence score
        sigma_scale = 1.5
        return np.exp(-grad_var / (2 * sigma_scale ** 2))

    def execute_stage_2(self, kinematics: Dict[str, np.ndarray]) -> Tuple[bool, Dict[str, Any]]:
        """
        Executes Stage 2: Target Classification.
        :param kinematics: Extracted \mathcal{K} = {(v_t, r_t)}_{t=1}^T matrix.
        :return: (Is Generic UAV, Semantic Classification Output)
        """
        v_seq = kinematics.get('velocity', np.array([]))
        r_seq = kinematics.get('range', np.array([]))

        if len(v_seq) == 0:
            return False, {"Category": "Unknown", "Status": "Insufficient Data"}

        # 1. Extract Nyquist boundary truncation ratio (Mechanism 1)
        rho_micro = self._mechanism_1_nyquist_folding_ratio(v_seq)

        # 2. Compute trajectory autocorrelation entropy & differential smoothness
        H_A = self._compute_trajectory_autocorrelation_entropy(v_seq)
        S_r_prime = self._compute_kinematic_smoothness(r_seq)

        # 3. Evaluate discriminant function \mathcal{C}_{L_2}
        feature_vector = np.array([H_A, rho_micro, S_r_prime])
        C_L2 = float(np.dot(self._w2, feature_vector) + self._b2)

        # 4. Apply decision logic (Algorithm 1, Lines 12-14)
        is_uav = C_L2 >= self.gamma_uav
        target_category = "Generic UAV" if is_uav else "Non-UAV (e.g., Bird/Clutter)"

        semantic_output = {
            "Target_Category": target_category,
            "C_L2_Discriminant_Score": round(C_L2, 4),
            "Extracted_Features": {
                "rho_micro_Nyquist": round(rho_micro, 4),
                "H_A_Autocorrelation_Entropy": round(H_A, 4),
                "S_r_prime_Smoothness": round(S_r_prime, 4)
            }
        }

        return is_uav, semantic_output


if __name__ == "__main__":
    # Example execution pipeline demonstrating L2 Classification
    # Assuming the target has passed L1 Fundamental Detection

    # Mock sequence extracted from ISAC-PLM JSON data
    # Simulating a UAV experiencing Nyquist folding due to fast rotors
    mock_kinematics = {
        'velocity': np.random.normal(48.5, 5.0, 100),  # Speeds near v_max
        'range': np.linspace(100, 105, 100) + np.random.normal(0, 0.1, 100)
    }

    # Initialize L2 Classifier with a v_max corresponding to M=16 (e.g., 50.0 m/s)
    classifier_l2 = L2_TargetClassifier(v_max=50.0, delta_v=2.5, gamma_uav=0.6)

    # Execute unified classification
    is_uav, semantic_result = classifier_l2.execute_stage_2(mock_kinematics)

    import json

    print(f"=== L2 Target Classification Output ===")
    print(json.dumps(semantic_result, indent=4))