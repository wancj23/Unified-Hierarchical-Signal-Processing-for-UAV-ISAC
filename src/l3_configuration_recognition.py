"""
@File       : l3_configuration_recognition.py
@Author     : Chengjuan Wan, et al.
@Institution: University of Chinese Academy of Sciences
@Description:
    Stage 3: UAV Configuration Recognition (L3) module for the WH-DRCS ISAC framework.
    This module implements Mechanism 2 (Adaptive Density-Based Kinematic Clustering)
    to accurately isolate discrete rotor signatures from heavily mixed micro-Doppler
    signals. It maps the derived cluster count K directly to the actual physical
    rotor topology (e.g., Quadrotor, Hexarotor) and evaluates confidence using
    silhouette coefficients and theoretical kurtosis mapping.
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from scipy.stats import kurtosis
from typing import Dict, Tuple, List, Any


class L3_ConfigurationRecognizer:
    """
    Implements Stage 3 of the Hierarchical Signal Processing Algorithm.
    Executes adaptive DBSCAN to extract valid cluster count K and maps it
    to physical UAV topology (Configuration Type C).
    """

    def __init__(self, tau_kurtosis: float = 2.0):
        """
        :param tau_kurtosis: Relaxation parameter (\tau) for kurtosis exponential mapping.
        """
        self.tau_kurtosis = tau_kurtosis

        # Pre-calibrated fusion weights \mathbf{w}_3 for recognition confidence
        # Derived from ISAC-PLM topological dataset via constrained optimization
        self._w3 = np.array([0.65, 0.35])

        # Topological Mapping Dictionary \mathcal{M}_{topo}(K)
        # Symmetrically maps the number of detected micro-Doppler clusters (2*K rotors)
        # to the specific UAV airframe configuration.
        self._topology_map = {
            2: "Helicopter (Single Main + Tail Rotor)",
            4: "Quadrotor (Standard Drone)",
            6: "Hexarotor (Heavy Payload)",
            8: "Octocopter (Industrial Grade)"
        }

        # Theoretical reference kurtosis (\kappa_{ref}) for different topologies
        self._kurtosis_ref = {2: 1.5, 4: 2.2, 6: 2.8, 8: 3.5}

    def _extract_local_velocity_extremum(self, v_seq: np.ndarray, p_seq: np.ndarray,
                                         percentile: float = 90.0) -> np.ndarray:
        """
        Extracts the local velocity extremum set \mathcal{V}_{peak}.
        Implements sliding-window maximum detection by retaining only the velocities
        associated with the top (100 - percentile)% reflection power.
        """
        if len(p_seq) == 0:
            return np.array([])

        # Dynamically calculate the power threshold to filter ambient noise
        power_threshold = np.percentile(p_seq, percentile)
        peak_mask = p_seq >= power_threshold

        V_peak = v_seq[peak_mask]
        return V_peak

    def _mechanism_2_adaptive_dbscan(self, V_peak: np.ndarray) -> Tuple[int, np.ndarray, float]:
        """
        Mechanism 2: Adaptive Density-Based Kinematic Clustering.
        Dynamically calibrates the neighborhood radius \epsilon to inherently
        prevent over-clustering in dense multi-rotor scenarios.
        """
        if len(V_peak) < 5:
            return 0, np.array([]), 0.0

        # 1. Compute sample standard deviation \sigma_{\mathcal{V}}
        sigma_v = np.std(V_peak)

        # 2. Dynamically scale neighborhood radius: \epsilon \leftarrow \sigma_{\mathcal{V}} \cdot \sqrt[3]{\ln |\mathcal{V}_{peak}|}
        # Addition of 1e-5 prevents log(0) or log(1) edge cases
        epsilon = sigma_v * np.cbrt(np.log(len(V_peak) + 1e-5))

        # 3. Execute DBSCAN on metric space (\mathcal{V}_{peak}, d)
        # Reshape for sklearn compatibility
        V_peak_2d = V_peak.reshape(-1, 1)
        clustering = DBSCAN(eps=epsilon, min_samples=3).fit(V_peak_2d)

        labels = clustering.labels_
        # Isolate connected components (excluding noise labeled as -1)
        unique_clusters = set(labels) - {-1}
        K_clusters = len(unique_clusters)

        return K_clusters, labels, epsilon

    def _evaluate_cluster_validity(self, V_peak: np.ndarray, labels: np.ndarray, K: int) -> Tuple[float, float]:
        """
        Evaluates cluster validity via average silhouette coefficient (s) and
        theoretical kurtosis mapping (\kappa).
        """
        if K < 2 or len(set(labels)) <= 1:
            return 0.0, 0.0  # Cannot compute silhouette for single cluster or pure noise

        # 1. Silhouette Coefficient (Spatial cohesion and separation)
        try:
            # Filter out noise points for strict silhouette evaluation
            valid_mask = labels != -1
            s_score = silhouette_score(V_peak[valid_mask].reshape(-1, 1), labels[valid_mask])
            # Normalize from [-1, 1] to [0, 1]
            s_score = (s_score + 1.0) / 2.0
        except ValueError:
            s_score = 0.0

        # 2. Kurtosis Mapping
        empirical_kurtosis = kurtosis(V_peak)
        kappa_ref = self._kurtosis_ref.get(K, 3.0)  # Default fallback

        # \exp\left(-\frac{|\kappa - \kappa_{ref}|}{\tau}\right)
        kurtosis_score = np.exp(- np.abs(empirical_kurtosis - kappa_ref) / self.tau_kurtosis)

        return float(s_score), float(kurtosis_score)

    def execute_stage_3(self, kinematics: Dict[str, np.ndarray]) -> Tuple[bool, Dict[str, Any]]:
        """
        Executes Stage 3: UAV Configuration Recognition.
        :param kinematics: Dictionary containing 'velocity' and 'power' arrays.
        :return: (Recognition Status Boolean, Semantic Configuration Output)
        """
        v_seq = kinematics.get('velocity', np.array([]))
        p_seq = kinematics.get('power', np.array([]))

        # 1. Extract local velocity extremum set \mathcal{V}_{peak}
        V_peak = self._extract_local_velocity_extremum(v_seq, p_seq, percentile=90.0)

        # 2. Adaptive DBSCAN (Mechanism 2)
        K_clusters, labels, epsilon = self._mechanism_2_adaptive_dbscan(V_peak)

        if K_clusters == 0:
            return False, {"Configuration_Type": "Unknown", "Status": "Clustering Failed"}

        # 3. Extract valid cluster count K and map to physical topology
        # Assuming symmetric Doppler signatures (positive/negative pairs per rotor)
        estimated_rotors = K_clusters

        # Find closest matching topology (rounding to nearest even number supported)
        closest_K = min(self._topology_map.keys(), key=lambda k: abs(k - estimated_rotors))
        config_type = self._topology_map[closest_K]

        # 4. Evaluate recognition confidence \mathcal{C}_{L_3}
        s_score, kurtosis_score = self._evaluate_cluster_validity(V_peak, labels, closest_K)

        # \mathcal{C}_{L_3} \leftarrow \mathbf{w}_3^T [s, \exp(...)]^T
        C_L3 = float(np.dot(self._w3, np.array([s_score, kurtosis_score])))

        semantic_output = {
            "Configuration_Type": config_type,
            "Detected_Clusters_K": K_clusters,
            "C_L3_Confidence": round(C_L3, 4),
            "Mechanism_2_Metrics": {
                "Adaptive_Epsilon": round(epsilon, 4),
                "Silhouette_Coefficient_s": round(s_score, 4),
                "Theoretical_Kurtosis_Match": round(kurtosis_score, 4)
            }
        }

        # Assuming a confidence threshold of 0.6 for successful recognition
        is_recognized = C_L3 >= 0.6
        return is_recognized, semantic_output


if __name__ == "__main__":
    # Mocking ISAC-PLM JSON extraction (simulating 'rda_RD5.py' behavior)
    # Simulating a Hexarotor (6 rotors, thus 6 distinct symmetric micro-Doppler peaks)
    np.random.seed(42)
    # Background noise
    noise_v = np.random.uniform(-100, 100, 1000)
    noise_p = np.random.uniform(0.1, 2.0, 1000)

    # 6 distinct velocity peaks representing 6 rotors (positive & negative folding combined)
    centers = [-65, -40, -15, 15, 40, 65]
    peaks_v = np.concatenate([np.random.normal(c, 2.5, 50) for c in centers])
    peaks_p = np.random.uniform(10.0, 50.0, 300)  # High reflection power

    mock_kinematics = {
        'velocity': np.concatenate([noise_v, peaks_v]),
        'power': np.concatenate([noise_p, peaks_p])
    }

    recognizer_l3 = L3_ConfigurationRecognizer()
    success, semantic_result = recognizer_l3.execute_stage_3(mock_kinematics)

    import json

    print(f"=== L3 Configuration Recognition Output ===")
    print(json.dumps(semantic_result, indent=4))