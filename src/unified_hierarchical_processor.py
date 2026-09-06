r"""
@File       : unified_hierarchical_processor.py
@Author     : Chengjuan Wan, et al.
@Institution: University of Chinese Academy of Sciences
@Description:
    Unified Hierarchical ISAC Signal Processing Pipeline.
    This master module tightly integrates the four defined hierarchical UAV
    sensing levels (L1 to L4) based on the WH-DRCS adaptive frame structure.
    It ingests raw discrete delay-Doppler matrices from the NIST ISAC-PLM
    platform and sequentially executes fundamental detection, target classification,
    configuration recognition, and fine-grained 3D attitude estimation.

    Reference: Algorithm 1 in "Design and Optimization of a Doppler-Resilient
    ISAC Framework for UAV Hierarchical Sensing".
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Import the four hierarchical stage modules
# (Ensure these are saved in the same directory or properly referenced in PYTHONPATH)
from l1_fundamental_detection import L1_FundamentalDetector, ISACPLM_DataParser
from l2_target_classification import L2_TargetClassifier
from l3_configuration_recognition import L3_ConfigurationRecognizer
from l4_attitude_estimation import L4_AttitudeEstimator


class UnifiedHierarchicalISACProcessor:
    r"""
    The master pipeline orchestrator for the WH-DRCS ISAC framework.
    Maps raw baseband data matrices into precise semantic outputs across
    diverse sensing levels.
    """

    def __init__(self, target_level: str = 'L4'):
        r"""
        Initializes the unified perception architecture.
        :param target_level: The maximum requested sensing level \in {'L1', 'L2', 'L3', 'L4'}.
                             Determines the depth of the processing pipeline, enabling
                             the Pareto-optimal trade-off between communication and sensing.
        """
        self.target_level = target_level.upper()
        self.valid_levels = {'L1': 1, 'L2': 2, 'L3': 3, 'L4': 4}

        if self.target_level not in self.valid_levels:
            raise ValueError("Target level must be one of 'L1', 'L2', 'L3', 'L4'.")

        # Paper-level frame profiles. These parameters are determined by the selected
        # sensing level and are not retuned for a test waveform, SNR, or scenario.
        self.frame_profiles = {
            'L1': {'M': 8, 'T_PRI': 50.00e-6, 'v_max': 25.0},
            'L2': {'M': 16, 'T_PRI': 25.00e-6, 'v_max': 50.0},
            'L3': {'M': 64, 'T_PRI': 10.00e-6, 'v_max': 125.0},
            'L4': {'M': 128, 'T_PRI': 15.00e-6, 'v_max': 83.3},
        }
        selected_profile = self.frame_profiles[self.target_level]

        # Frozen paper configuration. Fixed hyperparameters are selected once on
        # the validation split and then used unchanged for test evaluation.
        self.l1_detector = L1_FundamentalDetector(
            pri_duration=selected_profile['T_PRI'],
            gamma_detect=0.85,
            eta_a=15.0
        )
        self.l2_classifier = L2_TargetClassifier(
            v_max=selected_profile['v_max'],
            delta_v=2.0,
            gamma_uav=0.65
        )
        self.l3_recognizer = L3_ConfigurationRecognizer(
            tau_kurtosis=2.0,
            epsilon_scale=0.05,
            min_samples=3,
            power_percentile=90.0,
            confidence_threshold=0.60
        )
        self.l4_estimator = L4_AttitudeEstimator(
            arm_length_L=0.5,
            dt=selected_profile['T_PRI'],
            min_peak_interval=0.005,
            peak_prominence=0.5,
            reliability_threshold=0.50
        )

    def execute_pipeline(self, raw_data_filepath: str) -> Dict[str, Any]:
        r"""
        Executes the complete multi-tiered pipeline mapping raw echoes into semantic perceptions.
        :param raw_data_filepath: Path to the NIST ISAC-PLM 'rda.json' output.
        :return: A comprehensive semantic perception dictionary.
        """
        final_semantic_output = {"Target_Level_Requested": self.target_level}

        # ==========================================
        # Data Ingestion
        # ==========================================
        try:
            parser = ISACPLM_DataParser(filepath=raw_data_filepath)
            kinematics_data = parser.extract_kinematic_sequence()
        except Exception as e:
            return {"Error": f"Data ingestion failed: {str(e)}"}

        # ==========================================
        # Stage 1: Fundamental Detection (L1)
        # ==========================================
        is_target, l1_result = self.l1_detector.execute_stage_1(kinematics_data)
        final_semantic_output['L1_Detection'] = l1_result

        if not is_target or self.valid_levels[self.target_level] < 2:
            return final_semantic_output

        # ==========================================
        # Stage 2: Target Classification (L2)
        # ==========================================
        is_uav, l2_result = self.l2_classifier.execute_stage_2(kinematics_data)
        final_semantic_output['L2_Classification'] = l2_result

        if not is_uav or self.valid_levels[self.target_level] < 3:
            return final_semantic_output

        # ==========================================
        # Stage 3: Configuration Recognition (L3)
        # ==========================================
        is_recognized, l3_result = self.l3_recognizer.execute_stage_3(kinematics_data)
        final_semantic_output['L3_Recognition'] = l3_result

        if not is_recognized or self.valid_levels[self.target_level] < 4:
            return final_semantic_output

        # ==========================================
        # Stage 4: Attitude Estimation (L4)
        # ==========================================
        # To strictly link L3 and L4, we extract the clustered rotor traces from L3's internal processing
        # In this unified pipeline, we reconstruct the isolated kinematics for Mechanism 2 output
        v_seq = kinematics_data.get('velocity', np.array([]))
        p_seq = kinematics_data.get('power', np.array([]))
        V_peak = self.l3_recognizer._extract_local_velocity_extremum(v_seq, p_seq)
        _, labels, _ = self.l3_recognizer._mechanism_2_adaptive_dbscan(V_peak)

        # Map labels to discrete kinematic sequences v_k(t)
        clustered_rotors = {}
        for k in set(labels):
            if k != -1:  # Ignore noise
                clustered_rotors[k] = V_peak[labels == k]

        is_reliable, l4_result = self.l4_estimator.execute_stage_4(clustered_rotors)
        final_semantic_output['L4_Attitude'] = l4_result

        return final_semantic_output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the WH-DRCS hierarchical sensing pipeline."
    )
    parser.add_argument(
        "--input",
        default="data/rda.json",
        help="Path to the line-delimited ISAC-PLM RDA JSON file."
    )
    parser.add_argument(
        "--level",
        default="L4",
        choices=["L1", "L2", "L3", "L4"],
        help="Maximum hierarchical sensing level to execute."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for the resulting semantic-output JSON file."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    input_path = Path(args.input)

    if not input_path.is_file():
        raise SystemExit(
            f"Input file '{input_path}' was not found. "
            "Generate it with ISAC-PLM or provide --input PATH."
        )

    processor = UnifiedHierarchicalISACProcessor(target_level=args.level)
    comprehensive_output = processor.execute_pipeline(str(input_path))
    rendered_output = json.dumps(comprehensive_output, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered_output + "\n", encoding="utf-8")
    else:
        print(rendered_output)
