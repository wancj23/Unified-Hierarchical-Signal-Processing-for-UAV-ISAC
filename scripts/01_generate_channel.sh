#!/bin/bash
# ==============================================================================
# @Script     : 01_generate_channel.sh
# @Author     : Chengjuan Wan, et al.
# @Description: Generates highly dynamic UAV channel realizations using the 
#               NIST Quasi-Deterministic (QD) channel model.
# ==============================================================================

set -e

GREEN='\033[1;32m'
BLUE='\033[1;34m'
NC='\033[0m'

echo -e "${BLUE}[INFO] Initializing NIST QD Channel Model for UAV Scenarios...${NC}"

cd ../simulation_env/nist-qd/

# Define channel scenario parameters for dynamic micro-Doppler simulation
SCENARIO="Hexarotor_Hover_and_Move"
FC_GHZ=60.0
SAMPLING_RATE=1.76e9
OUTPUT_DIR="../../data/"

echo -e "${BLUE}[INFO] Scenario: ${SCENARIO} | Carrier: ${FC_GHZ} GHz | Fs: ${SAMPLING_RATE} Hz${NC}"
echo -e "${BLUE}[INFO] Executing ray-tracing and multi-path synthesis...${NC}"

# Execute the NIST QD channel generator
python3 generate_channel.py --scenario ${SCENARIO} --fc ${FC_GHZ} --fs ${SAMPLING_RATE} --out_dir ${OUTPUT_DIR}

echo -e "${GREEN}[SUCCESS] Channel realization complete. Data exported to ${OUTPUT_DIR}channel_matrix.mat${NC}"