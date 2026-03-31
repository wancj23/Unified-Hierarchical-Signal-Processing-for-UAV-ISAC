#!/bin/bash
# ==============================================================================
# @Script     : 02_run_isac_plm.sh
# @Author     : Chengjuan Wan, et al.
# @Description: Executes the modified NIST ISAC-PLM platform injected with 
#               the proposed WH-DRCS waveform to generate baseband radar echoes.
# ==============================================================================

set -e

GREEN='\033[1;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m'

echo -e "${BLUE}[INFO] Booting ISAC-PLM (IEEE 802.11ay/bf EDMG mode)...${NC}"

cd ../simulation_env/isac-plm/

# Physical layer configurations for the proposed scalable ZAZ
WAVEFORM="WH-DRCS"
ZAZ_DIMENSION_M=8  # M=8 corresponds to the high-throughput communication mode
CHANNEL_FILE="../../data/channel_matrix.mat"
OUTPUT_JSON="../../data/rda.json"

echo -e "${YELLOW}[PARAM] Physical Layer Waveform: ${WAVEFORM}${NC}"
echo -e "${YELLOW}[PARAM] Sequence Set Size (M): ${ZAZ_DIMENSION_M}${NC}"
echo -e "${BLUE}[INFO] Applying deterministic orthogonal masking and pi/2-BPSK modulation...${NC}"

# Execute the baseband processing and export the discrete delay-Doppler matrix
python3 run_isac_simulation.py \
    --waveform ${WAVEFORM} \
    --m_size ${ZAZ_DIMENSION_M} \
    --channel_input ${CHANNEL_FILE} \
    --export_rda ${OUTPUT_JSON}

echo -e "${GREEN}[SUCCESS] Baseband processing complete. Delay-Doppler matrix saved to ${OUTPUT_JSON}${NC}"