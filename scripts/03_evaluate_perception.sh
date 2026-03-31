#!/bin/bash
# ==============================================================================
# @Script     : 03_evaluate_perception.sh
# @Author     : Chengjuan Wan, et al.
# @Description: Runs the Unified Hierarchical ISAC Signal Processing Pipeline.
#               Maps raw delay-Doppler data into high-level semantic perceptions.
# ==============================================================================

set -e

GREEN='\033[1;32m'
BLUE='\033[1;34m'
RED='\033[1;31m'
NC='\033[0m'

echo -e "${BLUE}[INFO] Initializing WH-DRCS Hierarchical Perception Framework...${NC}"

cd ../src/

# Target sensing level defines the depth of the hierarchical pipeline
TARGET_LEVEL="L4"
INPUT_DATA="../data/rda.json"

echo -e "${BLUE}[INFO] Pipeline Configuration: Target Level = ${TARGET_LEVEL}${NC}"
echo -e "${BLUE}[INFO] Loading raw discrete delay-Doppler matrix from ${INPUT_DATA}...${NC}"

# Verify environment dependencies
if ! python3 -c "import numpy, scipy, sklearn" &> /dev/null; then
    echo -e "${RED}[ERROR] Required Python packages are missing. Please run 'pip install -r ../requirements.txt'${NC}"
    exit 1
fi

echo -e "${BLUE}[INFO] Executing Mechanisms 1 & 2 (Nyquist-Folding & Adaptive DBSCAN)...${NC}"

# Execute the core semantic perception mapping algorithm
python3 unified_hierarchical_processor.py --level ${TARGET_LEVEL} --data ${INPUT_DATA}

echo -e "\n${GREEN}[SUCCESS] Hierarchical Semantic Extraction Complete.${NC}"