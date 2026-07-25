#!/bin/bash

# ==============================================================================
# AUTOMATED PIPELINE: ROOT to PT -> Deep Learning Inference -> ROOT/TXT Output
# ==============================================================================
# This script is designed to process an event list, convert the raw ROOT data 
# to PyTorch tensors (.pt), and run multiple Deep Learning models on them.
# 
# PREREQUISITES FOR USERS:
# 1. Update the 'USER CONFIGURATION' section below with your own paths.
# 2. Ensure you have the 'SNDSW' environment available for root2pt.py.
# 3. Ensure you have PyTorch installed, as well as numpy, pandas and other essential packages.
# ==============================================================================

# Check if both input text file and output directory are provided
if [ "$#" -ne 2 ]; then
    echo "============================================"
    echo "[ERROR] Usage: $0 <events_list.txt> <output_directory>"
    echo ""
    echo "[INFO] Expected format for events_list.txt:"
    echo "       <run_number> <event_number>"
    echo "       (one pair per line, separated by a space)"
    echo ""
    echo "Example content of events_list.txt:"
    echo "10919 123456"
    echo "10919 123457"
    echo "============================================"
    exit 1
fi

INPUT_TXT=$(realpath $1)

# Create output directory if it doesn't exist and get its absolute path
mkdir -p "$2"
PT_OUT_DIR=$(realpath "$2")
BASE_DIR=$(pwd)

MODEL_1_DIR="resnet_v6_TBHadrons23_MCElectrons_2layer_samexy_cut_ideal"
MODEL_2_DIR="MC_electrons_ResNets_SciFi_2layers_R256_100gev_-0.5_s2_q13_ft0_layer5"
MODEL_3_DIR="log_input_more_data_MC_electrons_ResNets_SciFi_R128_400gev_0_s2_q13_ft0_layer2"
# ==============================================================================
# USER CONFIGURATION (Modify these paths to match your personal setup)
# ==============================================================================
# The Conda environment name where PyTorch is installed
CONDA_ENV_NAME="myenv"

# If your conda command is not recognized inside a bash script, you might need 
# to source your conda base profile here. Uncomment and change the path if needed:
# source /afs/cern.ch/work/b/beturk/private/miniconda3/etc/profile.d/conda.sh


echo "============================================"
echo "[INFO] 1. Setting up SNDSW Environment for ROOT parsing..."
echo "============================================"
# Deactivate any currently active conda environments to prevent package conflicts 
# with the SND software environment. (Called twice just in case of nested envs)
conda deactivate
conda deactivate

# Source the SND Software setup script
source /cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh

# Instead of 'alienv enter' (which opens an interactive sub-shell and halts the script),
# we use 'eval $(alienv load ...)' to safely inject the variables into the current script.
eval $(alienv load sndsw/latest)

echo "[INFO] Running root2pt.py..."
# Assuming root2pt.py is located in the directory where this script is called
#python root2pt.py --txt "$INPUT_TXT" --outdir "$PT_OUT_DIR"


echo "============================================"
echo "[INFO] 2. Switching to Conda Environment for Deep Learning..."
echo "============================================"

# Unload alienv to cleanly return to standard environment variables
eval $(alienv unload sndsw/latest)

# This prevents the Python 3.9 vs 3.13 clash (SRE module mismatch)
unset PYTHONPATH
unset PYTHONHOME

# Activate the user's deep learning conda environment
conda activate "$CONDA_ENV_NAME"

echo "============================================"
echo "[INFO] 3. Running Model 1 (Classification)..."
echo "============================================"
cd "$MODEL_1_DIR"
#python main_analysis.py "$PT_OUT_DIR"
cd "$BASE_DIR"

echo "============================================"
echo "[INFO] 4. Running Model 2 (Energy Recon 1)..."
echo "============================================"
cd "$MODEL_2_DIR"
#python main_analysis.py "$PT_OUT_DIR"
cd "$BASE_DIR"

echo "============================================"
echo "[INFO] 5. Running Model 3 (Energy Recon 2)..."
echo "============================================"
cd "$MODEL_3_DIR"
#python main_analysis.py "$PT_OUT_DIR"
cd "$BASE_DIR"

echo "============================================"
echo "[INFO] 6. Consolidating outputs into TXT and ROOT formats..."
echo "============================================"
# Return to the base directory where save_results.py is located
cd "$BASE_DIR"

# The models automatically create a "_DL_processed" folder next to the provided PT directory
# We pass this processed directory to save_results.py to combine everything.
python save_results.py --dir "${PT_OUT_DIR}" \
                       --out_txt "$PT_OUT_DIR/final_results.txt" \
                       --out_root "$PT_OUT_DIR/final_results.root"

echo "============================================"
echo "[SUCCESS] Pipeline execution complete!"
echo "[INFO] Processed PyTorch tensors (.pt): ${PT_OUT_DIR}_DL_processed"
echo "[INFO] Final merged outputs (.txt and .root): $PT_OUT_DIR"
echo "============================================"