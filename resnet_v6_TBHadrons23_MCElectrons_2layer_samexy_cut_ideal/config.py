from dl_recon_core_sparse.resnets_cls_models import *
from cuts import *
import numpy as np

LOGARITHMIC_SCALING=False
LOGARITHMIC_SCALING_INNER=1
LOGARITHMIC_SCALING_OUTER=1
# ==========================================
# 1. RUN CONFIGURATION & HYPERPARAMETERS
# ==========================================
TB_RECALIBRATION_S2Y=False
TB_USE_SAME_XY_S2=True
IS_MC_TUNING=False

out_name = "NO NEED FOR OUT NAME"
RUN_TRAINING = False
OUTPUT_NAME_EN_RECON = "resnet_v6_neutrinos"

PATIENCE = 20
LEARNING_RATE = 1e-4
LEARNING_RATE2 = 1e-5

BATCH_SIZE = 256
BATCH_SIZE_VAL = 256
BATCH_SIZE2 = 512
BATCH_SIZE_TEST = 512

TOTAL_TEST_SIZE=20000

PERC_TRAIN = 42000  # None, particle number, or fraction (e.g., 0.1)
PERC_VAL = None

TARGET_DATA_DIR = None
EN_MIN = 0
EN_MAX= 2000
DA_1_DATALOADER = False # FOR D.A.


# ==========================================
# 2. TASK & LABEL DEFINITIONS
# ==========================================
IS_CLS = True
IS_ENERGY_RECON = False
IS_BINARY = True


if IS_BINARY:
    FINAL_CLASS_DIM = 1
else:
    FINAL_CLASS_DIM = 6 
    print("write manuelly FINAL_CLASS_DIM", FINAL_CLASS_DIM)

IS_SINGLE_NETWORK = True
if IS_SINGLE_NETWORK:
    FEATURE_SIZE = FINAL_CLASS_DIM
else:
    FEATURE_SIZE = 64

INNER_DIM=16 #used for 2nd classifier

GET_PARTICLES_WITH_INDEX = [0, 1, 2, 3]  
CLASS_NAMES_CONF_MATRIX = [
    r"$\nu_e$",
    r"$\nu_\mu$",
    r"$\nu_\tau$",
    "NC"
]

GET_PARTICLES_WITH_INDEX_LARGER = [0, 1, 2, 3, 4, 5]
CLASS_NAMES_CONF_MATRIX_LARGER = [
    r"$\nu_e$",
    r"$\nu_\mu$",
    r"$\nu_\tau$",
    "NC",
    r"$\mu$",
    "PG Hadrons",
]



# ==========================================
# 3. DETECTOR SYSTEM ROUTING (NEW DATALOADER LOGIC)
# ==========================================
SHOWER_WIDTH = 256               # Crops channel width to 2*SHOWER_WIDTH (512)
USE_HIGHEST_N_LAYER = 2         # How many Z-planes to keep
KEYS_FOR_DATA_LOADER = ["scifi_signals", "en3d", "y"]
#KEYS_FOR_DATA_LOADER = ["scifi_signals", "us_signals", "ds", "en3d", "y"]

# DO NOT TOUCH BELOW, JUST CHANGE MODEL NAME.
USE_US = False 
USE_DS = False 
if "us_signals" in KEYS_FOR_DATA_LOADER:
    USE_US = True 

if "ds" in KEYS_FOR_DATA_LOADER:
    USE_DS = True 

if USE_US and USE_DS:
    USE_SCIFI_US_DS = True
    USE_SCIFI_US = False
elif USE_US:
    USE_SCIFI_US = True
    USE_SCIFI_US_DS = False
else:
    USE_ONLY_SCIFI = True 
    USE_SCIFI_US = False
    USE_SCIFI_US_DS = False

if USE_ONLY_SCIFI:
    MODEL = ResNets_scifi_R256Optimized_2layer(2, FEATURE_SIZE)
    CLASSIFIER_MLP = Classifier_mlp(FEATURE_SIZE, INNER_DIM, FINAL_CLASS_DIM) #NOT USED
    TRAINED_MODEL_NAME = "none"
    TRAINED_CLS_NAME = "none"

elif USE_SCIFI_US:
    KEYS_FOR_DATA_LOADER = ["scifi_signals", "us_signals", "en3d", "y"]
    CLASSIFIER_MLP = Classifier_mlp(FEATURE_SIZE, INNER_DIM, FINAL_CLASS_DIM) #NOT USED
    # Add your US model initialization here if needed

elif USE_SCIFI_US_DS:
    MODEL = ResNets(2, FEATURE_SIZE)
    CLASSIFIER_MLP = Classifier_mlp(FEATURE_SIZE, INNER_DIM, FINAL_CLASS_DIM) #NOT USED
    TRAINED_MODEL_NAME = "none"
    TRAINED_CLS_NAME = "none"


# ==========================================
# 5. LOSS FUNCTION & EVALUATION
# ==========================================
pos_weight = False
focal_loss = False
gamma = 2

aploss = False
delta = 1

VAL_ACC_BEST_EPOCH=True
VAL_PREJ=False

import numpy as np
bins = np.array([0,5,10,20,30,50,70,100,120,150,200,300,500,700,1000,1500])
bin_centers = 0.5 * (bins[:-1] + bins[1:]) 
REQUIRED_EFF=0.9