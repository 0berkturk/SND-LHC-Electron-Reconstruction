from dl_recon_core_sparse.resnets_energy_models import *
from cuts import *
import numpy as np

out_name = "NO NEED FOR OUT NAME"
OUTPUT_NAME_EN_RECON = "resnet"

TB_RECALIBRATION_S2Y=True
TB_USE_SAME_XY_S2=False
IS_MC_TUNING=False

RUN_TRAINING=False 

TARGET_DATA_DIR = None
EN_MIN=0
EN_MAX=20000
DA_1_DATALOADER = False

PERC_TRAIN = None  ## CAN BE PUT Number of particles as well
PERC_VAL = None

TOTAL_TEST_SIZE=5000

IS_CLS = False
IS_ENERGY_RECON = True
IS_BINARY = True # always true for energy recon.

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


PATIENCE = 20
LEARNING_RATE = 1e-4
LEARNING_RATE2 = 1e-5
N_BLOCKS = [1, 1, 4, 4]  # L
N_CHANNELS = [96, 192, 192, 512]  # D
BATCH_SIZE=256*2
BATCH_SIZE_VAL = 256*2
BATCH_SIZE2 = 256
BATCH_SIZE_TEST=512
# ==========================================
# 3. DETECTOR SYSTEM ROUTING (NEW DATALOADER LOGIC)
# ==========================================
SHOWER_WIDTH=256               # Crops channel width to 2*SHOWER_WIDTH (512)
USE_HIGHEST_N_LAYER=5         # How many Z-planes to keep
KEYS_FOR_DATA_LOADER = ["scifi_signals", "en3d", "y"]
#KEYS_FOR_DATA_LOADER = ["scifi_signals", "us_signals", "ds", "en3d", "y"]

# DO NOT TOUCH BELOW, JUST CHANGE MODEL NAME. No need to upload other keys to apply cuts. just define above the keys to use
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
    MODEL = ResNets_scifi_R256Optimized_onlydl(2, FEATURE_SIZE)
    CLASSIFIER_MLP = Classifier_mlp(1, 1)  #NOT USED
    TRAINED_MODEL_NAME = "none"
    TRAINED_CLS_NAME = "none"

elif USE_SCIFI_US:
    CLASSIFIER_MLP = Classifier_mlp(1, 1) #NOT USED
    # Add your US model initialization here if needed

elif USE_SCIFI_US_DS:
    MODEL = ResNets(2, FEATURE_SIZE)
    CLASSIFIER_MLP =Classifier_mlp(1, 1)  #NOT USED
    TRAINED_MODEL_NAME = "none"
    TRAINED_CLS_NAME = "none"


## LOSS FUNCTIONS
LOSS_FUNC_TYPE = "L1"  # "L1", "L1", "L1"

train_loss_response = True  # if False, train energy directly
val_loss_response = True  # if True, train energy after first training
train_loss_rel_bias=False
val_loss_rel_bias=False


NEUTRINO_FLAVOR_INDEX=[0,3]
VAL_PREJ=False

## BINS
width_smaller=5
en_max_smaller=350
en_min_smaller=0
bins2 = np.array([])#np.arange(en_min_smaller, en_max_smaller+width_smaller, width_smaller)

e_recon_etrue_small=np.arange(en_min_smaller, en_max_smaller+width_smaller, width_smaller) # used for 1 histogram only

bins3=np.array([5,10,15,20,25,30,35,40,45,50])
bins4 = np.array([])#np.arange(10, 2000+50, 50)
