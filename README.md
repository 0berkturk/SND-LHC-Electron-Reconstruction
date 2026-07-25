# SND@LHC Deep Learning Reconstruction Pipeline

This repository contains an automated pipeline designed to extract raw event data from SND@LHC ROOT files, convert them into PyTorch tensors, run multiple Deep Learning models (Classification and Energy Reconstruction) on the data, and finally merge the predictions into easily accessible `.txt` and `.root` formats.

## 📌 Overview

The pipeline executes the following workflow automatically via `run_pipeline.sh`:
1. **Data Extraction:** Reads a list of `run_number` and `event_number` pairs, locates the corresponding `.root` files in EOS, applies predefined cuts, and converts the SciFi hit data into PyTorch `.pt` tensors using `root2pt.py`.
2. **Deep Learning Inference:** Runs three distinct ResNet models (one for classification, two for energy reconstruction) on the extracted `.pt` tensors.
3. **Consolidation:** Merges the output of all three models into a single tabular format and saves the final results as `final_results.txt` and `final_results.root` using `save_results.py`.

---

## 🛠 Prerequisites & Environments

Because the SND@LHC software (`sndsw`) and modern Deep Learning frameworks (PyTorch) often have conflicting Python versions and dependencies, this pipeline utilizes a **dual-environment approach**. The bash script automatically switches between them.

### 1. SNDSW Environment (For ROOT parsing)
The pipeline automatically sources the required `sndsw` environment from CVMFS (`/cvmfs/sndlhc.cern.ch/SNDLHC-2025/Oct7/setUp.sh`) to access the `ROOT` and `SndlhcGeo` libraries. You do not need to install anything for this step if you are running on LXPLUS.

### 2. Deep Learning Environment (Conda)
You need an environment with standard machine learning and data processing libraries to run the inference and consolidation scripts. 

**Required Packages:**
*   `torch` (PyTorch)
*   `pandas`
*   `numpy`
*   `uproot`
*   `awkward`

**Using Your Own Environment:**
By default, the script points to a Conda environment named `myenv`. **You are highly encouraged to use your own environment or virtualenv**, provided it contains the packages listed above. Simply update the `CONDA_ENV_NAME` variable inside `run_pipeline.sh` to match your environment's name.

---

## 🚀 How to Use

### Step 1: Prepare the Event List
Create a text file (e.g., `events_list.txt`) containing the events you want to process. Format it with the `run_number` and `event_number` separated by a space, one pair per line:

```text
10919 123456
10919 123457
10919 123458
