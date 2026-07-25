import os
import glob
import re
import torch
import torch.nn as nn
import config  # Assuming this is in the same directory
from torch.utils.data import DataLoader, TensorDataset, Dataset, ConcatDataset
from dl_recon_core_sparse.data_loader import SNDSparseDataset
from collections import defaultdict
keys_list = [
    "scifi_hitx_in_64r",
    "scifi_hity_in_64r",
    "scifi_hitx_in_128r",
    "scifi_hity_in_128r",
    "max_total_hdw",
    "station_hdw",
    "max_hor_hdw",
    "max_ver_hdw",
    "scifi_notime_total_hits",
    "scifi_notime_total_qdc",
    "scifi_notime_hits_per_layer",
    "scifi_notime_qdc_per_layer",
    "scifi_05usualtime_total_hits",
    "scifi_05usualtime_total_qdc",
    "scifi_05usualtime_hits_per_layer",
    "scifi_05usualtime_qdc_per_layer",
    "scifi_05_18_total_hits",
    "scifi_05_18_total_qdc",
    "scifi_05_18_hits_per_layer",
    "scifi_05_18_qdc_per_layer",
    "scifi_05_22_total_hits",
    "scifi_05_22_total_qdc",
    "scifi_05_22_hits_per_layer",
    "scifi_05_22_qdc_per_layer",
    "scifi_05_23_total_hits",
    "scifi_05_23_total_qdc",
    "scifi_05_23_hits_per_layer",
    "scifi_05_23_qdc_per_layer",
    "us_notime_total_hits",
    "us_notime_total_qdc",
    "us_notime_hits_per_layer",
    "us_notime_qdc_per_layer",
    "us_3usualtime_total_hits",
    "us_3usualtime_total_qdc",
    "us_3usualtime_hits_per_layer",
    "us_3usualtime_qdc_per_layer",
    "dsh_notime_total_hits",
    "dsh_notime_total_qdc",
    "dsh_notime_hits_per_layer",
    "dsh_notime_qdc_per_layer",
    "dsv_notime_total_hits",
    "dsv_notime_total_qdc",
    "dsv_notime_hits_per_layer",
    "dsv_notime_qdc_per_layer",
    "run_id",
    "event_number",
    "event_time"
]

def extract_event_features(scifi_sig, cut_name, prop_dict):
    valid_mask = (scifi_sig != 0)
    hits_per_station = torch.sum(valid_mask, dim=(1, 3)) 
    total_hits = torch.sum(hits_per_station, dim=1)      
    
    clean_qdc = torch.where(valid_mask, scifi_sig, torch.tensor(0.0, device=scifi_sig.device))    
    qdc_per_station = torch.sum(clean_qdc, dim=(1, 3))   
    total_qdc = torch.sum(qdc_per_station, dim=1)      

    topK_values_qdc, _ = torch.topk(qdc_per_station, k=2, dim=1)
    frac_abs_qdc = torch.log(torch.where(topK_values_qdc[:, 1] > 0, topK_values_qdc[:, 0] / topK_values_qdc[:, 1], torch.tensor(1.0, device=scifi_sig.device)))

    topK_values_hits, _ = torch.topk(hits_per_station, k=2, dim=1)
    frac_abs_hits = torch.log(torch.where(topK_values_hits[:, 1] > 0, topK_values_hits[:, 0].float() / topK_values_hits[:, 1].float(), torch.tensor(1.0, device=scifi_sig.device)))

    features = {
        "Total Hits"+cut_name: total_hits.cpu(),
        "Total QDC"+cut_name: total_qdc.cpu(),
        "Log of Fraction Abs QDC"+cut_name: frac_abs_qdc.cpu(),
        "Log of Fraction Abs Hits"+cut_name: frac_abs_hits.cpu()
    }
    for i in range(config.USE_HIGHEST_N_LAYER):
        features[f"Station {i+1} Hits"+cut_name] = hits_per_station[:, i].cpu()
        features[f"Station {i+1} QDC"+cut_name] = qdc_per_station[:, i].cpu()

    for key, val in features.items():
        prop_dict[key].append(val)

    return prop_dict


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        print(torch.version.cuda)
        return torch.device('cuda')
    else:
        return torch.device('cpu')
device = get_default_device()

def load_model(checkpoint_dir, device):
    model_ecal = config.MODEL
    model_ecal.to(device)
    
    ecal_files = glob.glob(os.path.join(checkpoint_dir, "model_ecal_*.pt"))
    if not ecal_files:
        raise FileNotFoundError(f"No checkpoint files found in: {checkpoint_dir}")

    def extract_epoch(filename):
        match = re.search(r"epoch-(\d+)\.pt", os.path.basename(filename))
        return int(match.group(1)) if match else -1

    latest_ecal = max(ecal_files, key=extract_epoch)
    print(f"Loading latest ecal file: {latest_ecal}")

    # Load checkpoint safely across devices
    checkpoint = torch.load(latest_ecal, map_location=device)
    model_ecal.load_state_dict(checkpoint['state_dict'])
        
    if torch.cuda.device_count() > 1:
        print(f"Number of GPUs being used: {torch.cuda.device_count()}")
        model_ecal = nn.DataParallel(model_ecal)
        
    # Extract just the filename without the .pt extension
    name_only = os.path.splitext(os.path.basename(latest_ecal))[0]
    
    return model_ecal, name_only


def all_in_one(file_names, dict_cuts, checkpoint_dir, base_save_name="energy_recon"):
    model, dl_name = load_model(checkpoint_dir, device)
    model.eval() 

    model_save_suffix = f"{base_save_name}_{dl_name}"

    for file_path in file_names:
        print(f"Processing file: {file_path}")

        file_dir = os.path.dirname(file_path)
        file_basename = os.path.splitext(os.path.basename(file_path))[0]
        
        # Ensure the output directory exists
        save_dir = file_dir 
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{file_basename}_{model_save_suffix}.pt")

        ith_dataset = SNDSparseDataset([0, file_path], EN_MIN=0, EN_MAX=20000, is_lhcdata=True)
        if len(ith_dataset) == 0:
            print("zero data, passing this")
            continue

        if os.path.exists(save_path):
            dict_results = torch.load(save_path, weights_only=False) 
        else:
            dict_results = {"path_of_model": checkpoint_dir}
        
            for tb_recal in dict_cuts.get("TB_RECALIBRATION_S2Y", [False]):
                for t_win in dict_cuts.get("t_window_data", [(0.5, 2.3)]):
                    for qdc in dict_cuts.get("qdc_threshold_value_scifi_data", [0]):
                        
                        data_name = f"S2Ycal{tb_recal}_qdcD{qdc}_twinD{t_win[0]}_{t_win[1]}"

                        if data_name in dict_results:
                            continue
                        else:
                            ith_dataset.update_hit_cuts( 
                                t_window_high_data=t_win[1], t_window_low_data=t_win[0], qdc_thresh_data=qdc,
                                TB_RECALIBRATION_S2Y=tb_recal
                            )

                            dataloader = DataLoader(
                                ith_dataset, 
                                batch_size=config.BATCH_SIZE,   
                                shuffle=False,        
                                num_workers=4, 
                                pin_memory=True
                            )
                            
                            temp_array = []
                            temp_idx = []
                            prop_dict = defaultdict(list)
                            
                            for batch in dataloader:
                                scifi, _, _, idx = batch
                                temp_idx.append(idx)
                                scifi = scifi.to(device)

                                with torch.no_grad():
                                    output = model(scifi).reshape(-1)
                                    temp_array.append(output.cpu())

                                prop_dict = extract_event_features(scifi, data_name, prop_dict)
                        
                        dict_results[data_name] = torch.cat(temp_array, dim=0)
                        
                        # Concatenate all indices from this pass
                        all_idx = torch.cat(temp_idx, dim=0)
                        dict_results["idx"] = all_idx 

                        # --- NEW LOGIC: Save run_id and event_number based on processed indices ---
                        # We cast to tensor in case the original data is stored as standard lists or numpy arrays
                        raw_data = ith_dataset.data
                        
                        if "run_id" in raw_data:
                            run_ids = raw_data["run_id"] if isinstance(raw_data["run_id"], torch.Tensor) else torch.tensor(raw_data["run_id"])
                            dict_results["run_id"] = run_ids[all_idx]
                            
                        if "event_number" in raw_data:
                            event_nums = raw_data["event_number"] if isinstance(raw_data["event_number"], torch.Tensor) else torch.tensor(raw_data["event_number"])
                            dict_results["event_number"] = event_nums[all_idx]
                        # -------------------------------------------------------------------------
                        
                        for key, list_of_tensors in prop_dict.items():
                            dict_results[key] = torch.cat(list_of_tensors, dim=0)
                            #print(key)
                        torch.save(dict_results, save_path)
                        print("run id", dict_results["run_id"])
                        print("event number", dict_results["event_number"])
                        print("model output",dict_results[data_name])
                        print(f"  -> Saved updated results to: {save_path}\n")
# Example Execution
import sys
if __name__ == "__main__":
    pt_directory = sys.argv[1]
    output_prefix = "MC_electrons_ResNets_SciFi_2layers_R256_100gev_-0.5_s2_q13_ft0_layer5"
    
    model_checkpoint_path = "/eos/experiment/sndlhc/users/beturk/DL_Electron_Reconstruction_Models/Energy_Recon/MC_electrons_ResNets_SciFi_2layers_R256_100gev_-0.5_s2_q13_ft0_layer5" # Update to actual path
    target_files = glob.glob(os.path.join(pt_directory, "*.pt"))
    
    if not target_files:
        print(f"[WARNING] no file found: {pt_directory}")
    else:
        print(f"[INFO] total of {len(target_files)} .pt files found.")
    
        my_cuts = {
            "TB_RECALIBRATION_S2Y": [False],
            "t_window_data": [(0.5, 2.3)],
            "qdc_threshold_value_scifi_data": [0]
        }

        all_in_one(target_files, my_cuts, os.path.join(model_checkpoint_path, "checkpoints"), output_prefix)
