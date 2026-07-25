import sys
import glob
import os
import ROOT
import numpy as np
import torch
import torch.nn.functional as F
import uproot
import time
import argparse
import SndlhcGeo
import shipunit

# ---------------------------------------------------------
# (Helper Functions)
# ---------------------------------------------------------

def hdw_all_fast_conv(scifi_qdc, HDW_CHANNEL=40):
    delta_ch = HDW_CHANNEL
    N = scifi_qdc.shape[0]
    hits = (scifi_qdc != 0).float()
    kernel_size = 2 * delta_ch + 1
    kernel = torch.ones(1, 1, kernel_size, device=scifi_qdc.device)
    kernel[:, :, delta_ch] = 0.0
    plane_hdw = torch.zeros(N, 2, 5, device=scifi_qdc.device)

    for station in range(5):
        for plane in range(2):
            x = hits[:, plane, station, :].unsqueeze(1)
            neighbors = F.conv1d(x, kernel, padding=delta_ch).squeeze(1)
            wi = neighbors * hits[:, plane, station, :]
            plane_hdw[:, plane, station] = wi.sum(dim=-1)

    station_hdw = plane_hdw.sum(dim=1)
    event_hdw, max_index = station_hdw.max(dim=1)
    idx = max_index.view(N, 1, 1).expand(-1, 2, 1)
    plane_hdw_max = plane_hdw.gather(dim=2, index=idx).squeeze(2)
    hor_hdw = plane_hdw_max[:, 0]
    ver_hdw = plane_hdw_max[:, 1]
    return station_hdw, event_hdw, hor_hdw, ver_hdw

def scifi_array_id(detID):
    n_plane = (detID // 1000000) % 10
    n_vert = (detID // 100000) % 10
    n_chan = detID % 1000
    n_chan += 128 * ((detID // 1000) % 10)
    n_chan += 128 * 4 * ((detID // 10000) % 10)
    return n_vert, n_plane - 1, n_chan

def find_highest_bin(values, num_bins=40):
    if len(values) == 0: return 0.0
    if np.min(values) == np.max(values): return float(values[0])
    counts, bin_edges = np.histogram(values, bins=num_bins, range=(0, 16))
    max_bin_idx = np.argmax(counts)
    return float((bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2.0)

def get_event_aggregates(indices, signals, times, num_layers, layer_dim=1, mean_time=None, time_window_high=None, time_window_low=None):
    idx_arr = np.array(indices, dtype=np.int16)
    sig_arr = np.array(signals, dtype=np.float32)
    time_arr = np.array(times, dtype=np.float32)
    
    if time_window_high is not None and len(time_arr) > 0:
        time_mask = (time_arr >= (mean_time - time_window_low)) & (time_arr <= (mean_time + time_window_high))
        idx_arr = idx_arr[time_mask]
        sig_arr = sig_arr[time_mask]
    
    total_hits = len(sig_arr)
    total_qdc = np.sum(sig_arr) if total_hits > 0 else 0.0
    hits_per_layer = np.zeros(num_layers, dtype=np.int32)
    qdc_per_layer = np.zeros(num_layers, dtype=np.float32)

    if total_hits > 0 and len(idx_arr) > 0:
        layers = idx_arr[:, layer_dim]
        for i in range(num_layers):
            layer_mask = (layers == i)
            hits_per_layer[i] = np.sum(layer_mask)
            qdc_per_layer[i] = np.sum(sig_arr[layer_mask])

    return total_hits, total_qdc, hits_per_layer, qdc_per_layer

def calculate_hdw(idx_list, sig_list, hdw_channel=40):
    idx_arr = np.array(idx_list, dtype=int)
    sig_arr = np.array(sig_list, dtype=np.float32)
    dense = np.zeros((2, 5, 1536), dtype=np.float32)
    dense[idx_arr[:, 0], idx_arr[:, 1], idx_arr[:, 2]] = sig_arr
    dense = torch.from_numpy(dense).unsqueeze(0)
    return hdw_all_fast_conv(dense, hdw_channel)

def passes_shower_density_cut(idx_list, sig_list, radius=64, min_hits=10):
    idx_arr = np.array(idx_list, dtype=int)
    sig_arr = np.array(sig_list, dtype=np.float32)
    dense = np.zeros((2, 5, 1536), dtype=np.float32)
    dense[idx_arr[:, 0], idx_arr[:, 1], idx_arr[:, 2]] = sig_arr

    max_layer = np.argmax(np.sum(dense, axis=(0, 2)))
    max_hor = np.argmax(dense[0, max_layer, :])
    max_ver = np.argmax(dense[1, max_layer, :])

    hor_start = max(0, max_hor - radius)
    hor_end   = min(1536, max_hor + radius)
    ver_start = max(0, max_ver - radius)
    ver_end   = min(1536, max_ver + radius)

    if hor_start == 0: hor_end = 2 * radius
    elif hor_end == 1536: hor_start = 1536 - (2 * radius)

    if ver_start == 0: ver_end = 2 * radius
    elif ver_end == 1536: ver_start = 1536 - (2 * radius)

    hor_mask = (idx_arr[:, 0] == 0) & (idx_arr[:, 2] >= hor_start) & (idx_arr[:, 2] < hor_end)
    ver_mask = (idx_arr[:, 0] == 1) & (idx_arr[:, 2] >= ver_start) & (idx_arr[:, 2] < ver_end)
    
    return np.sum(hor_mask), np.sum(ver_mask)



def process_target_event(tree, scifi_geometry, target_event_number):
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        
        hdr = tree.EventHeader
        if hdr.GetEventNumber() != target_event_number:
            continue
            
        print(f"\n[INFO] Event {target_event_number} found. (Entry: {i})")

        if not hasattr(tree, 'Digi_ScifiHits') or len(tree.Digi_ScifiHits) < 10:
            print("❌ no cuts passed")
            return None

        if hdr.GetBeamMode() != 11 or not hdr.isIP1():
            print(f"❌ BeamMode ({hdr.GetBeamMode()}) not equal to 11 or isIP1() False.")
            return None

        scifi_geometry.InitEvent(hdr)

        evt_scifi_idx, evt_scifi_sig, evt_scifi_time = [], [], []
        layer_channels = {}
        event_vetoed = False

        for aHit in tree.Digi_ScifiHits:
            if not aHit.isValid(): continue
            detID = aHit.GetDetectorID()
            n_vert, n_plane, n_chan = scifi_array_id(detID)

            sig = aHit.GetSignal()
            rawtime = aHit.GetTime()
            corrected_time = scifi_geometry.GetCorrectedTime(aHit.GetDetectorID(), rawtime * shipunit.snd_TDC2ns, 0) / shipunit.snd_TDC2ns

            evt_scifi_idx.append([n_vert, n_plane, n_chan])
            evt_scifi_sig.append(sig)
            evt_scifi_time.append(corrected_time)

            layer_key = (n_plane, n_vert)
            if layer_key not in layer_channels:
                layer_channels[layer_key] = []
            layer_channels[layer_key].append(n_chan)

        if len(evt_scifi_idx) <= 10:
            print("❌ less than 10 scifi hits")
            return None

        valid_planes = []
        all_planes_in_event = set(plane for plane, vert in layer_channels.keys())
        for p in all_planes_in_event:
            if (p, 0) in layer_channels and (p, 1) in layer_channels:
                valid_planes.append(p)
                
        unique_planes = sorted(valid_planes)
        has_consecutive = False
        for j in range(len(unique_planes) - 1):
            if unique_planes[j+1] == unique_planes[j] + 1:
                has_consecutive = True
                break
                
        if not has_consecutive:
            print("❌ no consecutive hits.")
            return None 

        evt_us_idx, evt_us_sig, evt_us_time = [], [], []
        evt_ds_h_idx, evt_ds_h_sig, evt_ds_h_time = [], [], []
        evt_ds_v_idx, evt_ds_v_sig, evt_ds_v_time = [], [], []
        skip_event_due_to_veto = False

        for aHit in tree.Digi_MuFilterHits:
            hit_id = aHit.GetDetectorID()
            bar = (hit_id % 1000)
            hit_system = aHit.GetSystem()
            hit_plane_2 = aHit.GetPlane()

            if hit_system == 1:
                skip_event_due_to_veto = True
                break

            elif hit_system == 2:
                for side in range(aHit.GetnSides()):
                    for channel in range(aHit.GetnSiPMs()):
                        ch = 8 * side + channel
                        real_channel = bar * 8 + channel
                        sig = aHit.GetSignal(ch)
                        if sig > 0:
                            evt_us_idx.append([side, hit_plane_2, real_channel])
                            evt_us_sig.append(sig)
                            evt_us_time.append(aHit.GetTime(ch))

            elif hit_system == 3:
                for side in range(aHit.GetnSides()):
                    for channel in range(aHit.GetnSiPMs()):
                        ch = 8 * side + channel
                        sig = aHit.GetSignal(ch)
                        if sig > 0:
                            if aHit.isVertical():
                                real_channel = bar - 60
                                evt_ds_v_idx.append([hit_plane_2, real_channel])
                                evt_ds_v_sig.append(sig)
                                evt_ds_v_time.append(aHit.GetTime(ch))
                            else:
                                real_channel = bar
                                evt_ds_h_idx.append([side, hit_plane_2, real_channel])
                                evt_ds_h_sig.append(sig)
                                evt_ds_h_time.append(aHit.GetTime(ch))

        if skip_event_due_to_veto:
            print("❌ veto hit exists")
            return None

        print("✅ Passed all the cuts")

        # -------------------------------------------------------------
        # Gather outputs
        ith_hit_x_64, ith_hit_y_64 = passes_shower_density_cut(evt_scifi_idx, evt_scifi_sig, 64)
        ith_hit_x_128, ith_hit_y_128 = passes_shower_density_cut(evt_scifi_idx, evt_scifi_sig, 128)
        ith_station_hdw, ith_max_total_hdw, ith_max_hor_hdw, ith_max_ver_hdw = calculate_hdw(evt_scifi_idx, evt_scifi_sig, 40)

        mean_time_scifi = find_highest_bin(evt_scifi_time)
        mean_time_us = find_highest_bin(evt_us_time)
        mean_time_ds = find_highest_bin(evt_ds_h_time + evt_ds_v_time)

        h_sf, q_sf, hl_sf, ql_sf = get_event_aggregates(evt_scifi_idx, evt_scifi_sig, evt_scifi_time, 5, 1)
        h_us, q_us, hl_us, ql_us = get_event_aggregates(evt_us_idx, evt_us_sig, evt_us_time, 5, 1)
        h_dsh, q_dsh, hl_dsh, ql_dsh = get_event_aggregates(evt_ds_h_idx, evt_ds_h_sig, evt_ds_h_time, 3, 1)
        h_dsv, q_dsv, hl_dsv, ql_dsv = get_event_aggregates(evt_ds_v_idx, evt_ds_v_sig, evt_ds_v_time, 4, 0)

        event_dict = {
            "run_id": torch.tensor([hdr.GetRunId()], dtype=torch.int32),
            "event_number": torch.tensor([hdr.GetEventNumber()], dtype=torch.int32),
            "event_time": torch.tensor([hdr.GetEventTime()], dtype=torch.int64),
            
            "scifi_indices": [torch.tensor(evt_scifi_idx, dtype=torch.int16)],
            "scifi_signals": [torch.tensor(evt_scifi_sig, dtype=torch.float32)],
            "scifi_hit_time": [torch.tensor(evt_scifi_time, dtype=torch.float32)],
            
            "us_indices": [torch.tensor(evt_us_idx, dtype=torch.int16)],
            "us_signals": [torch.tensor(evt_us_sig, dtype=torch.float32)],
            "us_signals_time": [torch.tensor(evt_us_time, dtype=torch.float32)],
            
            "ds_h_indices": [torch.tensor(evt_ds_h_idx, dtype=torch.int16)],
            "ds_h_signals": [torch.tensor(evt_ds_h_sig, dtype=torch.float32)],
            "ds_h_times": [torch.tensor(evt_ds_h_time, dtype=torch.float32)],
            
            "ds_v_indices": [torch.tensor(evt_ds_v_idx, dtype=torch.int16)],
            "ds_v_signals": [torch.tensor(evt_ds_v_sig, dtype=torch.float32)],
            "ds_v_times": [torch.tensor(evt_ds_v_time, dtype=torch.float32)],

            "scifi_mean_hit_time": [float(mean_time_scifi)],
            "us_mean_hit_time": [float(mean_time_us)],
            "ds_mean_hit_time": [float(mean_time_ds)],

            "scifi_hitx_in_64r": [ith_hit_x_64],
            "scifi_hity_in_64r": [ith_hit_y_64],
            
            "max_total_hdw": [ith_max_total_hdw],
            "station_hdw": [ith_station_hdw],
            
            "scifi_notime_total_hits": torch.tensor([h_sf], dtype=torch.int32),
            "scifi_notime_total_qdc": torch.tensor([q_sf], dtype=torch.float32),
            "scifi_notime_hits_per_layer": torch.from_numpy(np.array([hl_sf], dtype=np.int32)),
            
            "us_notime_total_hits": torch.tensor([h_us], dtype=torch.int32),
            "us_notime_total_qdc": torch.tensor([q_us], dtype=torch.float32),
            
            "dsh_notime_total_hits": torch.tensor([h_dsh], dtype=torch.int32),
            "dsh_notime_total_qdc": torch.tensor([q_dsh], dtype=torch.float32),
            
            "dsv_notime_total_hits": torch.tensor([h_dsv], dtype=torch.int32),
            "dsv_notime_total_qdc": torch.tensor([q_dsv], dtype=torch.float32),
        }
        
        return event_dict

    print(f"❌ Event {target_event_number} is not in the root file.")
    return None

# ---------------------------------------------------------
# MAIN EXECUTION 
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process LHC events.")
    parser.add_argument("--run", type=int, help="Run Number (e.g., 10919)")
    parser.add_argument("--event", type=int, help="Event Number (e.g., 123456)")
    parser.add_argument("--txt", type=str, help="Text file containing list of run and event pairs (one per line, e.g., '10919 123456')")
    parser.add_argument("--path", type=str, default="/eos/experiment/sndlhc/convertedData/physics/", help="Base path to data")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory for .pt file(s)")
    args = parser.parse_args()

    # Create list of tasks to process
    tasks = []
    if args.txt:
        if not os.path.exists(args.txt):
            print(f"[ERROR] Provided text file not found: {args.txt}")
            sys.exit(1)
        with open(args.txt, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    tasks.append((int(parts[0]), int(parts[1])))
        if not tasks:
            print("[ERROR] No valid run/event pairs found in text file.")
            sys.exit(1)
    else:
        if args.run is None or args.event is None:
            print("[ERROR] You must provide either --txt OR both --run and --event.")
            sys.exit(1)
        tasks.append((args.run, args.event))

    # Output directory validation
    os.makedirs(args.outdir, exist_ok=True)

    geo_dict = {
        "2022": "/eos/experiment/sndlhc/convertedData/physics/2022/geofile_sndlhc_TI18_V4_2022.root",
        "2023": "/eos/experiment/sndlhc/convertedData/physics/2023/geofile_sndlhc_TI18_V3_2023.root",
        "2024": "/eos/experiment/sndlhc/convertedData/physics/2024/geofile_sndlhc_TI18_V12_2024.root",
        "2025": "/eos/experiment/sndlhc/convertedData/physics/2025/geofile_sndlhc_TI18_V8_2025.root",
        "2026": "/eos/experiment/sndlhc/convertedData/physics/2026/geofile_sndlhc_TI18_V4_2026.root",
    }

    # Tracking loaded resources for batch processing
    current_geo_year = None
    scifi_geometry = None
    geo = None
    
    current_root_file = None
    root_file_obj = None
    tree = None
    
    batch_results = []

    for run, event in tasks:
        run_str = str(run).zfill(6)
        file_name = f"sndsw_raw-{event // 1000000:04d}.root"
        
        # Recursive search
        search_pattern = os.path.join(args.path, "**", f"run_{run_str}", file_name)
        files = glob.glob(search_pattern, recursive=True)
        
        if not files:
            print(f"[ERROR] File not found! Search pattern: {search_pattern}")
            continue

        target_file = files[0]
        print(f"\n[INFO] Processing Run: {run} | Event: {event}")
        print(f"[INFO] File found: {target_file}")

        # Determine year for geometry mapping
        geo_year = None
        geo_path = None
        for year, path in geo_dict.items():
            if f"/{year}/" in target_file:  
                geo_year = year
                geo_path = path
                break
                
        if not geo_path:
            print(f"[ERROR] Could not detect year (2022-2026) from file path: {target_file}")
            continue
            
        # Initialize geometry if not already loaded for this year
        if geo_year != current_geo_year:
            print(f"[INFO] Initializing Geometry file: {geo_path}")
            geo = SndlhcGeo.GeoInterface(geo_path)
            scifi_geometry = geo.modules['Scifi']
            current_geo_year = geo_year

        # Open ROOT file if not already open
        if target_file != current_root_file:
            if root_file_obj:
                root_file_obj.Close()
                
            root_file_obj = ROOT.TFile.Open(target_file, "READ")
            if not root_file_obj or root_file_obj.IsZombie():
                print(f"[ERROR] ROOT file is corrupted or cannot be opened: {target_file}")
                current_root_file = None
                continue

            tree = root_file_obj.Get("rawConv")
            if not tree:
                print("[ERROR] 'rawConv' tree not found in file. (Did you provide an MC file?)")
                root_file_obj.Close()
                root_file_obj = None
                current_root_file = None
                continue
                
            current_root_file = target_file

        # Process the event
        result_dict = process_target_event(tree, scifi_geometry, event)

        if result_dict is not None:
            if args.txt:
                batch_results.append(result_dict)
            else:
                out_path = os.path.join(args.outdir, f"event_{run}_{event}.pt")
                torch.save(result_dict, out_path)
                print(f"[SUCCESS] Result saved: {out_path}")
        else:
            print("[INFO] File not saved because it did not pass the cuts.")

    # Cleanup open file
    if root_file_obj:
        root_file_obj.Close()

    # Save batched results if processing from a text file
# Save batched results if processing from a text file
    if args.txt:
        if batch_results:
            base_name = os.path.splitext(os.path.basename(args.txt))[0]
            out_path = os.path.join(args.outdir, f"{base_name}_batch.pt")
            

            combined_dict = {}
            for key in batch_results[0].keys():
                first_elem = batch_results[0][key]
                
                if isinstance(first_elem, list):
                    combined_dict[key] = []
                    for event in batch_results:
                        combined_dict[key].extend(event[key])
                elif isinstance(first_elem, torch.Tensor):
                    tensors = [event[key] for event in batch_results]
                    combined_dict[key] = torch.cat(tensors, dim=0)
                else:
                    combined_dict[key] = [event[key] for event in batch_results]
            
            torch.save(combined_dict, out_path)
            # -----------------------------------
            
            print(f"\n[SUCCESS] All passed events ({len(batch_results)}) saved to: {out_path}")
        else:
            print("\n[INFO] No events passed the cuts; nothing saved.")