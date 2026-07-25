import torch
import numpy as np
import os.path
from torch.utils.data import DataLoader, TensorDataset
import config
import pandas as pd
#from dl_recon_core.data_loader_energy import *
#from dl_recon_core.data_loader_common_functions import *

def plot_1d_beam_energy_graphs(true_en_list, qdc_energy_list,out_name ,xlabel="True Energy [GeV]",ylabel='Average QDC Energy[GeV]',title="Average QDC vs True Energy",outdir="qdc_comparision",show_ideal=False):
    plt.figure()
    # Sum over channels, width, and height dimensions (dimensions 1, 2, 3)
    for j in range(len(qdc_energy_list)):
        total_qdc_scifi=qdc_energy_list[j]
        true_en=true_en_list[j]

        average_scifi_qdc = []
        std_scifi_qdc = []

        average_scifi_qdc.append(total_qdc_scifi.mean().item())
        print(average_scifi_qdc)
        std_scifi_qdc.append(total_qdc_scifi.std().item())

        plt.errorbar(true_en, average_scifi_qdc, yerr=std_scifi_qdc, fmt='s-',alpha=0.7)
    
    if show_ideal:
        ax = plt.gca()
        ax.axline((0, 0), slope=1, linestyle='--', color='black', label='y=x')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    if os.path.exists(outdir) is False:
        os.mkdir(outdir)
    plt.savefig(f"{outdir}/{out_name}_{title}.png", dpi=300)
    plt.clf()

def convert_numpy_cpu(predicted_en):
    if isinstance(predicted_en, torch.Tensor):
        # Detach from graph
        predicted_en = predicted_en.detach()

        # Move to CPU only if needed
        if predicted_en.device.type != "cpu":
            predicted_en = predicted_en.cpu()

        # Convert to NumPy
        predicted_en = predicted_en.numpy()
    return predicted_en

def load_checkpoint(model, checkpoint):
    state_dict = checkpoint['state_dict']
    new_state_dict = {}

    if list(state_dict.keys())[0].startswith("module."):
        # checkpoint was saved with DataParallel → remove "module."
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v
    else:
        # checkpoint was saved without DataParallel → keep as-is
        new_state_dict = state_dict

    model.load_state_dict(new_state_dict)
    return model


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def compute_energy_metrics(true_en, predicted_en, bins,plot_save_name,particle_name):
    # Assign each event to a bin index
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_idx = np.digitize(true_en, bins) - 1  
    nbins = len(bins) - 1

    # Preallocate arrays
    bias = np.full(nbins, np.nan)
    rel_bias = np.full(nbins, np.nan)
    resolution = np.full(nbins, np.nan)
    rel_resolution = np.full(nbins, np.nan)
    mean_response = np.full(nbins, np.nan)
    median_response = np.full(nbins, np.nan)
    res_response = np.full(nbins, np.nan)

    response = predicted_en / true_en

    for i in range(nbins):
        idx = bin_idx == i
        if np.sum(idx) < 3:
            continue
        
        t = true_en[idx]
        p = predicted_en[idx]
        r = response[idx]


        ith_bias = p-t
        ith_rel_bias = ith_bias/t

        bias[i] = np.mean(ith_bias)
        rel_bias[i] = np.mean(ith_rel_bias)
        resolution[i] = np.std( ith_bias)
        rel_resolution[i] = np.std( ith_rel_bias )

        mean_response[i] = np.mean(r)
        median_response[i] = np.median(r)
        res_response[i] = np.std(r)


    # resolution(std of bias) graph
    #rel resolution(std of rel bias)
    # response graph with its std value-> change std value
    # mean of bias graph
    # mean of relative bias
    
    ## res vs Etrue graphs
    plt.figure()
    plt.plot(bin_centers, rel_resolution, "o-")
    plt.xlabel(r"$E_{True}$[GeV]")
    plt.ylabel(r"Energy Resolution")
    plt.title(r"Energy Resolution vs $E_{True}$"+f" of {particle_name}")
    for y in [0.1, 0.2]:
        plt.axhline(y, linestyle="--", linewidth=1, alpha=0.6)
        plt.text(bin_centers[-1], y, f"  {y}", va="center", ha="left")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{plot_save_name}_resolution.png", dpi=300)
    plt.close()


    # 4 graphs

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # --- Bias ---
    axes[0,0].plot(bin_centers,  bias, "o-")
    axes[0,0].set_xlabel(r"$E_{True}$[GeV]")
    axes[0,0].set_ylabel(r"Mean of Bias  $(E_{True} - E_{reco})$")
    axes[0,0].set_title(r"Mean of Bias vs $E_{True}$")
    axes[0,0].grid(True, alpha=0.3)

    # --- Relative Bias ---
    axes[0,1].plot(bin_centers, rel_bias, "o-")
    axes[0,1].set_xlabel(r"$E_{True}$[GeV]")
    axes[0,1].set_ylabel(r"Mean of Relative Bias  $(E_{True} - E_{reco})/E_{True}$")
    axes[0,1].set_title(r"Mean of Relative Bias vs $E_{True}$")
    axes[0,1].grid(True, alpha=0.3)

    # --- Resolution ---
    axes[1,0].plot(bin_centers, resolution, "o-")
    axes[1,0].set_xlabel(r"$E_{True}$[GeV]")
    axes[1,0].set_ylabel(r"Resolution")
    axes[1,0].set_title(r"Resolution vs $E_{True}$")
    axes[1,0].grid(True, alpha=0.3)

    # --- Relative Resolution ---
    axes[1,1].plot(bin_centers, rel_resolution, "o-")
    axes[1,1].set_xlabel(r"$E_{True}$[GeV]")
    axes[1,1].set_ylabel(r"Relative Resolution")
    axes[1,1].set_title(r"Relative Resolution vs $E_{True}$")
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{plot_save_name}_bias_resolution.png", dpi=200)
    plt.close()

    # Response plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    # --- Left: Response vs Energy ---
    axes[0].plot(bin_centers, mean_response, "o-", label="Mean Response")
    axes[0].plot(bin_centers, median_response, "s-", label="Median Response")
    axes[0].set_xlabel(r"$E_{True}$[GeV]")
    axes[0].set_ylabel(r"Response  $(E_{reco}/E_{True})$")
    axes[0].set_title(r"Response vs $E_{True}$"+f" of {particle_name}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Right: Response Resolution vs Energy ---
    axes[1].plot(bin_centers, res_response, "o-", color="red")
    axes[1].set_xlabel(r"$E_{True}$[GeV]")
    axes[1].set_ylabel(r"Response Resolution  $\sigma(E_{reco}/E_{True})$")
    axes[1].set_title(r"Response Resolution vs $E_{True}$"+f" of {particle_name}")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{plot_save_name}_response_combined.png", dpi=200)
    plt.close()

    return [bin_centers, rel_resolution]




def plot_simple_plots(y_tensor,x_tensor,y_label=r"$E_{True}$[GeV]",x_label=r"$E_{reconstructed}$[GeV]",title=r"$E_{reconstructed}$ vs $E_{True}$",plot_save_name="ex"):
    y_tensor=convert_numpy_cpu(y_tensor)
    x_tensor=convert_numpy_cpu(x_tensor)
    max_en=np.max(x_tensor)
    min_en=np.min(x_tensor)
    N=50
    bins = np.linspace(min_en,max_en,N)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    bins_recon = np.linspace(np.min(y_tensor),np.max(y_tensor),N)

    cmap = plt.cm.viridis.copy()
    cmap.set_under("white")

    plt.figure()
    plt.hist2d(x_tensor, y_tensor, bins=(bins, bins_recon), cmap=cmap, vmin=0.1)
    plt.colorbar(label="Counts")
    
    # YENİ: İdeal durumu (y = x) gösteren kırmızı kesik çizgi
    plt.plot(bins, bins, 'r--', label='Ideal: y = x', linewidth=1.5)
    plt.legend()

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(f"{plot_save_name}.png", dpi=200)
    plt.close()

def save_from_eos_pipeline(excell_dirs, energy, res_val, highest_bin_val):
    # 1. Tuple'dan yolları çıkar
    eosdir, test_dir = excell_dirs
    test_dir = test_dir
    clean_eosdir = os.path.normpath(eosdir)
    
    training_name = os.path.basename(clean_eosdir)
    
    parent_dir = os.path.dirname(clean_eosdir)
    filename = os.path.join(parent_dir, "ALL_RESULTS.xlsx") 
    
    create_results_on_excell(filename, training_name, test_dir, energy, res_val, highest_bin_val)
    print(f"[{training_name}],{test_dir} sonuçları {filename} dosyasına eklendi.")


def create_results_on_excell(filename, training_name, test_dir, energy_str, res_val, highest_bin_val):
    
    # 1. Sütun yapısı
    metrics = ["Resolution", "Highest Bin"]
    energies = ["50GeV", "100GeV", "150GeV", "200GeV", "250GeV", "300GeV"]
    
    # Senin kodundan gelen hazır string (örn: "1-400GeV" veya "50GeV") şablonda yoksa sütunlara ekle
    if energy_str not in energies:
        energies.append(energy_str)

    col_index = pd.MultiIndex.from_product([metrics, energies], names=["Metric", "Energy"])

    # 2. Dosya kontrolü ve okuma
    if os.path.exists(filename):
        df = pd.read_excel(filename, header=[0, 1], index_col=[0, 1])
    else:
        row_index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=["Directory", "Test Dir"])
        df = pd.DataFrame(columns=col_index, index=row_index)

    # 3. Satır kontrolü
    target_row = (training_name, test_dir)

    if target_row not in df.index:
        new_row = pd.DataFrame(index=pd.MultiIndex.from_tuples([target_row], names=["Directory", "Test Dir"]), columns=df.columns)
        df = pd.concat([df, new_row])

    # 4. Veriyi gir
    df.loc[target_row, ("Resolution", energy_str)] = res_val
    df.loc[target_row, ("Highest Bin", energy_str)] = highest_bin_val

    # 5. Kaydet
    df.sort_index(inplace=True)
    df.to_excel(filename)
    # print(f"Excel güncellendi: {training_name} -> {test_dir} ({energy_str})") # İstersen açabilirsin


def plot_res_energy(predicted_en, true_en, plot_save_name, particle_name=" ", BEAM_OR_TRUE_ENERGY="True",excell_dirs=False):
    print("predicted energies", predicted_en)
    predicted_en = convert_numpy_cpu(predicted_en)
    true_en = convert_numpy_cpu(true_en)
    max_en = np.max(true_en)
    min_en = np.min(true_en)
    print("max min true en", max_en, min_en)
    
    N = 50
    bins = np.linspace(min_en, max_en, N)
    bins_recon = np.linspace(np.min(predicted_en), np.max(predicted_en), N)

    cmap = plt.cm.viridis.copy()
    cmap.set_under("white")
    
    # Relative difference and response
    diff = (predicted_en - true_en) / true_en
    response = predicted_en / true_en
    
    # --- HELPER: Define the LaTeX label once to ensure consistency and clean code ---
    # This creates a string like: "$E_{Beam}$" or "$E_{True}$"
    label_true = rf"$E_{{{BEAM_OR_TRUE_ENERGY}}}$"
    label_reco = r"$E_{reconstructed}$"

    ## 1) e true vs e recon 2d histograms (Large)
    plt.figure()
    plt.hist2d(true_en, predicted_en, bins=(bins, bins_recon), cmap=cmap, vmin=0.1)
    plt.colorbar(label="Counts")
    
    plt.xlabel(rf"{label_true} [GeV]")    
    plt.ylabel(rf"{label_reco} [GeV]")
    plt.title(rf"{label_reco} vs {label_true} of {particle_name}")
    
    # Fixed string concatenation here
    plt.plot(bins_recon, bins_recon, 'r--', label=rf'Ideal: {label_reco} = {label_true}')
    plt.legend()
    plt.savefig(f"{plot_save_name}_2D_reco_vs_true.png", dpi=200)
    plt.close()

    ## 2) etrue vs erecon 2d histograms (Smaller Bins)
    plt.figure()
    plt.hist2d(true_en, predicted_en, bins=config.e_recon_etrue_small, cmap=cmap, vmin=0.1)
    plt.colorbar(label="Counts")
    
    plt.xlabel(rf"{label_true} [GeV]")    
    plt.ylabel(rf"{label_reco} [GeV]")
    plt.title(rf"{label_reco} vs {label_true} of {particle_name}")
    plt.plot(bins_recon, bins_recon, 'r--', label=rf'Ideal: {label_reco} = {label_true}')
    plt.legend()
    plt.savefig(f"{plot_save_name}_2D_reco_vs_true_smaller.png", dpi=200)
    plt.close()

    ## 3) bias and response 2d histograms
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist2d(true_en, diff, bins=(bins, np.linspace(-2, 2, 40)), cmap=cmap, vmin=0.1)
    plt.colorbar(label="Counts")
    plt.xlabel(rf"{label_true} [GeV]") # Fixed
    plt.ylabel(rf"({label_reco} - {label_true}) / {label_true}") # Fixed
    plt.title(rf"Relative Difference vs {label_true} of {particle_name}") # Fixed

    plt.subplot(1, 2, 2)
    plt.hist2d(true_en, response, bins=(bins, np.linspace(0, 4, 40)), cmap=cmap, vmin=0.1)
    plt.colorbar(label="Counts")
    plt.xlabel(rf"{label_true} [GeV]") # Fixed
    plt.ylabel(rf"Response ({label_reco} / {label_true})") # Fixed
    plt.title(rf"Response vs {label_true} of {particle_name}") # Fixed

    plt.tight_layout()
    plt.savefig(f"{plot_save_name}_2D.png", dpi=200)
    plt.close()

    ith_rel_bias = (predicted_en - true_en ) / true_en
    rel_resolution = np.std(ith_rel_bias)
    counts, bins = np.histogram(predicted_en, bins=bins_recon)
    max_idx = np.argmax(counts) 
    max_bin_start = bins[max_idx]
    max_bin_end = bins[max_idx + 1]
    max_count = counts[max_idx]

    stats_text = (f"Resolution: {rel_resolution:.4f}\n"
                f"Highest Bin: {(max_bin_start + max_bin_end)/2}")

    plt.hist(predicted_en, bins=bins_recon, label=stats_text)
    plt.yscale('log')
    plt.xlabel(label_reco)
    plt.ylabel('Counts')
    plt.title("Reconstructed Energy Histogram")
    plt.legend(loc='best') 
    plt.grid(True, which='both', alpha=0.3)

    plt.savefig(f"{plot_save_name}_1dhist.png", dpi=300)
    plt.close()

    if excell_dirs:
        if np.std(true_en)==0:
            energy = f"{int(np.mean(true_en))}GeV"
        else:
            energy=f"{np.min(true_en)}-{np.max(true_en)}GeV"
        save_from_eos_pipeline(excell_dirs, energy, rel_resolution, ((max_bin_start + max_bin_end)/2)-int(np.mean(true_en)))

 
    # Metrics calculations (unchanged)
    res_bin1 = compute_energy_metrics(true_en, predicted_en, bins, plot_save_name, particle_name)
    if config.bins2.size > 0:
        res_bin2 = compute_energy_metrics(true_en, predicted_en, config.bins2, plot_save_name+"bins_smaller", particle_name)
    else:
        res_bin2 = 0
    if config.bins3.size > 0:
        res_bin3 = compute_energy_metrics(true_en, predicted_en, config.bins3, plot_save_name+"bins_ideal_above5", particle_name)
    else:
        res_bin3 = 0
    if config.bins4.size > 0:
        res_bin4 = compute_energy_metrics(true_en, predicted_en, config.bins4, plot_save_name+"bins_above_10gev", particle_name)
    else:
        res_bin4 = 0
        
    return [res_bin1, res_bin2, res_bin3, res_bin4]
def test_model_params_hist(scifi_sig,logits,energy_list_check, filename1 ,out_dir,common_dir):
    scifi_fac = convert_numpy_cpu(torch.load(out_dir+"/model_scifi_param.pt"))
    scifi_sig = convert_numpy_cpu(scifi_sig)
    logits = convert_numpy_cpu(logits)
    print(logits.shape)
    print(scifi_sig.shape)

    correction_fact = logits - scifi_fac*scifi_sig
    plot_simple_plots(scifi_fac*scifi_sig,energy_list_check,r"$E_{True}$[GeV]", r"$E_{recon, Linear term}$[GeV]", r"$E_{recon, Linear term}$[GeV] vs $E_{True}$[GeV] of"+filename1,common_dir+filename1+"scifi_Etrue")
    plot_simple_plots(correction_fact,energy_list_check,r"$E_{True}$[GeV]", r"$E_{recon, DL term}$[GeV]", r"$E_{recon, DL term}$[GeV] vs $E_{True}$[GeV] of"+filename1,common_dir+filename1+"DL_Etrue")
    plot_simple_plots(scifi_fac*scifi_sig, correction_fact, r"$E_{recon, Linear term}$[GeV]" , r"$E_{recon, DL term}$[GeV]", r" $E_{recon, Linear term}$[GeV] vs. $E_{recon, DL term}$[GeV] of"+filename1,common_dir+filename1+"scifi_DL")

