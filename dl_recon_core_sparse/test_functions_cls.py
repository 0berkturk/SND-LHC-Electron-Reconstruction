import torch
import numpy as np
import os.path
from torch.utils.data import DataLoader, TensorDataset
import config
import pandas as pd
#from dl_recon_core.data_loader_common_functions import *

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

import torch

def binary_metrics(y_true, logits, signal_index=1, threshold=0.5):

    if logits.dim() == 1 or logits.size(1) == 1:
        logits = logits.view(-1)
        y_true_bin = (y_true == signal_index).long()  # convert to {0,1}
        probs = torch.sigmoid(logits)
        y_pred_bin = (probs >= threshold).long()
    
    # ----- Multiclass classification -----
    else:
        probs = torch.softmax(logits, dim=1)
        y_true_bin = (y_true == signal_index).long()  # make positive class=signal_index
        pred_class = torch.argmax(probs, dim=1)
        y_pred_bin = (pred_class == signal_index).long()

    # Stats
    TP = torch.sum((y_true_bin == 1) & (y_pred_bin == 1)).item()
    FP = torch.sum((y_true_bin == 0) & (y_pred_bin == 1)).item()
    FN = torch.sum((y_true_bin == 1) & (y_pred_bin == 0)).item()
    TN = torch.sum((y_true_bin == 0) & (y_pred_bin == 0)).item()

    # Metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (TP + TN) / (TP + TN + FP + FN)


    return precision, recall, f1, accuracy




import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_confusion_matrix(logits, targets,out_name,title2=" ",class_names=None,GET_PARTICLES_WITH_INDEX=None):
    if class_names is None:
        class_names = [r"$\nu_e$", r"$\nu_\mu$", r"$\nu_\tau$","NC"]
        GET_PARTICLES_WITH_INDEX = [0,1,2,3]

    preds = torch.argmax(logits, dim=1)

    cm = confusion_matrix(targets.numpy(), preds.numpy(), labels=GET_PARTICLES_WITH_INDEX)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=False,annot_kws={"size": 6})
    plt.ylabel('True label',fontsize=6)
    plt.xlabel('Predicted label',fontsize=6)
    plt.xticks(rotation=45, ha='right', fontsize=6)  # 🔹 X ekseni etiketleri (class_names)
    plt.yticks(rotation=0, fontsize=6)
    plt.title(f'Confusion Matrix{title2}',fontsize=6)
    plt.savefig(out_name,dpi=300)
    plt.clf()

    plt.close()

# Example usage:
# plot_confusion_matrix(logits, targets)
def order_data(list_rej, list_eff):
    data_frame = pd.DataFrame({'x': list_rej, 'y': list_eff})
    data_frame = data_frame.drop_duplicates(subset=['x', 'y'])
    #cvt = cvt.sort_values(by=['x'], ascending=True)
    #cvt.sort_values('y', inplace=True)
    data_frame = data_frame[(data_frame['y'] != 1) | ((data_frame['y'] == 1) & (data_frame['x'] == data_frame.loc[data_frame['y'] == 1, 'x'].max()))]
    return data_frame["x"], data_frame["y"]

def convert_2_binary_probs(logits, targets,en3d,signal_index):
    if config.IS_BINARY:
        logits=logits.reshape(-1)
        targets = (targets == signal_index).long()
        probs = logits#torch.sigmoid(logits)
        print("\nshape logits and targets",logits.shape, targets.shape)
        signal_probs = probs[targets==1]
        background_probs = probs[targets==0]
    else:
        print("\nshape logits and targets", logits.shape, targets.shape)
        targets = (targets == signal_index).long()
        probs = torch.softmax(logits,dim=1)
        binary_probs = probs[:,signal_index] ##get only first one, or binary output
        print(binary_probs)
        signal_probs = binary_probs[targets==1]
        background_probs = binary_probs[targets==0]
    return signal_probs, background_probs, en3d[targets==0]


def s_eff_b_rej(signal_probs, background_probs, out_name, title2=" ", is_for_eletrons=False):
    title= f'Background Rejection vs Electron Efficiency(TB Data)\n{title2}'
    x_label= 'Electron Efficiency'
    y_label = r'Background Rejection (Hadrons)'

    print("signal probs shape",signal_probs.shape)
    print("background probs",background_probs,background_probs.shape)

    number_of_signal = len(signal_probs)
    number_of_background = len(background_probs)

    all_preds = torch.cat([signal_probs, background_probs]).cpu().numpy()
    thresholds = np.unique(np.quantile(all_preds, np.linspace(0, 1, 1000)))


    number_of_selected_signal = np.array([(signal_probs>=ith_threshold).sum().item() for ith_threshold in thresholds])
    number_of_misclassified_signal = np.array([(background_probs>ith_threshold).sum().item() for ith_threshold in thresholds])
    bck_eff = number_of_misclassified_signal/number_of_background
    number_of_misclassified_signal[number_of_misclassified_signal==0] = 1

    signal_eff = number_of_selected_signal/number_of_signal
    background_rej = number_of_background/number_of_misclassified_signal

    dict={}
    dict["signal_eff"] = signal_eff
    dict["background_rej"] = background_rej
    dict["background_eff"] = bck_eff
    out_name = out_name +"_all_backg_rej_eff_signal_ind"
    torch.save(dict, out_name+"_dict.pt")

    ordered_background_rej, ordered_signal_eff = order_data(background_rej,signal_eff)
    #print(thresholds)
    #print("ordered_background_rej",ordered_background_rej)
    #print("ordered_signal_eff",ordered_signal_eff)
    print("number_of_background",number_of_background)
    plt.plot(ordered_signal_eff, ordered_background_rej, color="purple", label="ResNet", linewidth=1)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)

    plt.grid()
    plt.yscale('log')
    plt.legend()
    save_name = out_name + "graph.png"
    plt.savefig(save_name, dpi=300)
    plt.clf()


    """    if is_for_eletrons:
        title= f'Background Efficiency vs Electron Efficiency(TB Data)\n{title2}'
        x_label= 'Electron Efficiency'
        y_label = r'Background Efficiency ($\nu_e$,,$\nu_\mu$, $\nu_\tau$, NC, Hadrons)'
    else:
        title= f'Background Efficiency vs Signal ($\nu_e$) Efficiency(TB Data)\n{title2}'
        x_label= r'Signal ($\nu_e$) Efficiency'
        y_label = r'Background Efficiency ($\nu_\mu$, $\nu_\tau$, NC)'"""

    title= f'Background Efficiency vs Signal Efficiency(TB Data)\n{title2}'
    x_label= r'Signal Efficiency'
    y_label = r'Background Efficiency'

    plt.plot(ordered_signal_eff, 1/ordered_background_rej, color="purple", label="ResNet", linewidth=1)
    plt.title(title)
    plt.xlabel(x_label)
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.ylabel(y_label)
    plt.grid()
    plt.yscale('log')
    plt.legend()
    save_name = out_name + "eff_v2.png"
    plt.savefig(save_name, dpi=300)
    plt.clf()

    return thresholds,signal_eff, background_rej

def find_threshold(thres,e_eff,rej,eff_asked):
    xy1 = pd.DataFrame({'x': thres, 'y': e_eff,'z':rej})
    pd.set_option('display.precision', 32)

    xy1['abs_diff'] = abs(xy1['y'] - eff_asked)
    sorted_xy = xy1.sort_values(by='abs_diff')
    closest_value = sorted_xy.iloc[0]

    e_eff = closest_value['y']

    sorted_xy = sorted_xy.loc[sorted_xy['y'] == e_eff].sort_values(by='x')
    "print(sorted ,sorted_xy)"
    closest_value = sorted_xy.iloc[-1]
    "print(1 ,closest_value)"

    threshold = closest_value['x']
    p_rej = closest_value['z']

    print(e_eff,p_rej)
    return threshold, int(e_eff*100), p_rej

def p_rej_energy(threshold, background_probs, background_en3d, out_name,e_eff_title, bins, is_for_eletrons=False):
    """    if is_for_eletrons:
        title= "Background Rejection vs True Energy at %"+str(e_eff_title)+r" Electron Efficiency"
        y_label = r'Background Rejection ($\nu_e$,,$\nu_\mu$, $\nu_\tau$, NC, Hadrons,Muons)'
    else:
        title= "Background Rejection vs True Energy at %"+str(e_eff_title)+r" Signal ($\nu_e$) Efficiency"
        y_label = r'Background Rejection ($\nu_\mu$, $\nu_\tau$, NC)'"""

    title= "Background Rejection vs True Energy at %"+str(e_eff_title)+r" Signal Efficiency"
    y_label = r'Background Rejection (Hadrons)'

    bins2=  0.5 * (bins[:-1] + bins[1:])

    background_probs = background_probs.cpu().numpy()
    background_en3d = background_en3d.cpu().numpy()

    proton_misidentified = background_en3d[(background_probs>threshold)]

    bin_total_proton, bin, p = plt.hist(background_en3d, bins)
    bin_proton_misidentified, bin, p = plt.hist(proton_misidentified, bins)
    bin_proton_misidentified[bin_proton_misidentified==0] = 1

    bin_proton_rej = bin_total_proton / bin_proton_misidentified

    save_dict={}
    save_dict['bin_proton_rej'] = bin_proton_rej
    save_dict['background_en3d'] = background_en3d
    save_dict['bins2'] = bins2
    save_dict['bins'] = bins
    torch.save(save_dict,out_name+"rej_energy.pt")

    plt.figure()
    plt.plot(bins2, bin_proton_rej, label="DL", color="b")
    plt.hist(background_en3d, bins, label="Total Background per Bin", alpha=0.1, color="aqua", ec="black")

    plt.grid()
    plt.xlabel('True Energy[GeV]')
    plt.ylabel(y_label)
    plt.title(title)
    plt.yscale('log')
    plt.legend()
    plt.savefig(out_name+"_"+str(len(proton_misidentified))+"_"+str(e_eff_title)+"_rej_energy.png", dpi=300)
    plt.clf()

    return threshold#bin_proton_rej,bin_total_proton, bins2




def single_better_p_rej_energy(bin_proton_rej, background_en3d, out_name, e_eff_title,bins,is_for_eletrons=False):
    """    if is_for_eletrons:
        title= "Background Rejection vs True Energy at %"+str(e_eff_title)+r" Electron Efficiency"
        y_label = r'Background Rejection ($\nu_e$,,$\nu_\mu$, $\nu_\tau$, NC, Hadrons)'
    else:
        title= "Background Rejection vs True Energy at %"+str(e_eff_title)+r" Signal ($\nu_e$) Efficiency"
        y_label = r'Background Rejection ($\nu_\mu$, $\nu_\tau$, NC)'"""
    title= "Background Rejection vs True Energy at %"+str(e_eff_title)+r" Signal Efficiency"
    y_label = r'Background Rejection (Hadrons)'
    print(bins)
    bins2 = 0.5 * (bins[:-1] + bins[1:])

    background_en3d = background_en3d.cpu().numpy()


    bin_total_proton, bin, p = plt.hist(background_en3d, bins)

    save_dict={}
    save_dict['bin_proton_rej'] = bin_proton_rej
    save_dict['background_en3d'] = background_en3d
    save_dict['bins2'] = bins2
    save_dict['bins'] = bins
    torch.save(save_dict, out_name+"rej_energy_better.pt")

    plt.figure()
    plt.plot(bins2, bin_proton_rej, label="ResNet", color="b")
    plt.hist(background_en3d, bins, label="Total Background per Bin", alpha=0.1, color="aqua", ec="black")

    plt.grid()
    plt.xlabel('True Energy[GeV]')
    plt.ylabel(y_label)
    plt.title(title)
    plt.yscale('log')
    plt.legend()
    plt.savefig(out_name+"_better_"+str(e_eff_title)+"_rej_energy_better.png", dpi=300)
    plt.clf()

def calc_signal_background(logits,labels,threshold,out_name):
    probs = torch.softmax(logits,dim=1)
    print("probs shape",probs[:,3].shape)
    index_muon = labels==1
    mis_cls_muon = probs[index_muon,0]>threshold
    print("mis_cls_muon shape",mis_cls_muon.shape)
    number_muon = index_muon.sum().item()
    number_mis_cls_muon = mis_cls_muon.sum().item()

    index_tau = labels==2
    mis_cls_tau = probs[index_tau,0]>threshold
    number_tau = index_tau.sum().item()
    number_mis_cls_tau = mis_cls_tau.sum().item()

    index_nc = labels==3
    mis_cls_nc = probs[index_nc,0]>threshold
    number_nc = index_nc.sum().item()
    number_mis_cls_nc = mis_cls_nc.sum().item()
    print("number of muon, tau, nc",number_muon,number_tau,number_nc)
    print("number of misclassified muon, tau, nc",number_mis_cls_muon,number_mis_cls_tau,number_mis_cls_nc)


    index_electron = labels==0
    correctly_cls_electron = probs[index_electron,0]>threshold
    number_electron = index_electron.sum().item()
    number_correctly_cls_electron = correctly_cls_electron.sum().item()
    print("number of electron, number of correctly classified electron",number_electron,number_correctly_cls_electron)

        # Example filename
    log_file = f"{out_name}.txt"

    with open(log_file, "w") as f:
        # Muon, tau, NC counts
        f.write(f"number of muon, tau, nc: {number_muon}, {number_tau}, {number_nc}\n")
        f.write(f"number of misclassified muon, tau, nc: {number_mis_cls_muon}, {number_mis_cls_tau}, {number_mis_cls_nc}\n")

        f.write(f"number of electron, number of correctly classified electron: {number_electron}, {number_correctly_cls_electron}\n")
        f.write(f"Efficiency of correctly classifying electron: {number_correctly_cls_electron / number_electron if number_electron > 0 else 0:.4f}\n")
        f.write(f"Efficiency of misclassifying muon as electron: {number_mis_cls_muon / number_muon if number_muon > 0 else 0:.4f}\n")
        f.write(f"Efficiency of misclassifying tau as electron: {number_mis_cls_tau / number_tau if number_tau > 0 else 0:.4f}\n")
        f.write(f"Efficiency of misclassifying NC as electron: {number_mis_cls_nc / number_nc if number_nc > 0 else 0:.4f}\n")

        n_background=number_mis_cls_muon + number_mis_cls_tau + number_mis_cls_nc

        if n_background==0:
            n_background=1
        n_sig_background = n_background + number_correctly_cls_electron

        s_to_b = number_correctly_cls_electron / n_background
        purity = number_correctly_cls_electron / n_sig_background
        f.write("signal to background ratio: " + str(s_to_b) + "\n")
        f.write("purity of signal: " + str(purity) + "\n")
        f.write("threshold: " + str(threshold) + "\n")

    return number_muon, number_mis_cls_muon, number_tau, number_mis_cls_tau, number_nc, number_mis_cls_nc

import torch

def calc_signal_background_def(logits, labels, out_name):
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)   # highest-probability class
    
    # Electron = 0, Muon = 1, Tau = 2, NC = 3
    index_muon = labels == 1
    mis_cls_muon = preds[index_muon] == 0
    number_muon = index_muon.sum().item()
    number_mis_cls_muon = mis_cls_muon.sum().item()

    index_tau = labels == 2
    mis_cls_tau = preds[index_tau] == 0
    number_tau = index_tau.sum().item()
    number_mis_cls_tau = mis_cls_tau.sum().item()

    index_nc = labels == 3
    mis_cls_nc = preds[index_nc] == 0
    number_nc = index_nc.sum().item()
    number_mis_cls_nc = mis_cls_nc.sum().item()

    index_electron = labels == 0
    correctly_cls_electron = preds[index_electron] == 0
    number_electron = index_electron.sum().item()
    number_correctly_cls_electron = correctly_cls_electron.sum().item()

    print("number of muon, tau, nc:", number_muon, number_tau, number_nc)
    print("number of misclassified muon, tau, nc:", number_mis_cls_muon, number_mis_cls_tau, number_mis_cls_nc)
    print("number of electron, correctly classified electron:", number_electron, number_correctly_cls_electron)

    log_file = f"{out_name}_def.txt"
    with open(log_file, "w") as f:
        f.write(f"number of muon, tau, nc: {number_muon}, {number_tau}, {number_nc}\n")
        f.write(f"number of misclassified muon, tau, nc: {number_mis_cls_muon}, {number_mis_cls_tau}, {number_mis_cls_nc}\n")
        f.write(f"number of electron, correctly classified electron: {number_electron}, {number_correctly_cls_electron}\n")
        f.write(f"Efficiency of correctly classifying electron: {number_correctly_cls_electron / number_electron if number_electron > 0 else 0:.4f}\n")
        f.write(f"Efficiency of misclassifying muon as electron: {number_mis_cls_muon / number_muon if number_muon > 0 else 0:.4f}\n")
        f.write(f"Efficiency of misclassifying tau as electron: {number_mis_cls_tau / number_tau if number_tau > 0 else 0:.4f}\n")
        f.write(f"Efficiency of misclassifying NC as electron: {number_mis_cls_nc / number_nc if number_nc > 0 else 0:.4f}\n")

        s_to_b = number_correctly_cls_electron / (number_mis_cls_muon + number_mis_cls_tau + number_mis_cls_nc + 1e-9)
        purity = number_correctly_cls_electron / (number_correctly_cls_electron + number_mis_cls_muon + number_mis_cls_tau + number_mis_cls_nc + 1e-9)
        f.write("signal to background ratio: " + str(s_to_b) + "\n")
        f.write("purity of signal: " + str(purity) + "\n")
        f.write("classification rule: argmax(prob)\n")

    return number_muon, number_mis_cls_muon, number_tau, number_mis_cls_tau, number_nc, number_mis_cls_nc


def plot_purities(logits,labels,out_name,title2=" "):

    probs = torch.softmax(logits,dim=1)

    thresholds = np.load("/eos/user/b/beturk/snd/dl/cvt_threshold01.npy")

    list_purity=[]
    list_s_to_b = []
    list_efficiency = []
    list_signal=[]

    for threshold in thresholds:
        index_muon = labels==1
        mis_cls_muon = probs[index_muon,0]>threshold
        number_muon = index_muon.sum().item()
        number_mis_cls_muon = mis_cls_muon.sum().item()

        index_tau = labels==2
        mis_cls_tau = probs[index_tau,0]>threshold
        number_tau = index_tau.sum().item()
        number_mis_cls_tau = mis_cls_tau.sum().item()

        index_nc = labels==3
        mis_cls_nc = probs[index_nc,0]>threshold
        number_nc = index_nc.sum().item()
        number_mis_cls_nc = mis_cls_nc.sum().item()

        index_electron = labels==0
        correctly_cls_electron = probs[index_electron,0]>threshold
        number_electron = index_electron.sum().item()
        number_correctly_cls_electron = correctly_cls_electron.sum().item()

        total_background = number_mis_cls_muon + number_mis_cls_tau + number_mis_cls_nc
        if total_background == 0:
            total_background = 1  # Prevent division by zero
        

        s_to_b = number_correctly_cls_electron / (total_background)
        purity = number_correctly_cls_electron / (number_correctly_cls_electron + total_background)

        list_purity.append(purity)
        list_s_to_b.append(s_to_b)
        list_efficiency.append(number_correctly_cls_electron/number_electron)
        list_signal.append(number_correctly_cls_electron)
    

    import matplotlib.pyplot as plt

    # Purity
    plt.figure()
    plt.plot(list_efficiency, list_purity, marker='o')
    plt.title(f"Purity vs. Signal Efficiency {title2}")
    plt.xlabel("Signal Efficiency")
    plt.ylabel("Purity")
    plt.savefig(out_name+"_purity_vs_efficiency.png", dpi=300)
    plt.grid(True)
    plt.clf()
    # Signal-to-background ratio
    plt.figure()
    plt.plot(list_efficiency, list_s_to_b, marker='o', color="orange")
    plt.title(f"Signal-to-Background Ratio vs. Signal Efficiency {title2}")
    plt.xlabel("Signal Efficiency")
    plt.ylabel("Signal-to-Background Ratio")
    plt.grid(True)
    plt.savefig(out_name+"_s_to_b_vs_efficiency.png", dpi=300)
    plt.clf()

    plt.figure()
    plt.plot(list_signal, list_purity, marker='o', color="green")
    plt.title(f"Purity vs. Number of Signal Events {title2}")
    plt.xlabel("Number of Signal Events")
    plt.ylabel("Purity")
    plt.grid(True)
    plt.savefig(out_name+"_purity_vs_number_signal.png", dpi=300)
    plt.clf()

    plt.figure()
    plt.plot(list_signal, list_s_to_b, marker='o', color="red")
    plt.title(f"Signal-to-Background Ratio vs. Number of Signal Events {title2}")
    plt.xlabel("Number of Signal Events")
    plt.ylabel("Signal-to-Background Ratio")
    plt.grid(True)
    plt.savefig(out_name+"_s_to_b_vs_number_signal.png", dpi=300)
    plt.clf()   

    plt.figure()
    plt.plot(list_signal, list_efficiency, marker='o', color="purple")
    plt.title(f"Signal Efficiency vs. Number of Signal Events {title2}")
    plt.xlabel("Number of Signal Events")     
    plt.ylabel("Signal Efficiency")
    plt.grid(True)
    plt.savefig(out_name+"_efficiency_vs_number_signal.png", dpi=300)
    plt.clf()

    return number_muon, number_mis_cls_muon, number_tau, number_mis_cls_tau, number_nc, number_mis_cls_nc







