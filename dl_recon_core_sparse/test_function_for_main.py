import os
import torch
import matplotlib.pyplot as plt
from dl_recon_core_sparse.test_functions_cls import *
from dl_recon_core_sparse.test_functions_energy import *
from dl_recon_core_sparse.data_loader import *
import shutil

def save_all_probs(all_tensors, feature_extractor, classifier, out_name, device):
    feature_extractor.eval()
    classifier.eval()
    
    print("\nRUNNING save_all_probs")
    test_loader = DataLoader(all_tensors, batch_size=config.BATCH_SIZE_TEST, shuffle=False)

    logits = []
    targets = []
    energy_list_check = []

    # Sadece enerji rekonstrüksiyonu için kullanılacak listeler
    linear_terms = []
    dl_terms = []

    # Linear term için scifi_param parametresini güvenli şekilde al (Sadece Enerji modunda)
    scifi_p = 0.0
    if config.IS_ENERGY_RECON:
        try:
            scifi_p = feature_extractor.module.scifi_param.item()
        except AttributeError:
            try:
                scifi_p = feature_extractor.scifi_param.item()
            except:
                scifi_p = 0.0 # Parametre yoksa veya Classification modeliyse 0 kalsın
    print("scifi parameter" ,scifi_p)
    # --- 3. INFERENCE LOOP ---
    print("Starting inference loop...")
    with torch.no_grad():
        for k, data in enumerate(test_loader):
            # Unpack data based on config
            if config.USE_ONLY_SCIFI:
                scifi, energy, labels = data
                x = scifi.to(device)
                scifi_tensor = scifi # Lineer hesaplama için cpu'ya almadan önceki hali
                energy, labels = energy.to(device), labels.to(device)
                
            elif config.USE_SCIFI_US:
                scifi, us, energy, labels = data
                scifi, us = scifi.to(device), us.to(device)
                energy, labels = energy.to(device), labels.to(device)
                x = (scifi, us)
                scifi_tensor = scifi
                
            elif config.USE_SCIFI_US_DS:
                scifi, us, ds, energy, labels = data
                scifi, us, ds = scifi.to(device), us.to(device), ds.to(device)
                energy, labels = energy.to(device), labels.to(device)
                x = (scifi, us, ds)
                scifi_tensor = scifi

            if config.IS_SINGLE_NETWORK:
                output = feature_extractor(x)
            else:
                output = classifier(feature_extractor(x))

            # Move to CPU immediately to save GPU memory
            output = output.cpu()
            labels = labels.cpu()
            energy = energy.cpu()

            logits.append(output)
            targets.append(labels)
            energy_list_check.append(energy)

            # Sadece enerji rekonstrüksiyonunda Linear ve DL terimlerini ayır
            if config.IS_ENERGY_RECON:
                scifi_tensor_cpu = scifi_tensor.cpu()
                lin_term = scifi_p * torch.sum(scifi_tensor_cpu, (1, 2, 3))
                dl_term = output - lin_term
                linear_terms.append(lin_term)
                dl_terms.append(dl_term)
        
    logits = torch.cat(logits, dim=0)
    targets = torch.cat(targets, dim=0)
    energy_list_check = torch.cat(energy_list_check, dim=0)
    
    if config.IS_ENERGY_RECON:
        linear_terms = torch.cat(linear_terms, dim=0)
        dl_terms = torch.cat(dl_terms, dim=0)

    # --- 4. SAVE AND VERIFY ---
    dataset_energies = torch.stack(all_tensors.energies).cpu()
    dataset_labels = all_tensors.labels.cpu()

    # Verification: Check if the order was preserved
    is_energy_ok = torch.equal(energy_list_check, dataset_energies)
    is_target_ok = torch.equal(targets, dataset_labels)

    IS_LOG_NORM_OUTPUT = getattr(config, "IS_LOG_NORM_OUTPUT", False)
    if IS_LOG_NORM_OUTPUT:
        logits= torch.exp(logits)

    save_tensor = {}
    save_tensor['new_model'] = logits
    save_tensor["en3d"] = energy_list_check
    save_tensor["y"] = targets 
    
    if config.IS_ENERGY_RECON:
        save_tensor["linear_term"] = linear_terms
        save_tensor["dl_term"] = dl_terms

    if is_energy_ok and is_target_ok:
        print("Saving prob data...")
        torch.save(save_tensor, out_name)
        print(f"Saved to {out_name}")
        print("Logits shape:", logits.shape)
        
        # Debug prints
        for key in save_tensor:
            if hasattr(save_tensor[key], 'shape'):
                print("key in saved file:", key, save_tensor[key].shape)
        
        return save_tensor
    else:
        print("\nCRITICAL ERROR: Data mismatch!")
        print(f"Energy match: {is_energy_ok}")
        print(f"Target match: {is_target_ok}")
        print("Expected shapes:", dataset_energies.shape, dataset_labels.shape)
        print("Actual shapes:  ", energy_list_check.shape, targets.shape)
        print("Exiting to prevent corrupt data save.")
        exit()


def run_test_for_energy(config, eos, model_ecal, classifier, device, name_only,datalist,test_name="define particle type name",signal_index = None, is_neutrino=False,particle_name=" ",BEAM_OR_TRUE_ENERGY="True",DATA_TYPE="MC",dict="None", EN_MIN=config.EN_MIN,EN_MAX=config.EN_MAX):
    datasets=[]
    all_configs_metrics = []
    for test_data_name in datalist:
        datasets.append(SNDSparseDataset(test_data_name, perc=config.TOTAL_TEST_SIZE, EN_MIN=EN_MIN, EN_MAX=EN_MAX ))

    if DATA_TYPE in ["PG_MC", "TB_MC"]:
        t_window_list = dict["t_window_mc"]
        qdc_threshold_value_scifi_list = dict["qdc_threshold_value_scifi_mc"]
        TB_RECALIBRATION_S2Y_list=["MC"]

    elif DATA_TYPE=="TB_Data":
        TB_RECALIBRATION_S2Y_list = dict["TB_RECALIBRATION_S2Y"]
        t_window_list = dict["t_window_data"]
        qdc_threshold_value_scifi_list =  dict["qdc_threshold_value_scifi_data"]

    for TB_RECALIBRATION_S2Y in TB_RECALIBRATION_S2Y_list:
        for qdc_threshold_value_scifi in qdc_threshold_value_scifi_list:
            for t_window in t_window_list:
                test_data = {}
                if BEAM_OR_TRUE_ENERGY == "Beam":
                    beam_energy_list = []
                    recon_en_list = []
                
                if DATA_TYPE == "TB_Data":
                    cut_dir_name = f"S2Ycal{TB_RECALIBRATION_S2Y}_qdcthredata{qdc_threshold_value_scifi}_twindata{t_window[0]}{t_window[1]}"
                else: 
                    cut_dir_name = f"MC_qdcthreMC{qdc_threshold_value_scifi}_twinMC{t_window[0]}{t_window[1]}"
 
                common_dir = f"{eos}/{cut_dir_name}/tests_{test_name}"
                os.makedirs(common_dir, exist_ok=True)
                common_out_name = f"{common_dir}/{test_name}_{name_only}_"


                for i,test_data_name in enumerate(datalist):
                    ith_dataset=datasets[i]
                    test_data_name1 = os.path.splitext(os.path.basename(test_data_name[1]))[0]
                    if "MC" in test_data_name[1]:
                        out_name = f"{eos}/probs_{test_data_name1}_{name_only}_{cut_dir_name}.pt"
                        ith_dataset.update_hit_cuts( 
                        t_window_high_mc=t_window[1], t_window_low_mc=t_window[0], qdc_thresh_mc=qdc_threshold_value_scifi,
                        )

                    else:
                        out_name = f"{eos}/probs_{test_data_name1}_{name_only}_{cut_dir_name}.pt"
                        ith_dataset.update_hit_cuts( 
                        t_window_high_data=t_window[1], t_window_low_data=t_window[0], qdc_thresh_data=qdc_threshold_value_scifi,
                        TB_RECALIBRATION_S2Y=TB_RECALIBRATION_S2Y)


                    if os.path.isfile(out_name):
                        ithtest_data = torch.load(out_name)
                        print("Loaded test data, found as", out_name)

                    else:
                        print("No saved probs found, creating one:", out_name)
                        ithtest_data = save_all_probs(ith_dataset, model_ecal, classifier, out_name, device)
                    # Concatenate across datasets
                    os.makedirs(common_out_name+test_data_name1+f"/{test_data_name1}", exist_ok=True)

                    excell_dirs = (eos,cut_dir_name)

                    plot_res_energy(ithtest_data["new_model"],ithtest_data["en3d"],common_out_name+test_data_name1+f"/{test_data_name1}",particle_name, BEAM_OR_TRUE_ENERGY, excell_dirs=excell_dirs)
                    #test_model_params_hist(ithtest_data["scifi_sig"], ithtest_data["new_model"], ithtest_data["en3d"], test_data_name1 ,eos,common_out_name)
                    if BEAM_OR_TRUE_ENERGY=="Beam":
                        beam_energy_list.append(ithtest_data["en3d"][0])
                        recon_en_list.append(ithtest_data["new_model"])
                    for key in ithtest_data:
                        print(key, ithtest_data[key].shape)
                        if key in test_data:
                            test_data[key] = torch.cat((test_data[key], ithtest_data[key]))
                        else:
                            test_data[key] = ithtest_data[key]
                    print(test_data[key].shape)

                    print("Added to test_data dict.\n")


                    if BEAM_OR_TRUE_ENERGY=="Beam":
                        plot_1d_beam_energy_graphs(beam_energy_list, recon_en_list,"MEAN_STD_OF_RECON_ENERGY" ,xlabel="Beam Energy [GeV]",ylabel='Average Recon. Energy[GeV]',title="Average Recon. vs True Energy",outdir=common_dir,show_ideal=True)

                # === Extract data ===
                if signal_index != None:
                    index = test_data["y"]==signal_index
                    res_array = plot_res_energy(test_data["new_model"][index],test_data["en3d"][index],common_out_name+"all",particle_name, BEAM_OR_TRUE_ENERGY)
                else:
                    res_array = plot_res_energy(test_data["new_model"],test_data["en3d"],common_out_name+"all",particle_name, BEAM_OR_TRUE_ENERGY)
                
                if "linear_term" in ithtest_data and "dl_term" in ithtest_data:
                    # 1. Lineer Terim vs True Energy
                    print("en3d", test_data["en3d"])
                    plot_simple_plots(y_tensor=test_data["linear_term"], 
                                        x_tensor=test_data["en3d"], 
                                        y_label=r"$E_{Linear Term}$ [GeV]", 
                                        x_label=r"$E_{True}$ [GeV]", 
                                        title=f"Linear Term vs True Energy\n({test_data_name1})", 
                                        plot_save_name=common_out_name+"Linear_vs_True")
                    # 2. DL (Deep Learning) Terimi vs True Energy
                    plot_simple_plots(y_tensor=test_data["dl_term"], 
                                        x_tensor=test_data["en3d"], 
                                        y_label=r"$E_{DL Term}$ [GeV]", 
                                        x_label=r"$E_{True}$ [GeV]", 
                                        title=f"Deep Learning Term vs True Energy\n({test_data_name1})", 
                                        plot_save_name=common_out_name+"DL_vs_True")
                    plot_simple_plots(y_tensor=test_data["dl_term"], 
                                        x_tensor=test_data["linear_term"], 
                                        y_label=r"$E_{DL Term}$ [GeV]", 
                                        x_label=r"$E_{Linear Term}$ [GeV]", 
                                        title=f"Deep Learning Term vs Linear Term\n({test_data_name1})", 
                                        plot_save_name=common_out_name+"DL_vs_Linear")
                

                # Store the result of this specific parameter combination
                all_configs_metrics.append({
                    "cut_dir_name": cut_dir_name,
                    "res_array": res_array
                })

    # OUTSIDE ALL LOOPS: Return the collected metrics
    return all_configs_metrics




def run_test_for_cls(config, eos, model_ecal, classifier, device, name_only,datalist,dict, bins,test_name="define particle type name", signal_index = 4, is_neutrino=False):
    all_configs_metrics = []
    datasets=[]
    for test_data_name in datalist:
        datasets.append(SNDSparseDataset(test_data_name,perc=config.TOTAL_TEST_SIZE))

    for TB_RECALIBRATION_S2Y in dict["TB_RECALIBRATION_S2Y"]:
        for qdc_threshold_value_scifi_data in dict["qdc_threshold_value_scifi_data"]:
            for qdc_threshold_value_scifi_mc in dict["qdc_threshold_value_scifi_mc"]:
                for t_window_data in dict["t_window_data"]:
                    for t_window_mc in dict["t_window_mc"]:
                        cut_dir_data_name=f"S2Ycal{TB_RECALIBRATION_S2Y}_qdcthredata{qdc_threshold_value_scifi_data}_twindata{t_window_data[0]}{t_window_data[1]}"
                        cut_dir_MC_name=f"qdcthremc{qdc_threshold_value_scifi_mc}_twinmc{t_window_mc[0]}{t_window_mc[1]}"

                        common_dir = f"{eos}/{cut_dir_data_name}_{cut_dir_MC_name}/tests_{test_name}"
                        os.makedirs(common_dir, exist_ok=True)
                        common_out_name = f"{common_dir}/{test_name}_{name_only}_"

                        test_data = {}

                        for i,test_data_name in enumerate(datalist):
                            ith_dataset=datasets[i]
                            test_data_name1 = os.path.splitext(os.path.basename(test_data_name[1]))[0]
                            if "MC" in test_data_name[1]:
                                out_name = f"{eos}/probs_{test_data_name1}_{name_only}_{cut_dir_MC_name}.pt"
                                ith_dataset.update_hit_cuts( 
                                t_window_high_mc=t_window_mc[1], t_window_low_mc=t_window_mc[0], qdc_thresh_mc=qdc_threshold_value_scifi_mc,
                                )

                            else:
                                out_name = f"{eos}/probs_{test_data_name1}_{name_only}_{cut_dir_data_name}.pt"
                                ith_dataset.update_hit_cuts( 
                                t_window_high_data=t_window_data[1], t_window_low_data=t_window_data[0], qdc_thresh_data=qdc_threshold_value_scifi_data,
                                TB_RECALIBRATION_S2Y=TB_RECALIBRATION_S2Y)


                            if os.path.isfile(out_name):
                                ithtest_data = torch.load(out_name)
                                print("Loaded test data, found as", out_name)

                            else:
                                print("No saved probs found, creating one:", out_name)
                                ithtest_data = save_all_probs(ith_dataset, model_ecal, classifier, out_name, device)

                            for key in ithtest_data:
                                print(key, ithtest_data[key].shape)
                                if key in test_data:
                                    test_data[key] = torch.cat((test_data[key], ithtest_data[key]))
                                else:
                                    test_data[key] = ithtest_data[key]

                            print("Added to test_data dict.\n")

                        #index = test_data["y"]==0

                        # === Extract data ===
                        logits = test_data["new_model"]#[index]
                        targets = test_data["y"]#[index]
                        energies = test_data["en3d"]#[index]

                        # NEW: Calculate binary loss and accuracy for this specific config
                        binary_targets = (targets == signal_index).float()
                        if len(logits.shape) > 1 and logits.shape[1] > 1:
                            probs = F.softmax(logits, dim=1)[:, signal_index]
                        else:
                            probs = torch.sigmoid(logits.squeeze())

                        
                        bce_loss = F.binary_cross_entropy(probs, binary_targets).item()
                        #binary_preds = (probs > 0.5).float()
                        #bin_acc = (binary_preds == binary_targets).float().mean().item()

                        all_configs_metrics.append({
                            'config_name': f"{cut_dir_data_name}_{cut_dir_MC_name}",
                            'common_dir': common_dir,
                            'loss': bce_loss
                        })

                        print("Loaded all data.")
                        with open(common_out_name+"tf_metrics.txt", "w") as f:
                            for i in range(10):
                                print(i, torch.sum(targets==i))
                                precision, recall, f1, accuracy = binary_metrics(targets,logits,i)
                            
                                f.write(f"for signal index i, {i}\n")
                                f.write(f"Precision: {precision:.4f}\n")
                                f.write(f"Recall: {recall:.4f}\n")
                                f.write(f"F1 Score: {f1:.4f}\n")
                                f.write(f"Accuracy: {accuracy:.4f}\n \n")
                        print(common_out_name)
                        # === Convert to binary probabilities ===
                        print(logits)
                        print(targets)
                        signal_probs, background_probs, background_en3d = convert_2_binary_probs(logits, targets, energies, signal_index)

                        if config.IS_BINARY:

                            plt.hist(signal_probs.cpu().numpy(), bins=30, alpha=0.5, label="Signal")
                            plt.hist(background_probs.cpu().numpy(), bins=30, alpha=0.5, label="Background")
                            plt.xlabel("Model's Logit Value")
                            plt.ylabel("Counts")
                            plt.title("Signal vs Background Logit Distribution")
                            plt.legend()
                            plt.yscale("log")
                            plt.savefig(common_out_name + "norm.png", dpi=300)
                            plt.clf()
                            print("shape of model", background_probs.shape,signal_probs.shape)
                            print("max and min of logits",background_probs.max().item(), signal_probs.min().item())
                            # return 0
                        else:
                            plot_confusion_matrix(logits, targets,
                                                out_name=common_out_name + "conf_matrix.png",
                                                class_names=config.CLASS_NAMES_CONF_MATRIX,GET_PARTICLES_WITH_INDEX=config.GET_PARTICLES_WITH_INDEX)
                            plot_confusion_matrix(logits, targets,
                                                out_name=common_out_name + "conf_matrix_larger.png",
                                                class_names=config.CLASS_NAMES_CONF_MATRIX_LARGER,GET_PARTICLES_WITH_INDEX=config.GET_PARTICLES_WITH_INDEX_LARGER)

                        # === Global efficiency and rejection ===
                        thresholds, signal_eff, background_rej = s_eff_b_rej(signal_probs, background_probs,
                                                                            common_out_name + str(signal_index))
                        required_eff = config.REQUIRED_EFF
                        threshold, _, global_rejection = find_threshold(thresholds, signal_eff, background_rej, required_eff)
                        p_rej_energy(threshold, background_probs, background_en3d,
                                    common_out_name + str(signal_index), str(int(required_eff * 100)),bins ,True)

                        #_, _, global_rejection = find_threshold(thresholds, signal_eff, background_rej, 0.5)
                        threshold, _, global_rejection = find_threshold(thresholds, signal_eff, background_rej, 0.8)
                        all_configs_metrics[-1]['rejection'] = global_rejection
                        print("rejection",global_rejection)
                        if is_neutrino:
                            calc_signal_background(logits,targets,threshold,common_out_name)
                            calc_signal_background_def(logits,targets,common_out_name)
                            plot_purities(logits,targets,common_out_name)
                        """

                        # === Energy binning ===
                        better_rej_at_given_eff = []

                        for i in range(len(bins) - 1):
                            en1, en2 = bins[i], bins[i + 1]
                            print(f"\nEnergy range: {en1}-{en2} GeV")

                            common_dir = f"{eos}/tests_{test_name}/energy_bins/en_{en1}_{en2}/"
                            os.makedirs(common_dir, exist_ok=True)
                            common_out_name = f"{common_dir}/{test_name}_{name_only}_"

                            cut = (energies > en1) & (energies < en2)
                            title2 = f"({en1}-{en2} GeV)"

                            print("Electron number:", torch.sum(targets[cut] == 4))
                            print("Hadron number:", torch.sum(targets[cut] == 5), "\n")
                            print("NC number:", torch.sum(targets[cut] == 3), "\n")

                            signal_probs, background_probs, _ = convert_2_binary_probs(
                                logits[cut], targets[cut], energies[cut], signal_index
                            )

                            if (len(signal_probs) > 5) and (len(background_probs) > 5):
                                if config.IS_BINARY:
                                    plt.hist(signal_probs.cpu().numpy(), bins=np.linspace(0, 1, 20), alpha=0.5, label="Signal")
                                    plt.hist(background_probs.cpu().numpy(), bins=np.linspace(0, 1, 20), alpha=0.5, label="Background")
                                    plt.xlabel("Model's Logit Value")
                                    plt.ylabel("Counts")
                                    plt.title("Signal vs Background Logit Distribution")
                                    plt.legend()
                                    plt.yscale("log")
                                    plt.savefig(common_out_name + ".png", dpi=300)
                                    plt.clf()
                                else:
                                    plot_confusion_matrix(logits[cut], targets[cut],
                                                        out_name=common_out_name + "conf_matrix.png",
                                                        class_names=config.CLASS_NAMES_CONF_MATRIX,
                                                        title2=title2)

                                thresholds, signal_eff, background_rej = s_eff_b_rej(
                                    signal_probs, background_probs,
                                    common_out_name + str(signal_index), title2
                                )
                                threshold, _, rejection_at_given_eff = find_threshold(
                                    thresholds, signal_eff, background_rej, required_eff
                                )
                                better_rej_at_given_eff.append(rejection_at_given_eff)

                                if is_neutrino:
                                    calc_signal_background(logits,targets,threshold,common_out_name)
                                    calc_signal_background_def(logits,targets,common_out_name)
                                    plot_purities(logits,targets,common_out_name)
                            else:
                                better_rej_at_given_eff.append(0)

                        # === Final combined rejection plot ===
                        common_dir = f"{eos}/tests_{test_name}"
                        common_out_name = f"{eos}/tests_{test_name}/{test_name}_{name_only}_"
                        single_better_p_rej_energy(better_rej_at_given_eff, background_en3d,
                                                common_out_name + str(signal_index),
                                                str(int(required_eff * 100)), bins)"""

    print("All-particle testing completed successfully.")

    # NEW: Sorting, Logging, and Moving the Best 5 Configurations
    # Sort from best to worst (Priority 1: Highest Accuracy, Priority 2: Lowest Loss)
    all_configs_metrics.sort(key=lambda x: (-x['rejection'], x['loss']))

    # Save to txt file
    summary_txt_path = f"{eos}/best_configs_summary_{test_name}_{name_only}.txt"
    with open(summary_txt_path, "w") as f:
        f.write("Rank | rejection | Loss | Config Name | Directory\n")
        f.write("-" * 90 + "\n")
        for rank, metric in enumerate(all_configs_metrics):
            f.write(f"{rank+1} | {metric['rejection']} | {metric['loss']:.6f} | {metric['config_name']} | {metric['common_dir']}\n")

    # Move top 5 to best_ones directory
    best_ones_dir = f"{eos}/best_ones_{test_name}_{name_only}"
    os.makedirs(best_ones_dir, exist_ok=True)

    for i, metric in enumerate(all_configs_metrics[:5]):
        src_dir = metric['common_dir']
        if os.path.exists(src_dir):
            # Append rank to folder name to easily identify top performers
            dest_dir = os.path.join(best_ones_dir, f"rank_{i+1}_{metric['config_name']}")
            shutil.move(src_dir, dest_dir)
            print(f"Moved Rank {i+1} ({metric['config_name']}) to {dest_dir}")