import os
import glob
import torch
import uproot
import argparse
import pandas as pd
import numpy as np

def extract_dl_data(pt_file_path):
    # DL modellerinden çıkan sonuç sözlüğünü yükle
    data = torch.load(pt_file_path, weights_only=False)
    
    # Dosya isminden modelin adını çıkaralım ki sütun isimleri çakışmasın
    filename = os.path.basename(pt_file_path).replace(".pt", "")
    model_id = filename.replace("event_list_batch_", "").replace("energy_recon_", "")
    
    # run_id ve event_number değerlerini güvenli bir şekilde numpy array olarak al
    run_ids = data.get("run_id", [])
    if isinstance(run_ids, torch.Tensor): run_ids = run_ids.numpy()
    else: run_ids = np.array(run_ids)
        
    event_nums = data.get("event_number", [])
    if isinstance(event_nums, torch.Tensor): event_nums = event_nums.numpy()
    else: event_nums = np.array(event_nums)
    
    if len(run_ids) == 0:
        return []

    # Model çıktıları olan özel anahtarları bul (Gereksizleri ve hit/qdc verilerini yoksay)
    ignore_keys = {"run_id", "event_number", "idx", "path_of_model"}
    prediction_keys = [
        k for k in data.keys() 
        if k not in ignore_keys and "Hits" not in k and "QDC" not in k and "Log" not in k
    ]
    
    # Eğer prediction key yoksa, muhtemelen bu orjinal input event_list_batch.pt dosyasıdır.
    # İşleme alma ve atla (Bu sayede .dim() hatasından kurtuluyoruz)
    if not prediction_keys:
        return []

    extracted_records = []
    for i in range(len(run_ids)):
        record = {
            "run_id": int(run_ids[i]),
            "event_number": int(event_nums[i]),
        }
        
        # Tahminleri ekle (Sınıflandırma veya Enerji)
        for key in prediction_keys:
            val_sequence = data[key]
            
            # Değerin bir tensor mu yoksa standart liste mi olduğunu kontrol et
            if isinstance(val_sequence, torch.Tensor):
                val = val_sequence[i].item() if val_sequence.dim() > 0 else val_sequence.item()
            else:
                val = val_sequence[i]
            
            # SÜTUN İSİMLENDİRMESİ: 3 farklı modelin aynı isimdeki verileri birbirini 
            # ezmesin diye kolonun başına model adını ekliyoruz.
            column_name = f"{model_id}_{key}"
            record[column_name] = val
            
        extracted_records.append(record)
        
    return extracted_records

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Path to directory containing .pt files")
    parser.add_argument("--out_txt", type=str, default="final_results.txt")
    parser.add_argument("--out_root", type=str, default="final_results.root")
    args = parser.parse_args()

    # Klasördeki tüm .pt dosyalarını bul
    processed_files = glob.glob(os.path.join(args.dir, "*.pt"))
    if not processed_files:
        print(f"[WARNING] No PT files found in {args.dir}")
        exit(1)

    all_data = []
    for f in processed_files:
        try:
            records = extract_dl_data(f)
            if records:
                all_data.extend(records)
                print(f"[INFO] Successfully processed DL outputs from {os.path.basename(f)}")
            else:
                print(f"[INFO] Skipped {os.path.basename(f)} (Input file, no DL predictions found inside)")
        except Exception as e:
            print(f"[ERROR] Could not read {f}: {e}")

    if not all_data:
        print("[ERROR] No valid prediction data could be extracted from any files.")
        exit(1)

    # Bütün veriyi tek bir tabloya çevir
    df = pd.DataFrame(all_data)
    
    # run_id ve event_number aynı olan satırları (3 modelin de aynı event'i tahmin ettiği satırları)
    # birleştirerek tek bir satır haline getiriyoruz. (Eksik verileri NaN bırakmadan birbirini tamamlar)
    grouped_df = df.groupby(["run_id", "event_number"], as_index=False).first()

    # 1. TXT olarak kaydet (Tab ile ayrılmış değerler)
    grouped_df.to_csv(args.out_txt, sep='\t', index=False)
    print(f"\n[SUCCESS] Saved merged text results to: {args.out_txt}")

    # 2. ROOT olarak kaydet (Awkward Array hatası giderildi)
    root_dict = {}
    for col in grouped_df.columns:
        # Pandas Object tiplerini (Metin vb.) Uproot desteklemez, listeye çeviriyoruz.
        if grouped_df[col].dtype == 'O': 
            root_dict[col] = grouped_df[col].astype(str).to_list()
        else:
            root_dict[col] = grouped_df[col].to_numpy()

    with uproot.recreate(args.out_root) as root_file:
        root_file["dl_results"] = root_dict
    
    print(f"[SUCCESS] Saved merged ROOT results to: {args.out_root}\n")