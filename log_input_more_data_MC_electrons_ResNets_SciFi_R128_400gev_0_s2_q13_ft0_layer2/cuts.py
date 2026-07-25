noise_sigma=2
q_max=13
qdc_threshold_value_scifi_mc=-0.5 ## en başta dataloaderda uygulanan cut. aşağıdakiler ise dataloaderdan çıktıktan sonra uygulanıyor

qdc_threshold_value_scifi_all_same_final_mc=0
qdc_threshold_value_scifi_data=qdc_threshold_value_scifi_all_same_final_mc

t_window_high_end_data = 2.3
t_window_low_end_data = 0.5 #use positive always

t_window_high_end_mc = 1
t_window_low_end_mc = 1

time_window_max_us_ds_data=3
time_window_min_us_ds_data=3

cut_min_scifi_hitx_in_64r = 1
cut_min_scifi_hity_in_64r = 1

ONLY_POS_QDC = False
BINARY_QDC_VALUES = False        # If True, sets all QDC > 0 to 1.0
# ==========================================
# EVENT-LEVEL CUTS FOR DATALOADER
# ==========================================

# ------------------------------------------
# 1. SCIFI CUTS (5 Layers)
# ------------------------------------------
"""# --- Total Hits ---
cut_min_scifi_notime_total_hits = 0
cut_max_scifi_notime_total_hits = float('inf')
cut_min_scifi_05usualtime_total_hits = 0
cut_max_scifi_05usualtime_total_hits = float('inf')

# --- Total QDC ---
cut_min_scifi_notime_total_qdc = 0.0
cut_max_scifi_notime_total_qdc = float('inf')
cut_min_scifi_05usualtime_total_qdc = 0.0
cut_max_scifi_05usualtime_total_qdc = float('inf')

# --- Hits Per Layer (Requires list of 5) ---
cut_min_scifi_notime_hits_per_layer = [0, 0, 0, 0, 0]
cut_max_scifi_notime_hits_per_layer = [float('inf')] * 5
cut_min_scifi_05usualtime_hits_per_layer = [0, 0, 0, 0, 0]
cut_max_scifi_05usualtime_hits_per_layer = [float('inf')] * 5

# --- QDC Per Layer (Requires list of 5) ---
cut_min_scifi_notime_qdc_per_layer = [0.0, 0.0, 0.0, 0.0, 0.0]
cut_max_scifi_notime_qdc_per_layer = [float('inf')] * 5
cut_min_scifi_05usualtime_qdc_per_layer = [0.0, 0.0, 0.0, 0.0, 0.0]
cut_max_scifi_05usualtime_qdc_per_layer = [float('inf'),float('inf'),float('inf'),float('inf'),float('inf')] 


# ------------------------------------------
# 2. UPSTREAM CUTS (US) (5 Layers)
# ------------------------------------------
# --- Total Hits ---
cut_min_us_notime_total_hits = 0
cut_max_us_notime_total_hits = float('inf')
cut_min_us_3usualtime_total_hits = 0
cut_max_us_3usualtime_total_hits = float('inf')

# --- Total QDC ---
cut_min_us_notime_total_qdc = 0.0
cut_max_us_notime_total_qdc = float('inf')
cut_min_us_3usualtime_total_qdc = 0.0
cut_max_us_3usualtime_total_qdc = float('inf')

# --- Hits Per Layer (Requires list of 5) ---
cut_min_us_notime_hits_per_layer = [0, 0, 0, 0, 0]
cut_max_us_notime_hits_per_layer = [float('inf')] * 5
cut_min_us_3usualtime_hits_per_layer = [0, 0, 0, 0, 0]
cut_max_us_3usualtime_hits_per_layer = [float('inf')] * 5

# --- QDC Per Layer (Requires list of 5) ---
cut_min_us_notime_qdc_per_layer = [0.0, 0.0, 0.0, 0.0, 0.0]
cut_max_us_notime_qdc_per_layer = [float('inf')] * 5
cut_min_us_3usualtime_qdc_per_layer = [0.0, 0.0, 0.0, 0.0, 0.0]
cut_max_us_3usualtime_qdc_per_layer = [float('inf')] * 5


# ------------------------------------------
# 3. DOWNSTREAM HORIZONTAL CUTS (DS-H) (3 Layers)
# ------------------------------------------
# --- Total Hits ---
cut_min_dsh_notime_total_hits = 0
cut_max_dsh_notime_total_hits = float('inf')
cut_min_dsh_3usualtime_total_hits = 0
cut_max_dsh_3usualtime_total_hits = float('inf')

# --- Total QDC ---
cut_min_dsh_notime_total_qdc = 0.0
cut_max_dsh_notime_total_qdc = float('inf')
cut_min_dsh_3usualtime_total_qdc = 0.0
cut_max_dsh_3usualtime_total_qdc = float('inf')

# --- Hits Per Layer (Requires list of 3) ---
cut_min_dsh_notime_hits_per_layer = [0, 0, 0]
cut_max_dsh_notime_hits_per_layer = [float('inf')] * 3
cut_min_dsh_3usualtime_hits_per_layer = [0, 0, 0]
cut_max_dsh_3usualtime_hits_per_layer = [float('inf')] * 3

# --- QDC Per Layer (Requires list of 3) ---
cut_min_dsh_notime_qdc_per_layer = [0.0, 0.0, 0.0]
cut_max_dsh_notime_qdc_per_layer = [float('inf')] * 3
cut_min_dsh_3usualtime_qdc_per_layer = [0.0, 0.0, 0.0]
cut_max_dsh_3usualtime_qdc_per_layer = [float('inf')] * 3


# ------------------------------------------
# 4. DOWNSTREAM VERTICAL CUTS (DS-V) (4 Layers)
# ------------------------------------------
# --- Total Hits ---
cut_min_dsv_notime_total_hits = 0
cut_max_dsv_notime_total_hits = float('inf')
cut_min_dsv_3usualtime_total_hits = 0
cut_max_dsv_3usualtime_total_hits = float('inf')

# --- Total QDC ---
cut_min_dsv_notime_total_qdc = 0.0
cut_max_dsv_notime_total_qdc = float('inf')
cut_min_dsv_3usualtime_total_qdc = 0.0
cut_max_dsv_3usualtime_total_qdc = float('inf')

# --- Hits Per Layer (Requires list of 4) ---
cut_min_dsv_notime_hits_per_layer = [0, 0, 0, 0]
cut_max_dsv_notime_hits_per_layer = [float('inf')] * 4
cut_min_dsv_3usualtime_hits_per_layer = [0, 0, 0, 0]
cut_max_dsv_3usualtime_hits_per_layer = [float('inf')] * 4

# --- QDC Per Layer (Requires list of 4) ---
cut_min_dsv_notime_qdc_per_layer = [0.0, 0.0, 0.0, 0.0]
cut_max_dsv_notime_qdc_per_layer = [float('inf')] * 4
cut_min_dsv_3usualtime_qdc_per_layer = [0.0, 0.0, 0.0, 0.0]
cut_max_dsv_3usualtime_qdc_per_layer = [float('inf')] * 4"""