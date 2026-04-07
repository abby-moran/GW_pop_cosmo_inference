import pandas as pd
import h5py
import glob
from scipy import stats
import numpy as np
import sys
sys.path.append('../src/')
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import os
from weighting import get_samples_from_event


if __name__ == "__main__":
    folder1 = "../../GW_2025/GWTC-21"
    folder2 = "../../GW_2025/GWTC-3"
    files = glob.glob(os.path.join(folder1, "*_nocosmo.h5"))
    files += glob.glob(os.path.join(folder2, "*_nocosmo.h5"))
    INCLUDE_LIST=[]
    with open("INCLUDE_LIST.txt", "r") as f:
        INCLUDE_LIST = set(line.strip() for line in f if line.strip())
        filtered_files = []
    for f in files:
        filename = os.path.basename(f)
        parts = re.split("_|-", filename)
        if len(parts) >= 2:
            event_name = parts[3] + "_" + parts[4]
            if event_name in INCLUDE_LIST:
                filtered_files.append(f)

    print(f"Filtered to {len(filtered_files)} files.")
    event_df = pd.DataFrame()
    PE_dfs = []
    for file in filtered_files:
        result = get_samples_from_event(file)
        if result is None:
            continue
        df_here = pd.DataFrame()
        df_here["mass_1"], df_here["mass_ratio"], df_here["luminosity_distance_Gpc"], df_here["prior_m1d_q_dL"] = result
        try:
            df_here = df_here.sample(1000, replace=False)
        except Exception as e:
            print(e)
            continue
        filename = os.path.basename(file)
        parts = re.split("_|-", filename)
        event_here = parts[3] + "_" + parts[4]
        df_here['evt'] = event_here
        print(f"Done {event_here}")
        PE_dfs.append(df_here)
    final_df = pd.concat(PE_dfs, ignore_index=True)
    final_df.to_hdf('./pe_samples.h5', key='samples', mode='w')
