import pandas as pd
import glob
import numpy as np
import sys
sys.path.append('../src/')
import warnings
from astropy.cosmology import Planck18
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import os
from weighting import get_samples_from_event
import argparse
import astropy.units as u
import configparser
import subprocess
import weighting 
import re
import ast


#load in config stuff
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to run config file")
args = parser.parse_args()

cfg = configparser.ConfigParser()
cfg.read(args.config)

#setup direcotry
base_runs_dir = "../runs"
run_name = cfg["run"]["run_dir"]
os.makedirs(base_runs_dir, exist_ok=True)
run_dir = os.path.join(base_runs_dir, f"{run_name}")
os.makedirs(run_dir, exist_ok=False)

# extract the paramters we need from the config file
output_file_PE = os.path.join(run_dir, cfg["run"]["output_file_PE"])
output_sel_file = os.path.join(run_dir, cfg["run"]["output_sel_file"])
sel_input = cfg["run"]["sel_input"]
data_paths = ast.literal_eval(cfg["run"]["data_paths"])

PE_samps = cfg["run"].getint("PE_samps", fallback=1000)

#copy the confg file, with the hash to the new directory
try:
    git_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        stderr=subprocess.DEVNULL
    ).decode().strip()
except Exception:
    git_hash = "UNKNOWN"
cfg["run"]["git_hash"] = git_hash
ini_file = cfg["run"]["ini_file"]
run_ini_path = os.path.join(run_dir, ini_file)
with open(run_ini_path, "w") as f:
    cfg.write(f)

if __name__ == "__main__":
    files = []
    print(f'data paths: {data_paths}')
    for path in data_paths:
        print(f'globbing path: {path}')
        files += glob.glob(path)
    #folder1 = 
    #folder2 = "../../GW_2025/GWTC-3"
    #files = glob.glob(os.path.join(folder1, "*_nocosmo.h5"))
    #files += glob.glob(os.path.join(folder2, "*_nocosmo.h5"))
    print(files)
    INCLUDE_LIST=[]
    with open("../runs/INCLUDE_LIST.txt", "r") as f:
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
    # PE_dfs = []
    rows = []
    for file in filtered_files:
        result = get_samples_from_event(file)
        if result is None:
            continue

        m1, q, dl, pdraw = result

        n = len(m1)
        if n < PE_samps:
            continue

        idx = np.random.choice(n, PE_samps, replace=False)

        filename = os.path.basename(file)
        parts = re.split("_|-", filename)
        event_here = parts[3] + "_" + parts[4]

        rows.append({
            "m1": m1[idx],
            "q": q[idx],
            "dl": dl[idx],
            "pdraw": np.log(pdraw[idx]),
            "evt": event_here
        })

        print(f"Done {event_here}")

    # stack into (nevents, PE_samps)
    m1_arr = np.stack([r["m1"] for r in rows])
    q_arr  = np.stack([r["q"] for r in rows])
    dl_arr = np.stack([r["dl"] for r in rows])
    pdraw_arr = np.stack([r["pdraw"] for r in rows])
    evt_arr = np.array([r["evt"] for r in rows])  # (nevents,)

    final_df = pd.DataFrame({
        "m1": list(m1_arr),
        "q": list(q_arr),
        "dl": list(dl_arr),
        "pdraw": list(pdraw_arr),
        "evt": evt_arr
    })

    final_df.to_hdf(output_file_PE, key="samples", mode="w")

    # Now selection samples!
    (m1, q, z, a1, a2, cos_tilt1, cos_tilt2, pdraw, ndraw) = weighting.extract_selection_samples(
                                                    sel_input,nsamp=None, desired_pop_wt=None)

    df = pd.DataFrame({'m1': m1, 'q': q, 'z': z, 'a1': a1, 
                    'a2':a2, 'cos_tilt_1': cos_tilt1, 'cos_tilt_2': cos_tilt2, 
                    'pdraw_m1sqz': pdraw, 'ndraw': ndraw}) #m1 is source frame

    df['dm1sz_dm1ddl'] = weighting.dm1sz_dm1ddl(df['z'])
    df['pdraw_sel'] = df['pdraw_m1sqz']*df['dm1sz_dm1ddl']
    df['m1d']= df['m1']*(1+df['z'])
    df['dl'] = Planck18.luminosity_distance(df['z'].to_numpy()).to(u.Gpc).value
    df.to_hdf(output_sel_file, key='true_parameters')


