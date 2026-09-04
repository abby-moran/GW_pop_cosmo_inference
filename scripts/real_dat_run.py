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
include_file=cfg["run"]["list_file"]
mass_sel=cfg["run"].getfloat("mass_sel", fallback=2.5)

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
    for path in data_paths:
        print(f'globbing path: {path}')
        files += glob.glob(path)
    print('files to start: ', len(files))
    INCLUDE_LIST=[]
    with open(include_file, "r") as f:
        INCLUDE_LIST = set(line.strip() for line in f if line.strip())
        filtered_files = []
    for f in files:
        filename = os.path.basename(f)
        match = re.search(r"(GW\d{6}(?:_\d{6})?)",os.path.basename(filename) )
        if not match:
            print('unable to find event name for file', filename)
            break
        event_name = match.group(1)
        if event_name in INCLUDE_LIST:
            filtered_files.append(f)
        else:
            print('event excluded: ', event_name)
        #parts = re.split("_|-", filename)
        #if len(parts) >= 2 and parts[1] != 'GWTC4p0':
        #    event_name = parts[3] + "_" + parts[4]
        #    if event_name in INCLUDE_LIST:
        #        filtered_files.append(f)
        #if len(parts) >= 2 and parts[1] == 'GWTC4p0' or parts[1] == 'GWTC5p0':
        #    event_name = parts[4] + "_" + parts[5]
        #    if event_name in INCLUDE_LIST:
        #        filtered_files.append(f)

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
            print(f"Warning: {file} has only {n} samples (< {PE_samps}); "
              f"duplicating samples to reach required count (reduced n_eff).")
            # repeat arrays enough times to cover PE_samps, then trim
            reps = int(np.ceil(PE_samps / n))
            m1 = np.tile(m1, reps)[:PE_samps]
            q = np.tile(q, reps)[:PE_samps]
            dl = np.tile(dl, reps)[:PE_samps]
            pdraw = np.tile(pdraw, reps)[:PE_samps]
            idx = np.arange(PE_samps)  # no need to subsample further
        else:
            idx = np.random.choice(n, PE_samps, replace=False)

        filename = os.path.basename(file)
        parts = re.split("_|-", filename)
        match = re.search(r"(GW\d{6}(?:_\d{6})?)",os.path.basename(filename) )
        if not match:
            print('unable to find event name for file', filename)
            break
        event_name = match.group(1)

        rows.append({
            "m1": m1[idx],
            "q": q[idx],
            "dl": dl[idx],
            "pdraw": np.log(pdraw[idx]),
            "evt": event_name
        })

        print(f"Done {event_name}")

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
                                                    sel_input,nsamp=None, desired_pop_wt=None, mass_sel=mass_sel)

    df = pd.DataFrame({'m1': m1, 'q': q, 'z': z, 'a1': a1, 
                    'a2':a2, 'cos_tilt_1': cos_tilt1, 'cos_tilt_2': cos_tilt2, 
                    'pdraw_m1sqz': pdraw, 'ndraw': ndraw}) #m1 is source frame

    df['dm1sz_dm1ddl'] = weighting.dm1sz_dm1ddl(df['z']) 
    #df['dm1sz_dm1ddl'] = (1 + df['z'])**-1 / (df['dluminosity_distance_dredshift'].to_numpy()[detected][inds] / 1e3)
    df['pdraw_sel'] = df['pdraw_m1sqz']*df['dm1sz_dm1ddl']
    # dm1_src, dq dz to dm1_det, dq ddL to 
    df['m1d']= df['m1']*(1+df['z'])
    df['dl'] = Planck18.luminosity_distance(df['z'].to_numpy()).to(u.Gpc).value
    df.to_hdf(output_sel_file, key='true_parameters')
