"""Build the real-data GWTC-5 PE input with EXACTLY the 259 BBH events of the
official LVK GWTC-5 population analysis (arXiv:2605.27226).

The event list and the per-event preferred sampling group are read from the
`events` / `event_sample_IDs` root attributes of the LVK popsummary release
file (Zenodo 20292639), so the selection of events is fully reproducible and
never hand-edited.  This replaces the event-loading part of
`real_dat_run.py`, which had several problems (see the run-config comments):

  * the GWTC-2.1 glob was restricted to `GW190*`, excluding all 10 O1/O2
    events;
  * events with fewer raw PE samples than `pe_samps` were silently skipped
    (GW190521_030229, GW200129_065458, GW150914_095045, ...);
  * it pointed at GWTC-4.0 instead of the GWTC-4.1 release actually used by
    the LVK analysis (GW231026_130704 and GW231113_150041 only exist in 4.1);
  * the recorded `evt` names were mangled for GWTC-4/5 files
    (`721_GW230601`, `25_GW240413`, with colliding duplicates).

This script FAILS HARD if any of the 259 events cannot be loaded -- no
silent drops.  Events with fewer than `pe_samps` samples are resampled WITH
replacement (flagged in the output) instead of being dropped.

Output is a new-format PE HDF5 (datasets `m1`, `q`, `dl`, `pdraw` of shape
(259, pe_samps), `pdraw` = log of the PE prior density) directly readable by
`run_inf.py`, plus provenance datasets (`evt`, `pe_group`, `n_raw`,
`sampled_with_replacement`) and attrs.  The selection file is copied
unchanged from `sel_source` into the new run directory.

Run from `scripts/`:
    uv run python extract_realGWTC5_259.py --config run_configs/<cfg>.ini
"""

import argparse
import ast
import configparser
import datetime
import glob
import os
import re
import shutil
import subprocess
import sys

import h5py
import numpy as np

sys.path.append("../src/")
from weighting import get_samples_from_event  # noqa: E402

EVENT_RE = re.compile(r"(GW\d{6}_\d{6})")

# Last-resort group priority, used only if neither the popsummary sample ID
# nor the release's own events_list group is available in the file.
FALLBACK_GROUPS = [
    "C01:Mixed",
    "C00:Mixed",
    "C00:NRSur7dq4",
    "C01:IMRPhenomXPHM-SpinTaylor",
    "C00:IMRPhenomXPHM-SpinTaylor",
    "PublicationSamples",
    "PrecessingSpinIMRHM",
]


def read_popsummary_events(popsummary_file):
    """Return (event_names, sample_ids) from the popsummary root attrs."""
    with h5py.File(popsummary_file, "r") as f:
        events = [e.decode() if isinstance(e, bytes) else str(e)
                  for e in f.attrs["events"]]
        sample_ids = [e.decode() if isinstance(e, bytes) else str(e)
                      for e in f.attrs["event_sample_IDs"]]
        nev = f["posterior/reweighted_event_samples"].shape[0]
    if len(events) != nev:
        raise RuntimeError(
            f"popsummary events attr ({len(events)}) does not match "
            f"reweighted_event_samples first axis ({nev})")
    return events, sample_ids


def index_release_files(data_paths):
    """Glob the catalog release patterns; map event name -> file path."""
    fmap = {}
    for pattern in data_paths:
        hits = glob.glob(pattern)
        print(f"{len(hits):4d} files match {pattern}")
        for fn in hits:
            m = EVENT_RE.search(os.path.basename(fn))
            if m:
                fmap.setdefault(m.group(1), []).append(fn)
    dupes = {k: v for k, v in fmap.items() if len(v) > 1}
    if dupes:
        raise RuntimeError(
            "Multiple release files match a single event -- narrow the "
            f"data_paths patterns: {dupes}")
    return {k: v[0] for k, v in fmap.items()}


def load_release_group_lists(data_paths):
    """Read events_list_bbh_only.txt (name,group per line) next to each
    release directory, if present.  Returns event name -> group."""
    gmap = {}
    for pattern in data_paths:
        d = os.path.dirname(pattern)
        lst = os.path.join(d, "..", "events_list_bbh_only.txt")
        for cand in (os.path.join(d, "events_list_bbh_only.txt"), lst):
            if os.path.isfile(cand):
                with open(cand) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(",")
                        if len(parts) == 2:
                            gmap[parts[0]] = parts[1]
                break
    return gmap


def pick_group(fn, preferred, release_group, override=None):
    """Choose the posterior_samples group for one file.

    Order: explicit config override (must exist -- hard error otherwise),
    then the popsummary sample ID, then the release's own events_list group,
    then FALLBACK_GROUPS.  Returns (group, how)."""
    with h5py.File(fn, "r") as f:
        keys = set(f.keys())
    if override is not None:
        if override not in keys:
            raise RuntimeError(
                f"group_overrides requests {override!r} but {fn} only has "
                f"{sorted(keys)}")
        return override, "config_override"
    if preferred in keys:
        return preferred, "popsummary"
    if release_group and release_group in keys:
        return release_group, "release_events_list"
    for g in FALLBACK_GROUPS:
        if g in keys:
            return g, "fallback_priority"
    raise RuntimeError(f"No usable posterior group in {fn}; keys={sorted(keys)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to run config file")
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)
    ext = cfg["extract"]
    run = cfg["run"]

    popsummary_file = ext["popsummary_file"]
    data_paths = ast.literal_eval(ext["data_paths"])
    pe_samps = ext.getint("pe_samps", fallback=7000)
    seed = ext.getint("seed", fallback=20260829)
    zmax = ext.getfloat("zmax", fallback=3.0)
    sel_source = ext.get("sel_source", fallback=None)
    # Per-event posterior-group overrides (dict literal: event name -> group).
    # An override must exist in the release file, else the run fails hard.
    group_overrides = ast.literal_eval(ext.get("group_overrides", fallback="{}"))

    base_runs_dir = "../runs"
    run_dir = os.path.join(base_runs_dir, run["run_dir"])
    os.makedirs(run_dir, exist_ok=True)
    output_file_PE = os.path.join(run_dir, run["output_file_PE"])
    if os.path.exists(output_file_PE):
        raise SystemExit(f"Refusing to overwrite existing {output_file_PE}; "
                         "move it aside first.")

    # ---- authoritative event list -------------------------------------
    events, sample_ids = read_popsummary_events(popsummary_file)
    print(f"popsummary lists {len(events)} events "
          f"({len(set(events))} unique) from {popsummary_file}")

    # ---- locate one release file per event ----------------------------
    fmap = index_release_files(data_paths)
    missing = [e for e in events if e not in fmap]
    if missing:
        raise SystemExit(
            f"BLOCKER: no PE release file found for {len(missing)} event(s): "
            f"{missing}\nSearched patterns:\n  " + "\n  ".join(data_paths))

    release_groups = load_release_group_lists(data_paths)

    # ---- load, subsample, stack ----------------------------------------
    rng = np.random.default_rng(seed)
    m1_rows, q_rows, dl_rows, logp_rows = [], [], [], []
    groups_used, group_srcs, n_raws, replaced = [], [], [], []

    for ev, sid in zip(events, sample_ids):
        fn = fmap[ev]
        group, how = pick_group(fn, sid, release_groups.get(ev),
                                override=group_overrides.get(ev))
        result = get_samples_from_event(fn, zmax=zmax, group=group)
        if result is None:
            raise SystemExit(f"BLOCKER: failed to load {ev} from {fn} "
                             f"(group {group})")
        m1, q, dl, prior = result
        n = len(m1)
        if n == 0:
            raise SystemExit(f"BLOCKER: {ev} has 0 samples after z<{zmax} cut")
        if not (np.all(m1 > 0) and np.all(q > 0) and np.all(q <= 1)
                and np.all(dl > 0) and np.all(np.isfinite(prior))
                and np.all(prior > 0)):
            raise SystemExit(f"BLOCKER: unphysical samples in {ev} ({fn})")
        with_replacement = n < pe_samps
        idx = rng.choice(n, pe_samps, replace=with_replacement)
        if with_replacement:
            print(f"WARNING: {ev} has only {n} samples < pe_samps={pe_samps}; "
                  "sampling WITH replacement")
        m1_rows.append(m1[idx])
        q_rows.append(q[idx])
        dl_rows.append(dl[idx])
        logp_rows.append(np.log(prior[idx]))
        groups_used.append(group)
        group_srcs.append(how)
        n_raws.append(n)
        replaced.append(with_replacement)
        print(f"done {ev:18s} group={group:30s} ({how}) n_raw={n}")

    m1_arr = np.stack(m1_rows)
    q_arr = np.stack(q_rows)
    dl_arr = np.stack(dl_rows)
    logp_arr = np.stack(logp_rows)

    assert m1_arr.shape == (len(events), pe_samps)
    assert not np.any(np.isnan(logp_arr)) and not np.any(np.isinf(logp_arr))

    # ---- write new-format PE file (run_inf.py reads m1/q/dl/pdraw) ------
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = "UNKNOWN"

    str_dt = h5py.string_dtype()
    with h5py.File(output_file_PE, "w") as f:
        f["m1"] = m1_arr        # detector-frame primary mass [Msun]
        f["q"] = q_arr          # mass ratio (0, 1]
        f["dl"] = dl_arr        # luminosity distance [Gpc]
        f["pdraw"] = logp_arr   # LOG of the PE prior density in (m1_det,q,dl)
        f.create_dataset("evt", data=np.array(events, dtype=str_dt))
        f.create_dataset("pe_group", data=np.array(groups_used, dtype=str_dt))
        f.create_dataset("pe_group_source", data=np.array(group_srcs, dtype=str_dt))
        f.create_dataset("pe_file", data=np.array(
            [os.path.basename(fmap[e]) for e in events], dtype=str_dt))
        f["n_raw"] = np.array(n_raws, dtype=np.int64)
        f["sampled_with_replacement"] = np.array(replaced, dtype=bool)
        f.attrs["popsummary_file"] = os.path.abspath(popsummary_file)
        f.attrs["data_paths"] = "\n".join(data_paths)
        f.attrs["group_overrides"] = repr(group_overrides)
        f.attrs["pe_samps"] = pe_samps
        f.attrs["seed"] = seed
        f.attrs["zmax"] = zmax
        f.attrs["git_hash"] = git_hash
        f.attrs["created"] = datetime.datetime.now().isoformat()
        f.attrs["script"] = os.path.basename(__file__)
    print(f"\nwrote {output_file_PE}: {m1_arr.shape[0]} events x "
          f"{m1_arr.shape[1]} samples")

    # ---- provenance event list in the run dir ---------------------------
    ev_list_path = os.path.join(run_dir, "event_list_259ev.txt")
    with open(ev_list_path, "w") as fh:
        for ev, g in zip(events, groups_used):
            fh.write(f"{ev},{g}\n")
    print(f"wrote {ev_list_path}")

    # ---- copy the (already fixed) selection file, never modifying it ----
    if sel_source:
        sel_dest = os.path.join(run_dir, run["output_sel_file"])
        if os.path.abspath(sel_source) == os.path.abspath(sel_dest):
            raise SystemExit("sel_source and output_sel_file are the same "
                             "path; refusing.")
        if not os.path.exists(sel_dest):
            shutil.copyfile(sel_source, sel_dest)
            print(f"copied selection file {sel_source} -> {sel_dest}")
        else:
            print(f"selection file {sel_dest} already present; left untouched")

    # ---- copy the config for provenance ---------------------------------
    cfg["run"]["git_hash"] = git_hash
    with open(os.path.join(run_dir, os.path.basename(args.config)), "w") as fh:
        cfg.write(fh)

    n_rep = int(np.sum(replaced))
    print(f"\nSUMMARY: {len(events)} events; {n_rep} sampled with replacement "
          f"(n_raw < {pe_samps}): "
          f"{[e for e, r in zip(events, replaced) if r]}")


if __name__ == "__main__":
    main()
