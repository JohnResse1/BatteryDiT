"""Build the SDU battery dataset for the BatteryDiT protocol.

Data source
-----------
Wang, S., Gao, F., and Tian, H.,
"Deep sorting of reused batteries for enabling long-term consistency
grouping with unknown prior conditions,"
Cell Reports Physical Science, 6, 102657 (2025).
DOI: 10.1016/j.xcrp.2025.102657

Dataset DOI: 10.5281/zenodo.14859405

The SDU (Shandong University) dataset contains commercial lithium-ion
cells subjected to primary-use and second-life aging experiments.

The raw records are independently processed here using the BatteryDiT
resampling, corruption, and evaluation protocol.
"""
from __future__ import annotations
import hashlib
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from ...corruptions import apply
from ...resampling import resample_charge
from ...schema import sample_path, write_sample

KNOWN_BAD_DIAGNOSTIC_CYCLES = {
    3: {221, 226},
    5: {161, 166, 171},
    10: {116},
    13: {261, 266},
    14: {266, 271, 276, 281},
    15: {256, 261, 266, 271},
    16: {266},
    17: {81, 86},
    18: {101, 106, 111, 116, 216},
    19: {76, 81, 86, 236, 241},
    20: {86, 91, 176, 196, 241, 246},
    21: {821, 826, 831, 836, 1221, 1511},
    22: {736, 741, 746, 751, 756, 761, 766, 811, 816, 1086},
    23: {116, 721, 726, 731, 736, 741, 1106, 1111},
    24: {946, 951, 956},
    27: {316},
    29: {156, 161, 166, 171, 176},
    30: {216, 221, 226},
    31: {281},
    32: {231, 236},
    37: {36, 41, 46},
    40: {166, 171},
    41: {206},
    42: {196, 201},
    43: {16, 266},
    44: {16},
    47: {56},
    48: {56, 61},
    51: {216, 221},
    52: {216, 221},
    53: {201},
    54: {201},
    55: {136},
    56: {201},
    59: {151, 201},
    60: {241},
    63: {106, 111},
    64: {96, 101},
    68: {276},
    69: {346, 601},
    70: {316, 561},
    73: {106},
    74: {226, 336, 491, 496},
    77: {11, 511, 516},
    79: {56, 61, 66, 446},
    80: {56, 61, 66, 446},
    83: {396},
    84: {41},
    85: {261},
    86: {276},
    87: {146, 151},
    88: {191},
}

CONDITIONS=("clean", "downsample", "segment", "random", "destructive")

def _seed(dataset, split, cell, cycle, condition):
    return int.from_bytes(hashlib.blake2b(f"{dataset}/{split}/{cell}/{cycle}/{condition}".encode(), digest_size=8).digest(), "little")

def _records(frame, source_number):
    records=[]
    frame=frame.copy()
    frame["Cycle_Index"]=pd.to_numeric(frame["Cycle_Index"],errors="coerce")
    frame["Step_Index"]=pd.to_numeric(frame["Step_Index"],errors="coerce")
    frame=frame.dropna(subset=["Cycle_Index","Step_Index"])
    for raw_cycle, cycle_frame in frame.groupby("Cycle_Index", sort=True):
        cycle=int(round(float(raw_cycle)))
        if cycle % 5 != 1 or cycle in KNOWN_BAD_DIAGNOSTIC_CYCLES.get(source_number,set()):
            continue
        charge=cycle_frame[cycle_frame["Step_Index"]==12]
        discharge=cycle_frame[cycle_frame["Step_Index"]==15]
        capacity=pd.to_numeric(discharge["Discharge_Capacity(Ah)"],errors="coerce").dropna()
        if charge.empty or capacity.empty: continue
        value=float(capacity.iloc[-1])
        if not .5 <= value <= 3.: continue
        coordinate=(pd.to_numeric(charge["Step_Time(s)"],errors="coerce").to_numpy() if "Step_Time(s)" in charge else np.arange(len(charge)))
        try:
            current,voltage=resample_charge(pd.to_numeric(charge["Current(A)"],errors="coerce").to_numpy(),pd.to_numeric(charge["Voltage(V)"],errors="coerce").to_numpy(),coordinate)
        except ValueError: continue
        records.append((cycle,current,voltage,value))
    return records

def build(dataset, data_root: Path, archive_root: Path, splits):
    requested={cell for cells in splits.values() for cell in cells}
    split_for={cell: split for split,cells in splits.items() for cell in cells}
    count=0
    with zipfile.ZipFile(archive_root / "Second_life_phase.zip") as archive:
        # The archive filename is a source-file number, whereas the frozen
        # split uses Battery_ID.  They are not interchangeable (e.g. SDU_10
        # is sourced from 2.csv), so build this mapping from the CSV metadata.
        members = {}
        for name in archive.namelist():
            if not name.endswith(".csv"):
                continue
            with archive.open(name) as handle:
                identity = pd.read_csv(handle, nrows=1)
            battery_id = str(identity.get("Battery_ID", identity.get("bat_num", pd.Series([Path(name).stem]))).iloc[0])
            members[f"SDU_{battery_id}"] = name
        for cell in sorted(requested):
            if cell not in members: raise KeyError(f"SDU source missing {cell}")
            source_number=int(Path(members[cell]).stem)
            with archive.open(members[cell]) as handle: records=_records(pd.read_csv(handle,low_memory=False),source_number)
            if len(records)<2: raise ValueError(f"{cell} has fewer than two usable diagnostic cycles")
            records.sort(key=lambda value:value[0]); first,last=records[0][0],records[-1][0]
            capacities=np.asarray([value[3] for value in records]); low,high=float(capacities.min()),float(capacities.max())
            if high<=low: raise ValueError(f"{cell} has no capacity span")
            split=split_for[cell]; conditions=CONDITIONS if split in {"vali", "test"} else ("clean",)
            for cycle,current,voltage,capacity in records:
                for condition in conditions:
                    i,v=apply(condition,current,voltage,_seed(dataset,split,cell,cycle,condition))
                    write_sample(sample_path(data_root,dataset,split,condition,cell,cycle),i,v,(capacity-low)/(high-low),(last-cycle)/(last-first)); count+=1
    return count
