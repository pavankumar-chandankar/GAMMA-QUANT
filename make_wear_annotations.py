import os
import json
import pandas as pd

SENS_FOLDER = "data/wear/raw/inertial"
ANNO_FOLDER = "data/wear/annotations"
SAMPLING_RATE = 50.0  # matches wear_loso.yaml

os.makedirs(ANNO_FOLDER, exist_ok=True)

# collect subjects from CSV filenames (strip .csv)
csv_files = sorted(f for f in os.listdir(SENS_FOLDER) if f.endswith(".csv"))
stems = [os.path.splitext(f)[0] for f in csv_files]

if not stems:
    raise RuntimeError(f"No CSV files found in {SENS_FOLDER}")

print("Found CSV stems:", stems)

# main subjects: sbj_0, sbj_1, ..., sbj_21
main_subjects = sorted(
    s for s in stems
    if s.startswith("sbj_") and s.count("_") == 1 and s[4:].isdigit()
)

# extra recordings: e.g. sbj_0_2, sbj_14_2, etc.
extra_subjects = sorted(
    s for s in stems
    if s not in main_subjects
)

print("Main subjects (LOSO subjects):", main_subjects)
print("Extra recording subjects:", extra_subjects)

# ---- build label_dict from one CSV ----
sample_csv = os.path.join(SENS_FOLDER, csv_files[0])
df_sample = pd.read_csv(sample_csv)

if "label" not in df_sample.columns:
    raise RuntimeError(f"'label' column not found in {sample_csv}. Columns: {list(df_sample.columns)}")

unique_labels = sorted(str(x) for x in df_sample["label"].dropna().unique())
print("Detected labels:", unique_labels)

label_dict = unique_labels.copy()
for i, val in enumerate(label_dict):
    if val.lower() == "null":
        label_dict.insert(0, label_dict.pop(i))
        break

print("Using label_dict:", label_dict)

label_to_id = {lab: i for i, lab in enumerate(label_dict)}

def build_annotations_for_subject(stem: str):
    """
    Build a list of annotation segments for one subject.
    Each segment is a contiguous run of the same label.
    """
    csv_path = os.path.join(SENS_FOLDER, stem + ".csv")
    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        raise RuntimeError(f"'label' column not found in {csv_path}. Columns: {list(df.columns)}")

    labels = df["label"].tolist()
    annotations = []

    if not labels:
        return annotations

    current_label = labels[0]
    start_idx = 0

    for idx in range(1, len(labels) + 1):
        # either end of sequence or label change
        if idx == len(labels) or labels[idx] != current_label:
            end_idx = idx - 1  # inclusive
            if pd.isna(current_label):
                # skip NaN / None labels
                pass
            else:
                lab_str = str(current_label)
                if lab_str not in label_to_id:
                    # unseen label: extend label_dict? For now, skip.
                    print(f"[WARN] Label '{lab_str}' in {stem} not in label_dict, skipping segment.")
                else:
                    t_start = start_idx / SAMPLING_RATE
                    t_end = (end_idx + 1) / SAMPLING_RATE  # end exclusive
                    annotations.append({
                        "segment": [float(t_start), float(t_end)],
                        "label_id": int(label_to_id[lab_str]),
                    })
            # start new segment
            if idx < len(labels):
                current_label = labels[idx]
                start_idx = idx

    return annotations

# ---- generate LOSO JSON for each validation subject ----
for val_main in main_subjects:
    database = {}

    # main subjects
    for s in main_subjects:
        subset = "Validation" if s == val_main else "Training"
        database[s] = {"subset": subset, "annotations": build_annotations_for_subject(s)}

    # extra recordings follow their base subject's subset (sbj_0_2 -> base=sbj_0)
    for extra in extra_subjects:
        parts = extra.split("_")
        if len(parts) >= 2:
            base = "_".join(parts[:2])  # "sbj_0_2" -> "sbj_0"
        else:
            base = extra
        base_subset = database.get(base, {"subset": "Training"})["subset"]
        database[extra] = {"subset": base_subset, "annotations": build_annotations_for_subject(extra)}

    anno_obj = {
        "database": database,
        "label_dict": label_dict
    }

    out_name = f"loso_{val_main}.json"
    out_path = os.path.join(ANNO_FOLDER, out_name)
    with open(out_path, "w") as f:
        json.dump(anno_obj, f)
    print("Wrote", out_path)

print("Done.")
