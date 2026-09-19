import os
import json
import random
import pandas as pd
from range_checker import RangeChecker

# Mapping Synthea descriptions to standard parameter keys
CODE_TO_PARAM = {
    "2339-0": "Glucose",
    "2345-7": "Glucose",
    "6299-2": "Urea Nitrogen",
    "3094-0": "Urea Nitrogen",
    "2947-0": "Sodium",
    "2951-2": "Sodium",
    "6298-4": "Potassium",
    "2823-3": "Potassium",
    "2069-3": "Chloride",
    "2075-0": "Chloride",
    "49765-1": "Calcium",
    "17861-6": "Calcium",
    "20565-8": "Carbon Dioxide",
    "2028-9": "Carbon Dioxide",
    "718-7": "Hemoglobin",
    "4544-3": "Hematocrit",
    "6690-2": "Leukocytes",
    "777-3": "Platelets",
    "789-8": "Erythrocytes",
    "787-2": "MCV",
    "4548-4": "HbA1c",
}

def build_datasets():
    """
    Build training datasets from Synthea observations.
    
    Process:
    1. Load raw Synthea data (observations.csv, patients.csv)
    2. Filter to target LOINC codes
    3. Group by encounter (single lab panel)
    4. Evaluate each parameter against reference ranges
    5. Create input/output pairs for fine-tuning
    6. Split into train/val/test (80/10/10)
    7. Save as JSONL files
    """
    
    obs_path = "data/raw/observations.csv"
    pat_path = "data/raw/patients.csv"
    
    # Verify files exist
    if not os.path.exists(obs_path):
        print(f"ERROR: {obs_path} not found")
        return
    if not os.path.exists(pat_path):
        print(f"ERROR: {pat_path} not found")
        return
    
    checker = RangeChecker()

    print("Loading data...")
    df_obs = pd.read_csv(obs_path)
    df_pat = pd.read_csv(pat_path)
    
    print(f"  Observations: {len(df_obs):,} records")
    print(f"  Patients: {len(df_pat):,} records")

    # Filter observations to target LOINC codes
    df_obs = df_obs[df_obs['CODE'].isin(CODE_TO_PARAM.keys())].copy()
    df_obs['parameter'] = df_obs['CODE'].map(CODE_TO_PARAM)
    df_obs['VALUE_NUM'] = pd.to_numeric(df_obs['VALUE'], errors='coerce')
    df_obs = df_obs.dropna(subset=['VALUE_NUM', 'ENCOUNTER'])
    
    print(f"  Filtered observations: {len(df_obs):,} records")

    # Create patient demographics map
    pat_map = df_pat.set_index('Id')[['GENDER', 'BIRTHDATE']].to_dict('index')
    print(f"  Patient map created: {len(pat_map):,} entries")

    # Group by encounter to form single lab panels
    encounters = df_obs.groupby('ENCOUNTER')
    print(f"  Encounters: {len(encounters):,}")
    
    samples = []
    skipped_impossible_hba1c = 0
    skipped_empty = 0
    skipped_other = 0

    for enc_id, group in encounters:
        patient_id = group['PATIENT'].iloc[0]
        pat_info = pat_map.get(
            patient_id, 
            {'GENDER': 'all', 'BIRTHDATE': '1980-01-01'}
        )
        gender = pat_info.get('GENDER', 'all')
        
        abnormal_findings = []
        lab_panel = {}
        
        # Track which parameters we've already processed
        # (avoid duplicates if same test appears twice in encounter)
        seen_params = set()
        skip_sample = False

        for _, row in group.iterrows():
            param = row['parameter']
            val = float(row['VALUE_NUM'])
            unit = row['UNITS']
            
            # SKIP: HbA1c < 3.5% is biologically impossible
            if param == "HbA1c" and val < 3.5:
                skip_sample = True
                skipped_impossible_hba1c += 1
                break  

            lab_panel[param] = f"{val} {unit}"

            # Only add to findings ONCE per parameter per encounter
            if param not in seen_params:
                status = checker.evaluate(param, val, gender=gender)
                if status != "Normal":
                    abnormal_findings.append({
                        "parameter": param,
                        "value": val,
                        "unit": unit,
                        "flag": status
                    })
                seen_params.add(param)

        # Skip if we marked it or if lab_panel is empty
        if skip_sample:
            continue
        
        if not lab_panel:
            skipped_empty += 1
            continue

        # Create training example
        input_text = (
            f"Patient Demographics: Gender={gender}. "
            f"Lab Results: "
        ) + ", ".join([f"{k}: {v}" for k, v in lab_panel.items()])
        
        target_output = {
            "abnormal_findings": abnormal_findings,
            "interpretation_summary": (
                f"Identified {len(abnormal_findings)} abnormal parameter(s)."
                if abnormal_findings
                else "All tested parameters within normal reference bounds."
            )
        }

        samples.append({
            "input": input_text,
            "output": json.dumps(target_output)
        })

    # Print skip statistics
    print(f"\nData filtering:")
    print(f"  Samples created: {len(samples):,}")
    print(f"  Skipped (impossible HbA1c): {skipped_impossible_hba1c:,}")
    print(f"  Skipped (empty panel): {skipped_empty:,}")

    # Shuffle and split 80 / 10 / 10
    print(f"\nShuffling and splitting...")
    random.seed(42)
    random.shuffle(samples)

    n_total = len(samples)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    splits = {
        "train": samples[:n_train],
        "val": samples[n_train:n_train + n_val],
        "test": samples[n_train + n_val:]
    }

    # Create output directory and save
    os.makedirs("data/processed", exist_ok=True)
    print(f"\nSaving datasets to data/processed/")
    
    for name, data in splits.items():
        out_path = f"data/processed/{name}.jsonl"
        with open(out_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"  Saved {len(data):,} samples -> {out_path}")

    # Print summary
    print(f"\n" + "="*80)
    print(f"SUMMARY")
    print(f"="*80)
    print(f"Total records: {n_total}")
    print(f"  Train: {len(splits['train'])} (80%)")
    print(f"  Val:   {len(splits['val'])} (10%)")
    print(f"  Test:  {len(splits['test'])} (10%)")
    print(f"="*80)

if __name__ == "__main__":
    build_datasets()