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
    obs_path = "data/raw/observations.csv"
    pat_path = "data/raw/patients.csv"
    
    checker = RangeChecker()

    print("Loading data...")
    df_obs = pd.read_csv(obs_path)
    df_pat = pd.read_csv(pat_path)

    # Filter observations to target LOINC codes
    df_obs = df_obs[df_obs['CODE'].isin(CODE_TO_PARAM.keys())].copy()
    df_obs['parameter'] = df_obs['CODE'].map(CODE_TO_PARAM)
    df_obs['VALUE_NUM'] = pd.to_numeric(df_obs['VALUE'], errors='coerce')
    df_obs = df_obs.dropna(subset=['VALUE_NUM', 'ENCOUNTER'])

    # Patient demographics map
    pat_map = df_pat.set_index('Id')[['GENDER', 'BIRTHDATE']].to_dict('index')

    # Group by encounter to form single lab panels
    encounters = df_obs.groupby('ENCOUNTER')
    samples = []

    for enc_id, group in encounters:
        patient_id = group['PATIENT'].iloc[0]
        pat_info = pat_map.get(patient_id, {'GENDER': 'all', 'BIRTHDATE': '1980-01-01'})
        gender = pat_info.get('GENDER', 'all')
        
        # Clinical Interpretation logic
        abnormal_findings = []
        lab_panel = {}

        for _, row in group.iterrows():
            param = row['parameter']
            val = float(row['VALUE_NUM'])
            unit = row['UNITS']
            status = checker.evaluate(param, val, gender=gender)

            lab_panel[param] = f"{val} {unit}"

            if status != "Normal":
                abnormal_findings.append({
                    "parameter": param,
                    "value": val,
                    "unit": unit,
                    "flag": status
                })

        if not lab_panel:
            continue

        # Format input and target output schema
        input_text = f"Patient Demographics: Gender={gender}. Lab Results: " + ", ".join([f"{k}: {v}" for k, v in lab_panel.items()])
        
        target_output = {
            "abnormal_findings": abnormal_findings,
            "interpretation_summary": f"Identified {len(abnormal_findings)} abnormal parameter(s)." if abnormal_findings else "All tested parameters within normal reference bounds."
        }

        samples.append({
            "input": input_text,
            "output": json.dumps(target_output)
        })

    # Shuffle and split 80 / 10 / 10
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

    os.makedirs("data/processed", exist_ok=True)
    for name, data in splits.items():
        out_path = f"data/processed/{name}.jsonl"
        with open(out_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"Saved {len(data):,} samples -> {out_path}")

if __name__ == "__main__":
    build_datasets()