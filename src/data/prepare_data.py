import os
import pandas as pd
from dataset_builder import CODE_TO_PARAM
from range_checker import RangeChecker


def prepare_data():
    obs_path = "data/raw/observations.csv"
    pat_path = "data/raw/patients.csv"
    out_path = "data/processed/lab_panels.csv"

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

    lab = df_obs[df_obs['CODE'].isin(CODE_TO_PARAM.keys())].copy()
    lab['parameter'] = lab['CODE'].map(CODE_TO_PARAM)
    lab['value'] = pd.to_numeric(lab['VALUE'], errors='coerce')
    lab = lab.dropna(subset=['value', 'ENCOUNTER'])

    print(f"  Filtered observations: {len(lab):,} records")

    lab = lab.merge(
        df_pat[['Id', 'GENDER', 'BIRTHDATE']],
        left_on='PATIENT', right_on='Id', how='left'
    )

    meta = lab.groupby('ENCOUNTER').agg(
        patient=('PATIENT', 'first'),
        date=('DATE', 'first'),
        gender=('GENDER', 'first'),
        birthdate=('BIRTHDATE', 'first'),
    )
    meta['age'] = meta['date'].str[:4].astype(int) - meta['birthdate'].str[:4].astype(int)
    meta = meta[['patient', 'date', 'age', 'gender']]

    print(f"  Encounters: {len(meta):,}")

    # skip children
    pediatric = meta.index[meta['age'] < 18]
    lab = lab[~lab['ENCOUNTER'].isin(pediatric)]

    # If a parameter has multiple DIFFERENT values in an encounter, that encounter is actually multiple real visits merged together.
    distinct_values = lab.groupby(['ENCOUNTER', 'parameter'])['value'].nunique()
    misaligned = distinct_values[distinct_values > 1].index.get_level_values('ENCOUNTER').unique()
    lab = lab[~lab['ENCOUNTER'].isin(misaligned)]

    # HbA1c < 3.5% is impossible, skip only this VALUE, not the whole record.
    impossible_hba1c = (lab['parameter'] == "HbA1c") & (lab['value'] < 3.5)
    lab = lab[~impossible_hba1c]

    # One row per encounter, one column per parameter
    panel = lab.pivot_table(index='ENCOUNTER', columns='parameter', values='value', aggfunc='first')
    lab_columns = panel.columns.tolist()

    df = meta.join(panel, how='inner').reset_index().rename(columns={'ENCOUNTER': 'encounter'})

    abnormal_count = pd.Series(0, index=df.index)
    for column in lab_columns:
        measured = df[column].notna()
        flags = [
            checker.evaluate(column, value, gender=gender, age=age)
            for value, gender, age in zip(
                df.loc[measured, column], df.loc[measured, 'gender'], df.loc[measured, 'age']
            )
        ]
        abnormal_count[measured] += [int(flag != "Normal") for flag in flags]

    df['abnormal_count'] = abnormal_count
    df['has_abnormal'] = (df['abnormal_count'] > 0).astype(int)

    print("\nData filtering:")
    print(f"  Panels created: {len(df):,}")
    print(f"  Skipped (pediatric < 18yo): {len(pediatric):,}")
    print(f"  Skipped (misaligned encounter - multiple visits merged): {len(misaligned):,}")
    print(f"  Implausible HbA1c values dropped (value-level, not record-level): {impossible_hba1c.sum():,}")

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\nSaved {len(df):,} panels x {df.shape[1]} columns -> {out_path}")
    print(f"  Abnormal: {df['has_abnormal'].sum():,} ({100 * df['has_abnormal'].mean():.1f}%)")
    print(f"  Normal:   {(df['has_abnormal'] == 0).sum():,} ({100 * (1 - df['has_abnormal'].mean()):.1f}%)")


if __name__ == "__main__":
    prepare_data()
