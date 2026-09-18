import pandas as pd
import os

class RangeChecker:
    def __init__(self, ref_file="config/reference_ranges.csv"):
        if not os.path.exists(ref_file):
            raise FileNotFoundError(f"Reference range config not found: {ref_file}")
        self.df_ref = pd.read_csv(ref_file)

    def evaluate(self, parameter: str, value: float, gender: str = "all", age: int = 40) -> str:
        """
        Evaluates a lab measurement and returns 'Low', 'Normal', or 'High'.
        """
        # Normalize gender input (M/F -> Male/Female)
        if gender:
            g = str(gender).strip().upper()
            if g in ["M", "MALE"]:
                gender = "Male"
            elif g in ["F", "FEMALE"]:
                gender = "Female"
            else:
                gender = "all"
        else:
            gender = "all"
        # Find matching parameter
        matches = self.df_ref[self.df_ref['parameter'].str.lower() == parameter.strip().lower()]
        
        if matches.empty:
            return "Normal"

        # Match by gender if specified, otherwise fallback to 'all'
        gender_matches = matches[matches['gender'].str.lower() == gender.strip().lower()]
        row = gender_matches.iloc[0] if not gender_matches.empty else matches.iloc[0]

        low = float(row['lower_bound'])
        high = float(row['upper_bound'])

        if value < low:
            return "Low"
        elif value > high:
            return "High"
        return "Normal"

if __name__ == "__main__":
    checker = RangeChecker()
    # Quick sanity check
    print("Glucose 168 mg/dL:", checker.evaluate("Glucose", 168.0))
    print("Hemoglobin 11.1 g/dL (Female):", checker.evaluate("Hemoglobin", 11.1, gender="Female"))
    print("Hemoglobin 14.5 g/dL (Male):", checker.evaluate("Hemoglobin", 14.5, gender="Male"))