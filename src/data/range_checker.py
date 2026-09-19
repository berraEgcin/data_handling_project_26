import pandas as pd
import os

class RangeChecker:
    def __init__(self, ref_file="config/reference_ranges.csv"):
        if not os.path.exists(ref_file):
            raise FileNotFoundError(f"Reference range config not found: {ref_file}")
        self.df_ref = pd.read_csv(ref_file)
        
        self.gender_map = {
            'M': 'Male',
            'F': 'Female',
            'male': 'Male',
            'female': 'Female',
            'Male': 'Male',
            'Female': 'Female',
            'all': 'all'
        }

    def _normalize_gender(self, gender: str) -> str:
        if not gender:
            return 'all'
        gender_clean = str(gender).strip().upper()
        return self.gender_map.get(gender_clean, 'all')

    def evaluate(self, parameter: str, value: float, gender: str = "all", age: int = 40) -> str:
        """
        Evaluates a lab measurement and returns 'Low', 'Normal', or 'High'.
        
        Args:
            parameter: Lab test name
            value: Numeric lab value
            gender: Patient gender (M/F or Male/Female)
            age: Patient age
        
        Returns:
            'Low', 'Normal', or 'High'
        """
        # Normalize parameter name
        param_clean = parameter.strip().lower()
        matches = self.df_ref[self.df_ref['parameter'].str.lower() == param_clean]
        
        if matches.empty:
            print(f"WARNING: Parameter '{parameter}' not found in reference ranges")
            return "Normal"

        gender_normalized = self._normalize_gender(gender)
        
        gender_matches = matches[matches['gender'].str.lower() == gender_normalized.lower()]
        
        if gender_matches.empty:
            gender_matches = matches[matches['gender'].str.lower() == 'all']
        
        # If still no match, use first available
        if gender_matches.empty:
            row = matches.iloc[0]
        else:
            row = gender_matches.iloc[0]

        low = float(row['lower_bound'])
        high = float(row['upper_bound'])
        
        if low > high:
            print(f"ERROR: Invalid range for {parameter}: {low}-{high}")
            return "Normal"
        
        if value < low:
            return "Low"
        elif value > high:
            return "High"
        else:
            return "Normal"

if __name__ == "__main__":
    checker = RangeChecker()
    
    # Test
    print("Testing range checker:")
    tests = [
        ("Glucose", 168.0, "all", "High"),
        ("Hemoglobin", 11.1, "F", "Low"),
        ("Hemoglobin", 14.5, "M", "Normal"),
        ("Erythrocytes", 4.5, "F", "Normal"),
        ("HbA1c", 6.0, "all", "High"),
    ]
    
    for param, value, gender, expected in tests:
        result = checker.evaluate(param, value, gender=gender)
        status = "+" if result == expected else "-"
        print(f"{status} {param}={value} ({gender}): {result}")