import json
import sys

def filter_implausible_hba1c(input_file, output_file):
    """Remove records with biologically implausible HbA1c values"""
    
    removed = 0
    kept = 0
    
    with open(input_file, 'r') as inf, open(output_file, 'w') as outf:
        for line in inf:
            record = json.loads(line)
            output = json.loads(record['output'])
            
            # Check all HbA1c findings
            valid = True
            for finding in output['abnormal_findings']:
                if finding['parameter'] == 'HbA1c':
                    # HbA1c < 3.5% is essentially impossible
                    if finding['value'] < 3.5:
                        valid = False
                        removed += 1
                        break
            
            if valid:
                outf.write(line)
                kept += 1
    
    print(f"Records kept: {kept}")
    print(f"Records removed (implausible HbA1c): {removed}")
    return kept

# Apply filter
for name in ['train', 'val', 'test']:
    filter_implausible_hba1c(
        f'data/processed/{name}.jsonl',
        f'data/processed/{name}_filtered.jsonl'
    )