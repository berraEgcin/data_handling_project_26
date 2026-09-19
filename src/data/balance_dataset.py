import json
import random

def balance_dataset(input_file, output_file, target_normal_ratio=0.40):
    normal_records = []
    abnormal_records = []
    
    with open(input_file, 'r') as f:
        for line in f:
            record = json.loads(line)
            output = json.loads(record['output'])
            
            if len(output['abnormal_findings']) == 0:
                normal_records.append(record)
            else:
                abnormal_records.append(record)
    
    total_abnormal = len(abnormal_records)
    target_normal = int(total_abnormal * target_normal_ratio / (1 - target_normal_ratio))
    
    print(f"Abnormal records: {total_abnormal}")
    print(f"Normal records available: {len(normal_records)}")
    print(f"Target normal records: {target_normal}")
    
    if len(normal_records) < target_normal:
        balanced_normal = normal_records * (target_normal // len(normal_records) + 1)
        balanced_normal = balanced_normal[:target_normal]
    else:
        balanced_normal = random.sample(normal_records, target_normal)
    
    balanced = balanced_normal + abnormal_records
    random.shuffle(balanced)
    
    # Save
    with open(output_file, 'w') as f:
        for record in balanced:
            f.write(json.dumps(record) + '\n')
    
    print(f"Balanced dataset saved: {len(balanced)} records")
    print(f"Ratio: {100*len(balanced_normal)/len(balanced):.1f}% normal")

# Apply to train set
balance_dataset('data/processed/train.jsonl', 'data/processed/train_balanced.jsonl')