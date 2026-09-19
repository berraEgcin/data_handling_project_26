import json

duplicates_by_dataset = {"train": 0, "val": 0, "test": 0}

for name in ["train", "val", "test"]:
    with open(f"data/processed/{name}.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            output = json.loads(record["output"])
            findings = output["abnormal_findings"]
            
            # Check for duplicates
            keys = [f"{f['parameter']}:{f['flag']}" for f in findings]
            if len(keys) != len(set(keys)):
                duplicates_by_dataset[name] += 1

print("Duplicate findings confirmed:")
for name, count in duplicates_by_dataset.items():
    print(f"  {name}: {count}")