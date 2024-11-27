import json

with open("data/oa_authors_merged.jsonl", "r") as f:
    for i, line in enumerate(f):
        
        try:
            ls = json.loads(line)
        except json.JSONDecodeError:
            print(i)
            raise
        