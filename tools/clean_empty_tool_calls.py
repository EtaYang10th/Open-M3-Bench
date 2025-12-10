#!/usr/bin/env python3
import json

path = 'results/internvl3.5-latest_test_mcp_fuzzy.json'

def bad_sample(sample):
    for turn in sample.get("dialogue", []):
        # case 1: tool_calls at top level of turn
        if turn.get("tool_calls", None) == []:
            return True
        # case 2: tool_calls inside "work"
        work = turn.get("work", {})
        if work.get("tool_calls", None) == []:
            print(f"'{sample['id']}',")
            return True
    return False

with open(path, "r") as f:
    data = json.load(f)

if isinstance(data, list):
    new_data = [s for s in data if not bad_sample(s)]
elif isinstance(data, dict):
    # if it's {id: sample_obj, ...}
    new_data = {k: v for k, v in data.items() if not bad_sample(v)}
else:
    raise TypeError("unexpected json format")

with open(path, "w") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)
