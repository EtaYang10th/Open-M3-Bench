#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path


def load_json_array(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
        return data
    except FileNotFoundError:
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: failed to parse JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)


def write_json_array(path: Path, data):
    # Preserve readable formatting similar to the source files
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Filter test_mcp.json by keeping only entries whose id appears in test_mcp_success.json")
    parser.add_argument(
        "--success",
        default="results/test_mcp.json",
        help="Path to test_mcp.json (will be modified in place)",
    )
    parser.add_argument(
        "--fail",
        default="results/test_mcp_success.json",
        help="Path to test_mcp_success.json (source of ids to keep)",
    )

    args = parser.parse_args()

    success_path = Path(args.success)
    fail_path = Path(args.fail)

    success_data = load_json_array(success_path)
    keep_source_data = load_json_array(fail_path)

    # Collect unique ids from success file to keep
    keep_ids = {entry.get("id") for entry in keep_source_data if isinstance(entry, dict) and "id" in entry}

    if None in keep_ids:
        keep_ids.discard(None)

    # Filter success data by keeping only entries whose ids are in keep_ids
    filtered = [entry for entry in success_data if isinstance(entry, dict) and entry.get("id") in keep_ids]

    # Write back in place
    write_json_array(success_path, filtered)

    removed_count = len(success_data) - len(filtered)
    print(f"Done. Removed {removed_count} entries from {success_path} based on {len(keep_ids)} success ids.")


if __name__ == "__main__":
    main()
