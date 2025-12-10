import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List


from models.model_loader import create_model_driver


ALLOWED_CATEGORIES: List[str] = [
  "Weather & Air Quality",
  "Geography & Travel",
  "Health",
  "Science & Space",
  "Academic & Knowledge",
  "Office Automation",
  "Computer Vision",
  "E-commerce & Finance"
]

BBOX_OPEN = "<<BBOX>>"
BBOX_CLOSE = "<</BBOX>>"
_BBOX_RE = re.compile(r"<<BBOX>>(.*?)<</BBOX>>", re.DOTALL)


def _load_env_from_file(env_path: Path) -> None:
  """
  Read simple .env (KEY=VALUE), ignore comments/blank lines, inject into os.environ.
  No third-party dependency required.
  """
  if not env_path.exists():
    return
  try:
    with env_path.open("r", encoding="utf-8") as f:
      for line in f:
        s = line.strip()
        if (not s) or s.startswith("#"):
          continue
        if "=" not in s:
          continue
        key, val = s.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and (key not in os.environ):
          os.environ[key] = val
  except Exception:
    # Fail silently; do not block main flow
    pass


def _build_messages(tool_full_name: str, description: str) -> List[Dict[str, str]]:
  system_prompt = (
    "You are a strict single-label classifier. You must select exactly one category "
    "from the following list, and the output must exactly match one of these strings:\n"
    f"{', '.join(ALLOWED_CATEGORIES)}\n\n"
    "Output constraint: respond with ONLY one bbox-wrapped final answer, no other text or punctuation.\n"
    f"Format: {BBOX_OPEN}CATEGORY{BBOX_CLOSE}. For example: {BBOX_OPEN}{ALLOWED_CATEGORIES[0]}{BBOX_CLOSE}.\n"
    "Do not explain, do not add newlines, and do not add any prefixes or suffixes."
  )
  user_prompt = (
    "Given the tool name and description, choose the single best category.\n"
    f"Tool: {tool_full_name}\n"
    f"Description: {description}"
  )
  return [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
  ]


def _extract_bbox_label(text: str) -> Optional[str]:
  m = _BBOX_RE.search(text or "")
  if not m:
    return None
  return (m.group(1) or "").strip()


def classify_one(driver, tool_full_name: str, description: str, max_retries: int = 3) -> str:
  messages = _build_messages(tool_full_name, description)
  attempt = 0
  while attempt < max_retries:
    attempt += 1
    visible, _ = driver.generate_once(messages)
    label = _extract_bbox_label(visible)
    if label in ALLOWED_CATEGORIES:
      return label  # valid
    # Retry if not matched
  raise RuntimeError(f"Invalid model output: {visible}, tool: {tool_full_name}")


def main() -> None:
  # Compute project root
  script_path = Path(__file__).resolve()
  repo_root = script_path.parent.parent

  # Preload .env (same path as scripts/evaluate_final_answer.sh)
  _load_env_from_file(repo_root / ".env")

  # Target JSON file
  json_path = repo_root / "save" / "mcp_tools_with_desc.json"
  if not json_path.exists():
    raise FileNotFoundError(f"File not found: {json_path}")

  with json_path.open("r", encoding="utf-8") as f:
    data: Dict[str, Any] = json.load(f)

  # Create gpt-5-mini driver (same choice as evaluate_final_answer.sh)
  driver = create_model_driver("gpt-5-mini", max_new_tokens=10240)

  updated = False
  # Iterate structure: top-level groups (services), second-level tools with description
  for group_name, group_obj in (data or {}).items():
    if not isinstance(group_obj, dict):
      continue
    for tool_name, tool_obj in list(group_obj.items()):
      if not isinstance(tool_obj, dict):
        continue
      # Only process items missing 'catalogry' to avoid rework
      if "catalogry" in tool_obj:
        continue
      desc = tool_obj.get("description", "")
      tool_full_name = f"{group_name}.{tool_name}"

      try:
        label = classify_one(driver, tool_full_name, desc)
      except Exception as e:
        # If classification fails, warn and skip; try next run
        print(f"[WARN] Classification failed: {tool_full_name}: {e}")
        continue

      tool_obj["catalogry"] = label
      updated = True
      print(f"[OK] {tool_full_name} -> {label}")
      # Write back each item to persist incrementally
      with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

  if updated:
    with json_path.open("w", encoding="utf-8") as f:
      json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Updated and wrote: {json_path}")
  else:
    print("No updates needed: no tools missing 'catalogry'.")


if __name__ == "__main__":
  main()


