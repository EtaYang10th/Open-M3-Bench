# mcp-yolo (local DINO-X alternative)

This server is a **drop-in local replacement** for `dinox-mcp` used in
Open-M3-Bench. It exposes the same tool names and the same argument/return
shape as `servers/DINO-X-MCP`, backed by Ultralytics YOLO / YOLO-World so the
benchmark can run without a paid DINO-X API key or external quota.

## Exposed tools

Server key: `mcp-yolo`

| Tool name                  | Args                                              | Backend                               |
| -------------------------- | ------------------------------------------------- | ------------------------------------- |
| `detect-all-objects`       | `imageFileUri`, `includeDescription`              | Ultralytics YOLO11 (COCO 80)          |
| `detect-objects-by-text`   | `imageFileUri`, `textPrompt`, `includeDescription`| Ultralytics YOLO-World (open-vocab)   |

Return shape (identical to DINO-X stdio-server): three `text` content blocks
in order
1. `Objects detected in image: <cat1> (n1), <cat2> (n2), ...`
2. `Detailed object detection results: [{"name","bbox","description?"}, ...]`
3. Coordinate note explaining `[xmin, ymin, xmax, ymax]`.

`textPrompt` is parsed with `.`, `,`, `|` as separators (DINO-X convention).

## Differences vs DINO-X

- `includeDescription=true` uses `"<category> (conf=0.82)"` as `description`
  instead of a VLM caption. Field name stays as `description`, block layout is
  unchanged, so downstream code (`round_runner.py`, `evaluate_cv_issues.py`)
  does not need changes.
- No pose keypoints, no visualization. The GT file `json/test_mcp_GT.json`
  does not use those DINO-X tools (only `detect-all-objects` ×36 and
  `detect-objects-by-text` ×16), so this is sufficient for the current bench.

## Environment variables

- `MCP_YOLO_DETECT_WEIGHTS` (default `yolo11n.pt`)
- `MCP_YOLO_WORLD_WEIGHTS` (default `yolov8s-world.pt`)
- `MCP_YOLO_WEIGHTS_DIR` (default `<this dir>/weights`)
- `MCP_YOLO_CONF` (default `0.25`)
- `MCP_YOLO_IMGSZ` (default `640`)
- `MCP_YOLO_LOG_LEVEL` (default `INFO`)

Weights are loaded from `servers/mcp-yolo/weights/` and auto-downloaded there
on first use.

## Register in `mcp_servers.json`

```json
"mcp-yolo": {
  "_comment": "local DINO-X drop-in via Ultralytics YOLO / YOLO-World",
  "command": "servers/mcp-yolo/.venv/bin/python",
  "args": ["servers/mcp-yolo/server.py"],
  "env": {}
}
```
