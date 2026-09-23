#!/usr/bin/env bash
# Static checks on the workflows/ library — no container needed, runs in <1s.
#   * every workflow is API format (flat id -> {class_type, inputs}), not a UI export
#   * every manifest param path names a real node and input
#   * at most ONE manifest per tag carries "default" — two defaults make
#     `comfy txt2img` silently pick whichever sorts first
#   * every model file a workflow names is present under $COMFYUI_DATA_PATH/models
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
[ -f .env ] && . ./.env
DATA_PATH="${COMFYUI_DATA_PATH:-/home/scott/LLMs/comfyui}"

echo "==> verify-workflows: static checks on workflows/"
python3 - "$DATA_PATH" <<'PY'
import json, sys, glob, os
data = sys.argv[1]; problems = []
defaults = {}
folders = {"unet_name": "diffusion_models", "clip_name": "text_encoders", "vae_name": "vae",
           "ckpt_name": "checkpoints", "model_name": "latent_upscale_models", "lora_name": "loras"}
for wf in sorted(glob.glob("workflows/*.json")):
    if wf.endswith(".params.json"): continue
    g = json.load(open(wf)); name = os.path.basename(wf)
    if isinstance(g, dict) and "nodes" in g and "links" in g:
        problems.append(f"{name}: UI export, not API format (Workflow -> Export (API))"); continue
    if not (isinstance(g, dict) and g and all(isinstance(n, dict) and "class_type" in n for n in g.values())):
        problems.append(f"{name}: not an API-format workflow"); continue
    for nid, n in g.items():
        for k, v in n["inputs"].items():
            if k in folders and isinstance(v, str) and v.endswith(".safetensors") and not v.startswith("__"):
                if not os.path.isfile(os.path.join(data, "models", folders[k], v)):
                    problems.append(f"{name}: node {nid} needs models/{folders[k]}/{v} (not fetched)")
    mf = wf[:-5] + ".params.json"
    if not os.path.isfile(mf): continue
    m = json.load(open(mf))
    for tag in m.get("tags", []):
        if tag != "default" and "default" in m.get("tags", []):
            defaults.setdefault(tag, []).append(name)
    for pname, p in m.get("params", {}).items():
        node, sep, key = p["path"].partition(".inputs.")
        if not sep or node not in g: problems.append(f"{os.path.basename(mf)}: {pname} -> {p['path']}: no such node"); continue
        if key not in g[node]["inputs"] and "default" in p:
            problems.append(f"{os.path.basename(mf)}: {pname} -> {p['path']}: no such input on {g[node]['class_type']}")
for tag, names in defaults.items():
    if len(names) > 1: problems.append(f"tag '{tag}' has {len(names)} defaults: {', '.join(names)}")
for p in problems: print("    FAIL:", p)
print("    checked", len([w for w in glob.glob("workflows/*.json") if not w.endswith(".params.json")]), "workflows;",
      "defaults:", ", ".join(f"{t}={n[0]}" for t, n in defaults.items() if len(n) == 1) or "none")
sys.exit(1 if problems else 0)
PY
echo "==> verify-workflows: PASS"
