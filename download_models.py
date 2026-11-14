#!/usr/bin/env python
"""Download and sync Qwen-Image 2.1 models across multiple developers and Google Drive.

Features:
- Discovers compatible models across multiple community developers (KasugaiSakura, Unsloth, AtomicChat, Comfy-Org).
- Checks Google Drive and local cache first for instant hits (no re-downloading).
- Interactive model selection: lists all compatible models with sizes and variants, prompts user which one to load.
- Seamless sync: downloads selected model and saves a backup copy to Google Drive.
- Supports custom Hugging Face models by passing `repo_id:filename` or URL.

CLI usage:
    python download_models.py --list-models             # Print full catalog of compatible models
    python download_models.py --interactive             # Ask user interactively which model to load
    python download_models.py --select 1                # Select model by catalog ID (1, 2, ...)
    python download_models.py --model Q4_K_M            # Select by quant tag or filename
    python download_models.py --select "unsloth/Qwen-Image-2.1-GGUF:qwen-image-2.1-Q4_K_M.gguf"
"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import struct
import sys
import urllib.parse

# Fast downloads — must be set before huggingface_hub is imported
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
if importlib.util.find_spec("hf_transfer") and importlib.metadata.version("huggingface_hub") < "1":
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import requests  # noqa: E402
from huggingface_hub import hf_hub_download, HfApi  # noqa: E402

DEFAULT_REPO = "KasugaiSakura/Qwen-Image-2.1-Uncensored-Abenzerps-GGUF"
COMFY_ORG_REPO = "Comfy-Org/Qwen-Image-2.1"

# Legacy dictionary preserved for full backward compatibility
MODELS = {
    "Q4_0": ("qwen-image-2.1-UC-Q4_0.gguf", DEFAULT_REPO, "main", 4.15),
    "Q4_K_M": ("qwen-image-2.1-UC-Q4_K_M.gguf", DEFAULT_REPO, "main", 4.60),
    "Q8_0": ("qwen-image-2.1-UC-Q8_0.gguf", DEFAULT_REPO, "main", 7.59),
    "BF16": ("qwen-image-2.1-UC-BF16.gguf", DEFAULT_REPO, "main", 14.23),
    "fp8": ("qwen-image-2.1-UC-fp8.safetensors", DEFAULT_REPO, "main", 7.12),
    "int8": ("qwen-image-2.1-UC-int8_convrot.safetensors", DEFAULT_REPO, "main", 7.26),
    "base-Q4_0": ("qwen-image-2.1-Q4_0.gguf", DEFAULT_REPO, "main", 4.05),
    "base-Q4_K_M": ("qwen-image-2.1-Q4_K_M.gguf", "unsloth/Qwen-Image-2.1-GGUF", "main", 4.60),
    "base-Q8_0": ("qwen-image-2.1-Q8_0.gguf", "unsloth/Qwen-Image-2.1-GGUF", "main", 7.59),
    "base-int8": ("diffusion_models/qwen_image_2.1_int8_convrot.safetensors", COMFY_ORG_REPO, "main", 7.26),
    "base-bf16": ("diffusion_models/qwen_image_2.1_bf16.safetensors", COMFY_ORG_REPO, "main", 14.23),
}

TEXT_ENCODERS = {
    "int8": ("text_encoders/qwen3vl_8b_int8_convrot.safetensors", DEFAULT_REPO, 9.35),
    "bf16": ("text_encoders/qwen3vl_8b_bf16.safetensors", DEFAULT_REPO, 17.53),
    "w4a8": ("text_encoders/qwen3vl_8b_w4a8.safetensors", COMFY_ORG_REPO, 6.3),
}

VAE = "vae/qwen_image_2.1_vae_bf16.safetensors"

# --- Multi-Developer Curated Repository Catalog ---
KNOWN_REPOSITORIES = [
    {
        "developer": "KasugaiSakura (abenzerps)",
        "repo": "KasugaiSakura/Qwen-Image-2.1-Uncensored-Abenzerps-GGUF",
        "variant": "Uncensored",
        "description": "De-censored community fine-tune; removed safety suppression vectors.",
        "revision": "main",
        "models": [
            {"filename": "qwen-image-2.1-UC-Q4_K_M.gguf", "quant": "Q4_K_M (4-bit)", "size_gb": 4.60, "notes": "Recommended standard (fast & balanced)"},
            {"filename": "qwen-image-2.1-UC-Q4_0.gguf", "quant": "Q4_0 (4-bit)", "size_gb": 4.15, "notes": "Smallest 4-bit quant"},
            {"filename": "qwen-image-2.1-UC-Q5_K_M.gguf", "quant": "Q5_K_M (5-bit)", "size_gb": 5.35, "notes": "Higher fidelity 5-bit"},
            {"filename": "qwen-image-2.1-UC-Q6_K.gguf", "quant": "Q6_K (6-bit)", "size_gb": 6.17, "notes": "Near-lossless 6-bit"},
            {"filename": "qwen-image-2.1-UC-Q8_0.gguf", "quant": "Q8_0 (8-bit)", "size_gb": 7.59, "notes": "High precision 8-bit"},
            {"filename": "qwen-image-2.1-UC-BF16.gguf", "quant": "BF16 (16-bit)", "size_gb": 14.23, "notes": "Full precision uncompressed GGUF"},
            {"filename": "qwen-image-2.1-UC-fp8.safetensors", "quant": "FP8", "size_gb": 7.12, "notes": "FP8 safetensors (UNETLoader)"},
            {"filename": "qwen-image-2.1-UC-int8_convrot.safetensors", "quant": "INT8", "size_gb": 7.26, "notes": "INT8 safetensors"}
        ]
    },
    {
        "developer": "Unsloth",
        "repo": "unsloth/Qwen-Image-2.1-GGUF",
        "variant": "Base / Official",
        "description": "Official Qwen-Image 2.1 base weights quantized by Unsloth.",
        "revision": "main",
        "models": [
            {"filename": "qwen-image-2.1-Q4_K_M.gguf", "quant": "Q4_K_M (4-bit)", "size_gb": 4.60, "notes": "Base official model 4-bit"},
            {"filename": "qwen-image-2.1-Q5_K_M.gguf", "quant": "Q5_K_M (5-bit)", "size_gb": 5.35, "notes": "Base official model 5-bit"},
            {"filename": "qwen-image-2.1-Q6_K.gguf", "quant": "Q6_K (6-bit)", "size_gb": 6.17, "notes": "Base official model 6-bit"},
            {"filename": "qwen-image-2.1-Q8_0.gguf", "quant": "Q8_0 (8-bit)", "size_gb": 7.59, "notes": "Base official model 8-bit"},
            {"filename": "qwen-image-2.1-F16.gguf", "quant": "FP16 (16-bit)", "size_gb": 14.23, "notes": "Base official model unquantized"}
        ]
    },
    {
        "developer": "AtomicChat",
        "repo": "AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF",
        "variant": "Turbo Uncensored",
        "description": "Turbo step-distilled uncensored variant (fewer steps needed).",
        "revision": "main",
        "models": [
            {"filename": "Qwen-Image-2.1-Turbo-Uncensored-AD-Q4_K.gguf", "quant": "Q4_K (4-bit)", "size_gb": 4.60, "notes": "Turbo accelerated 4-bit"},
            {"filename": "Qwen-Image-2.1-Turbo-Uncensored-Q8_0.gguf", "quant": "Q8_0 (8-bit)", "size_gb": 7.59, "notes": "Turbo accelerated 8-bit"},
            {"filename": "Qwen-Image-2.1-Turbo-Uncensored-BF16.gguf", "quant": "BF16 (16-bit)", "size_gb": 14.23, "notes": "Turbo accelerated BF16"}
        ]
    },
    {
        "developer": "Comfy-Org (Official)",
        "repo": "Comfy-Org/Qwen-Image-2.1",
        "variant": "Official Upstream",
        "description": "Official ComfyUI upstream releases in safetensors format.",
        "revision": "main",
        "models": [
            {"filename": "diffusion_models/qwen_image_2.1_int8_convrot.safetensors", "quant": "INT8", "size_gb": 7.26, "notes": "Official upstream INT8"},
            {"filename": "diffusion_models/qwen_image_2.1_bf16.safetensors", "quant": "BF16", "size_gb": 14.23, "notes": "Official upstream BF16"},
            {"filename": "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors", "quant": "Turbo INT8", "size_gb": 7.26, "notes": "Official turbo INT8"}
        ]
    }
]


def model_key(name):
    """Case-insensitive lookup for backward-compatible models dictionary."""
    for k in MODELS:
        if k.lower() == str(name).lower():
            return k
    raise argparse.ArgumentTypeError(f"unknown model {name!r}; choose from {', '.join(MODELS)}")


def diffusion_file(name):
    """Return model filename given key, ID, or direct filename."""
    if not name:
        return "qwen-image-2.1-UC-Q4_K_M.gguf"
    name_str = str(name).strip()
    # Check if direct filename
    if name_str.endswith((".gguf", ".safetensors")):
        return os.path.basename(name_str)
    # Check legacy dict
    for k, v in MODELS.items():
        if k.lower() == name_str.lower():
            return os.path.basename(v[0])
    return name_str


# ---- Catalog Discovery & Management -------------------------------------------------

def build_model_catalog(drive_root=None, comfy_dir="ComfyUI", fetch_online=True):
    """Scan Google Drive, local ComfyUI storage, and known community repositories to build
    a comprehensive catalog of compatible Qwen-Image 2.1 models."""
    catalog = []
    seen_filenames = set()
    
    # 1. Scan Google Drive Models Cache first
    if drive_root:
        drive_diff_dir = os.path.join(drive_root, "models", "Diffusion Models")
        if os.path.isdir(drive_diff_dir):
            for fname in sorted(os.listdir(drive_diff_dir)):
                if fname.lower().endswith((".gguf", ".safetensors")) and not fname.startswith("."):
                    full_p = os.path.join(drive_diff_dir, fname)
                    sz_gb = round(os.path.getsize(full_p) / 1e9, 2)
                    seen_filenames.add(fname.lower())
                    catalog.append({
                        "filename": fname,
                        "developer": "Google Drive Cache",
                        "variant": "Cached in Drive",
                        "quant": "GGUF" if fname.endswith(".gguf") else "Safetensors",
                        "size_gb": sz_gb,
                        "status": "Ready (Drive)",
                        "source": "Google Drive Cache",
                        "repo": None,
                        "subpath": fname,
                        "revision": "main",
                        "notes": "Instantly copied from your Google Drive (zero download time)"
                    })

    # 2. Scan Local ComfyUI models directory
    if comfy_dir:
        local_diff_dir = os.path.join(comfy_dir, "models", "diffusion_models")
        if os.path.isdir(local_diff_dir):
            for fname in sorted(os.listdir(local_diff_dir)):
                if fname.lower().endswith((".gguf", ".safetensors")) and not fname.startswith("."):
                    if fname.lower() not in seen_filenames:
                        full_p = os.path.join(local_diff_dir, fname)
                        sz_gb = round(os.path.getsize(full_p) / 1e9, 2)
                        seen_filenames.add(fname.lower())
                        catalog.append({
                            "filename": fname,
                            "developer": "Local Machine",
                            "variant": "Local Disk",
                            "quant": "GGUF" if fname.endswith(".gguf") else "Safetensors",
                            "size_gb": sz_gb,
                            "status": "Ready (Local)",
                            "source": "Local Disk",
                            "repo": None,
                            "subpath": fname,
                            "revision": "main",
                            "notes": "Already present on local disk"
                        })

    # 3. Add Curated Multi-Developer Models
    for dev_repo in KNOWN_REPOSITORIES:
        dev = dev_repo["developer"]
        repo = dev_repo["repo"]
        var = dev_repo["variant"]
        rev = dev_repo.get("revision", "main")
        for m in dev_repo["models"]:
            fname = os.path.basename(m["filename"])
            is_in_drive = fname.lower() in seen_filenames
            
            # Check local file presence if comfy_dir provided
            is_local = False
            if comfy_dir:
                local_f = os.path.join(comfy_dir, "models", "diffusion_models", fname)
                is_local = os.path.isfile(local_f) and os.path.getsize(local_f) > 1024

            status = "Ready (Local)" if is_local else ("Ready (Drive)" if is_in_drive else "Hugging Face")
            catalog.append({
                "filename": fname,
                "developer": dev,
                "variant": var,
                "quant": m["quant"],
                "size_gb": m["size_gb"],
                "status": status,
                "source": "Google Drive Cache" if is_in_drive else "Hugging Face",
                "repo": repo,
                "subpath": m["filename"],
                "revision": rev,
                "notes": m.get("notes", dev_repo.get("description", ""))
            })

    # Assign sequential 1-based IDs
    for idx, item in enumerate(catalog, start=1):
        item["id"] = idx

    return catalog


def format_model_catalog_table(catalog, as_html=False):
    """Format the model catalog as an ASCII/terminal table or an HTML table."""
    if as_html:
        rows = []
        for item in catalog:
            badge_color = "#137333" if "Ready" in item["status"] else "#1a73e8"
            bg_color = "#e6f4ea" if "Ready" in item["status"] else "#e8f0fe"
            rows.append(f"""
            <tr style="border-bottom: 1px solid #e0e0e0; font-size: 13px;">
                <td style="padding: 10px; font-weight: bold; text-align: center; color: #202124;">[{item['id']}]</td>
                <td style="padding: 10px; font-family: monospace; font-weight: 600; color: #1a73e8;">{item['filename']}</td>
                <td style="padding: 10px; color: #3c4043;">{item['developer']}</td>
                <td style="padding: 10px; color: #5f6368;"><span style="background: #f1f3f4; padding: 2px 6px; border-radius: 4px;">{item['variant']}</span></td>
                <td style="padding: 10px; text-align: center; color: #3c4043;">{item['quant']}</td>
                <td style="padding: 10px; text-align: center; color: #3c4043;">{item['size_gb']} GB</td>
                <td style="padding: 10px; text-align: center;">
                    <span style="background: {bg_color}; color: {badge_color}; padding: 3px 8px; border-radius: 12px; font-weight: 500; font-size: 11px;">
                        {item['status']}
                    </span>
                </td>
                <td style="padding: 10px; color: #5f6368; font-size: 12px;">{item['notes']}</td>
            </tr>
            """)
            
        html = f"""
        <div style="font-family: sans-serif; margin-top: 10px;">
            <h3 style="margin: 0 0 10px 0; color: #202124;">Available Compatible Qwen-Image 2.1 Models ({len(catalog)} options):</h3>
            <div style="overflow-x: auto;">
                <table style="border-collapse: collapse; width: 100%; max-width: 1100px; background: white; border: 1px solid #dadce0; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                    <thead>
                        <tr style="background: #f8f9fa; border-bottom: 2px solid #dadce0; text-align: left; font-size: 12px; color: #5f6368; text-transform: uppercase;">
                            <th style="padding: 10px; text-align: center;">ID</th>
                            <th style="padding: 10px;">Model Filename</th>
                            <th style="padding: 10px;">Developer / Provider</th>
                            <th style="padding: 10px;">Variant</th>
                            <th style="padding: 10px; text-align: center;">Quantization</th>
                            <th style="padding: 10px; text-align: center;">Size</th>
                            <th style="padding: 10px; text-align: center;">Status</th>
                            <th style="padding: 10px;">Notes</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(rows)}
                    </tbody>
                </table>
            </div>
            <p style="font-size: 12px; color: #5f6368; margin-top: 8px;">
                <b>How to load:</b> Set <code>MODEL_CHOICE = 1</code> (or enter model ID/filename) in the cell below to load your selection.
            </p>
        </div>
        """
        return html
    else:
        # Terminal ASCII formatting
        header = f"{'ID':<4} | {'Model Filename':<40} | {'Developer':<24} | {'Quant':<14} | {'Size':<8} | {'Status':<16} | {'Notes'}"
        sep = "-" * len(header)
        lines = [sep, header, sep]
        for item in catalog:
            sz_str = f"{item['size_gb']} GB"
            lines.append(
                f"{item['id']:<4} | {item['filename']:<40} | {item['developer']:<24} | "
                f"{item['quant']:<14} | {sz_str:<8} | {item['status']:<16} | {item['notes']}"
            )
        lines.append(sep)
        return "\n".join(lines)


def resolve_model_choice(choice, catalog):
    """Resolve user input (integer ID, filename, quant tag, or custom repo:file) into a catalog entry."""
    if not choice:
        return catalog[0] if catalog else None
        
    choice_str = str(choice).strip()
    
    # 1. Match by numeric ID (e.g. 1, 2, "3")
    if choice_str.isdigit():
        target_id = int(choice_str)
        for item in catalog:
            if item.get("id") == target_id:
                return item
        raise ValueError(f"Model ID [{target_id}] out of range. Choose between 1 and {len(catalog)}.")

    # 2. Check for custom Hugging Face repo format: "repo_id:filename"
    if ":" in choice_str and "/" in choice_str:
        repo_part, file_part = choice_str.split(":", 1)
        repo_part, file_part = repo_part.strip(), file_part.strip()
        fname = os.path.basename(file_part)
        return {
            "id": 999,
            "filename": fname,
            "developer": repo_part.split("/")[0],
            "variant": "Custom Hugging Face",
            "quant": "Custom",
            "size_gb": 0.0,
            "status": "Hugging Face",
            "source": "Hugging Face (Custom)",
            "repo": repo_part,
            "subpath": file_part,
            "revision": "main",
            "notes": f"Custom user model from {repo_part}"
        }

    # 3. Match by exact or partial filename
    choice_lower = choice_str.lower()
    for item in catalog:
        if item["filename"].lower() == choice_lower:
            return item

    # 4. Match by quant tag (e.g. Q4_K_M, Q8_0, BF16)
    for item in catalog:
        if choice_lower in item["filename"].lower() or choice_lower == item.get("quant", "").lower():
            return item

    # 5. Fallback check in legacy MODELS dictionary
    for k, v in MODELS.items():
        if k.lower() == choice_lower:
            target_fname = os.path.basename(v[0])
            for item in catalog:
                if item["filename"].lower() == target_fname.lower():
                    return item
                    
    raise ValueError(f"Could not resolve model choice '{choice}'. Please provide a valid ID (1-{len(catalog)}), filename, or repo:filename.")


def ask_model_selection(catalog, default_choice="1"):
    """Interactively ask user which model to load, falling back to default if non-interactive."""
    print("\n" + format_model_catalog_table(catalog, as_html=False))
    prompt_msg = f"\n>> Enter model number to load [1-{len(catalog)}] (press Enter for default [{default_choice}]): "
    
    try:
        if sys.stdin and sys.stdin.isatty():
            user_input = input(prompt_msg).strip()
            if not user_input:
                user_input = default_choice
        else:
            print(f">> Non-interactive session: selecting default model [{default_choice}].")
            user_input = default_choice
    except Exception:
        user_input = default_choice

    resolved = resolve_model_choice(user_input, catalog)
    print(f">> Selected: [{resolved.get('id', '?')}] {resolved['filename']} ({resolved['developer']} - {resolved['quant']})")
    return resolved


# ---- Download & Sync Logic ----------------------------------------------------------

class LoraError(RuntimeError):
    pass


def _direct_url(url):
    """Hugging Face page links (…/blob/…) -> download links (…/resolve/…)."""
    return re.sub(r"^(https://huggingface\.co/[^/]+/[^/]+)/blob/", r"\1/resolve/", url.strip())


def _safe_name(name):
    name = os.path.basename(urllib.parse.unquote(name)).strip()
    return re.sub(r"[^\w.\-+()\[\] ]", "_", name)


def download_lora(url, comfy_dir="ComfyUI", name=None, token=None, progress=None):
    """Download a LoRA into ComfyUI/models/loras and return its file name."""
    url = _direct_url(url)
    if not url.startswith(("https://", "http://")):
        raise LoraError(f"not a URL: {url!r}")
    if token is None:
        token = os.environ.get("CIVITAI_TOKEN" if "civitai.com" in url else "HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    lora_dir = os.path.join(comfy_dir, "models", "loras")
    os.makedirs(lora_dir, exist_ok=True)

    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if r.status_code in (401, 403):
            raise LoraError(f"{r.status_code}: this link needs a token (Civitai API key / HF token)")
        if r.status_code != 200:
            raise LoraError(f"download failed: HTTP {r.status_code} for {url}")
        if "text/html" in r.headers.get("content-type", ""):
            raise LoraError("the link returned a web page, not a file — use direct download link")
        if not name:
            m = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)", r.headers.get("content-disposition", ""))
            name = m.group(1) if m else urllib.parse.urlparse(r.url).path
        name = _safe_name(name)
        if not name.lower().endswith(".safetensors"):
            raise LoraError(f"{name!r} is not a .safetensors file — only safetensors LoRAs are accepted")
        dst = os.path.join(lora_dir, name)
        total = int(r.headers.get("content-length") or 0)
        if os.path.isfile(dst) and total and os.path.getsize(dst) == total:
            print(f">> LoRA {name} already there")
            return name
        part, done = dst + ".part", 0
        with open(part, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    with open(part, "rb") as f:
        head = f.read(9)
    if len(head) < 9 or head[8:9] != b"{" or struct.unpack("<Q", head[:8])[0] > done:
        os.remove(part)
        raise LoraError(f"{name!r} is not a valid safetensors file")
    os.replace(part, dst)
    print(f">> LoRA {name} ({done / 1e6:.1f} MB) -> {lora_dir}")
    return name


def fetch(repo, filename, local_dir, revision="main"):
    path = hf_hub_download(repo_id=repo, filename=filename, local_dir=local_dir, revision=revision)
    print(f"   {path}  ({os.path.getsize(path) / 1e9:.2f} GB)")
    return path


def sync_file_with_drive(filename, repo, repo_subpath, local_target_dir, drive_target_dir=None, revision="main"):
    """Check Google Drive first:
      - If found in Drive: copy/link to local_target_dir
      - If not found in Drive: check local_target_dir; if missing, download from Hugging Face.
        Then save a copy into drive_target_dir for future sessions.
    """
    os.makedirs(local_target_dir, exist_ok=True)
    local_path = os.path.join(local_target_dir, filename)
    drive_path = os.path.join(drive_target_dir, filename) if drive_target_dir else None

    # 1. Check Google Drive first
    if drive_path and os.path.isfile(drive_path) and os.path.getsize(drive_path) > 1024:
        size_gb = os.path.getsize(drive_path) / 1e9
        print(f">> [Drive Cache Hit] Found {filename} ({size_gb:.2f} GB) in Drive: {drive_target_dir}")
        if not os.path.isfile(local_path) or os.path.getsize(local_path) != os.path.getsize(drive_path):
            print(f">> Copying from Drive to local fast storage: {local_path} ...")
            shutil.copy2(drive_path, local_path)
            print(f">> Copied to local storage.")
        else:
            print(f">> Local copy already up to date: {local_path}")
        return local_path

    # 2. Check local directory
    if os.path.isfile(local_path) and os.path.getsize(local_path) > 1024:
        print(f">> Local file ready: {local_path}")
        if drive_path and drive_target_dir and not os.path.isfile(drive_path):
            try:
                print(f">> Backing up local file to Drive ({drive_target_dir})...")
                os.makedirs(drive_target_dir, exist_ok=True)
                shutil.copy2(local_path, drive_path)
                print(f">> Backup copy saved to Drive.")
            except OSError as e:
                print(f"   (Warning: could not backup to Drive: {e})")
        return local_path

    # 3. Not in Drive, not local -> Download from Hugging Face
    if not repo:
        raise FileNotFoundError(f"File {filename} not found in Drive and no Hugging Face repo specified.")
        
    print(f">> [Hugging Face] {filename} not found in Drive or local storage. Downloading from {repo}@{revision}...")
    fetched = fetch(repo, repo_subpath, local_target_dir, revision=revision)
    if os.path.abspath(fetched) != os.path.abspath(local_path):
        if os.path.isfile(fetched) and not os.path.isfile(local_path):
            shutil.move(fetched, local_path)

    # 4. Save a copy to Google Drive for next time
    if drive_path and drive_target_dir:
        try:
            print(f">> Saving copy to Google Drive ({drive_target_dir})...")
            os.makedirs(drive_target_dir, exist_ok=True)
            shutil.copy2(local_path, drive_path)
            print(f">> Saved copy to Drive successfully.")
        except OSError as e:
            print(f"   (Warning: could not save copy to Drive: {e})")

    return local_path


def sync_selected_model(entry, drive_root="/content/drive/MyDrive/Qwen-Image", comfy_dir="ComfyUI",
                        text_encoder="int8", skip_vae=False):
    """Download/sync the selected model entry, matching Text Encoder, and VAE."""
    models_dir = os.path.join(comfy_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    drive_diff_dir = os.path.join(drive_root, "models", "Diffusion Models") if drive_root else None
    drive_text_encoder_dir = os.path.join(drive_root, "models", "Text Encoders") if drive_root else None
    drive_vae_dir = os.path.join(drive_root, "models", "VAE") if drive_root else None

    # 1. Sync Diffusion Model
    diff_filename = entry["filename"]
    local_diff_dir = os.path.join(models_dir, "diffusion_models")
    print(f"\n>> [1/3] Syncing Selected Diffusion Model: {diff_filename} ({entry.get('developer', 'Unknown')} | ~{entry.get('size_gb', 0)} GB)...")
    sync_file_with_drive(
        filename=diff_filename,
        repo=entry.get("repo"),
        repo_subpath=entry.get("subpath", diff_filename),
        local_target_dir=local_diff_dir,
        drive_target_dir=drive_diff_dir,
        revision=entry.get("revision", "main")
    )

    # 2. Sync Text Encoder
    if text_encoder:
        fname, repo, size = TEXT_ENCODERS.get(text_encoder, TEXT_ENCODERS["int8"])
        enc_filename = os.path.basename(fname)
        local_enc_dir = os.path.join(models_dir, "text_encoders")
        print(f"\n>> [2/3] Syncing Text Encoder: {enc_filename} (~{size} GB)...")
        sync_file_with_drive(
            filename=enc_filename,
            repo=repo,
            repo_subpath=fname,
            local_target_dir=local_enc_dir,
            drive_target_dir=drive_text_encoder_dir,
            revision="main"
        )

    # 3. Sync VAE
    if not skip_vae:
        vae_filename = os.path.basename(VAE)
        local_vae_dir = os.path.join(models_dir, "vae")
        print(f"\n>> [3/3] Syncing VAE: {vae_filename} (~0.68 GB)...")
        sync_file_with_drive(
            filename=vae_filename,
            repo=DEFAULT_REPO,
            repo_subpath=VAE,
            local_target_dir=local_vae_dir,
            drive_target_dir=drive_vae_dir,
            revision="main"
        )

    print("\n>> All required model components verified.")
    return diff_filename


def sync_models(model="Q4_K_M", text_encoder="int8", skip_vae=False,
                drive_root="/content/drive/MyDrive/Qwen-Image", comfy_dir="ComfyUI"):
    """Backward-compatible entrypoint: resolve model by name or ID and sync."""
    catalog = build_model_catalog(drive_root=drive_root, comfy_dir=comfy_dir, fetch_online=False)
    entry = resolve_model_choice(model, catalog)
    return sync_selected_model(
        entry=entry,
        drive_root=drive_root,
        comfy_dir=comfy_dir,
        text_encoder=text_encoder,
        skip_vae=skip_vae
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comfy-dir", default="ComfyUI", help="path to ComfyUI directory")
    ap.add_argument("--drive-root", default=None, help="Google Drive root directory (e.g. /content/drive/MyDrive/Qwen-Image)")
    ap.add_argument("--list-models", action="store_true", help="Print table of all available compatible models and exit")
    ap.add_argument("--interactive", action="store_true", help="Ask interactively which model to load")
    ap.add_argument("--select", help="Select model by ID number, filename, or 'repo_id:filename'")
    ap.add_argument("--model", "--quant", dest="model", default="Q4_K_M", help="Legacy model name or quant tag (e.g. Q4_K_M, Q8_0)")
    ap.add_argument("--text-encoder", default="int8", choices=sorted(TEXT_ENCODERS))
    ap.add_argument("--skip-text-encoder", action="store_true")
    ap.add_argument("--skip-vae", action="store_true")
    ap.add_argument("--lora", action="append", default=[], metavar="URL", help="download LoRA (.safetensors)")
    ap.add_argument("--lora-token", help="API token for LoRAs")
    ap.add_argument("--lora-only", action="store_true")
    args = ap.parse_args()

    models_dir = os.path.join(args.comfy_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    for url in args.lora:
        try:
            download_lora(url, args.comfy_dir, token=args.lora_token)
        except (LoraError, requests.RequestException) as e:
            sys.exit(f">> LoRA download failed: {e}")
    if args.lora_only:
        return

    catalog = build_model_catalog(drive_root=args.drive_root, comfy_dir=args.comfy_dir, fetch_online=False)

    if args.list_models:
        print("\n" + format_model_catalog_table(catalog, as_html=False))
        return

    if args.interactive:
        selected_entry = ask_model_selection(catalog, default_choice="1")
    elif args.select:
        selected_entry = resolve_model_choice(args.select, catalog)
    else:
        selected_entry = resolve_model_choice(args.model, catalog)

    diff_file = sync_selected_model(
        entry=selected_entry,
        drive_root=args.drive_root,
        comfy_dir=args.comfy_dir,
        text_encoder=None if args.skip_text_encoder else args.text_encoder,
        skip_vae=args.skip_vae
    )
    print(f"\n>> Model `{diff_file}` is ready for generation.")


if __name__ == "__main__":
    main()
