"""
sweep.py

Score every crop in the dataset using the trained classifier, then
generate an HTML review board for fast labelling.

Usage:
    # Score everything, generate review page
    python sweep.py --dataset ./dataset --model police_classifier.pt

    # Only scan specific folders
    python sweep.py --dataset ./dataset --model police_classifier.pt --folders car truck

    # Lower threshold for review (default 0.5)
    python sweep.py --dataset ./dataset --model police_classifier.pt --threshold 0.3

    # After review, move confirmed filenames to police_confirmed/
    python sweep.py --dataset ./dataset --confirm confirmed.txt

Output:
    dataset/scores.csv   — every crop and its score
    dataset/review.html  — visual review board
"""

import os
import re
import csv
import json
import shutil
import base64
import argparse
import webbrowser
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# DEDUP
# ══════════════════════════════════════════════════════════════════════════════

def dedup_5min(paths):
    """Keep at most one image per camera per 5-minute window.

    Filename: JamCams_{cam1}_{cam2}_{YYYYMMDD}_{HHMMSS}_{ms}_{conf}pct_{hash}.jpg
    Bucket key: (cam1_cam2, date, hour, minute // 5)
    """
    seen = set()
    kept = []
    for path in paths:
        name = os.path.basename(path)
        m = re.match(r"JamCams_(\d+)_(\d+)_(\d{8})_(\d{6})_", name)
        if not m:
            kept.append(path)
            continue
        cam    = m.group(1) + "_" + m.group(2)
        date   = m.group(3)
        t      = m.group(4)
        bucket = (cam, date, t[0:2], int(t[2:4]) // 5)
        if bucket not in seen:
            seen.add(bucket)
            kept.append(path)
    return kept


def _cam_date_key(fname):
    """Return (cam_id, date_str) from a JamCam filename, or None if not parseable."""
    m = re.match(r"JamCams_(\d+)_(\d+)_(\d{8})_", os.path.basename(fname))
    if not m:
        return None
    return m.group(1) + "_" + m.group(2), m.group(3)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

def load_model(weights_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1),
    )

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def score_file(model, device, path):
    """Score a single image file. Returns probability [0, 1]."""
    try:
        img = Image.open(path).convert("RGB")
        tensor = _transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(tensor).squeeze()
            return round(torch.sigmoid(logit).item(), 4)
    except Exception:
        return -1.0


def score_files_batch(model, device, paths, batch_size=32):
    """Score a list of image paths in batches for GPU efficiency."""
    scores = [-1.0] * len(paths)
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        tensors = []
        valid_idx = []
        for j, p in enumerate(batch_paths):
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(_transform(img))
                valid_idx.append(j)
            except Exception:
                pass
        if tensors:
            batch = torch.stack(tensors).to(device)
            with torch.no_grad():
                logits = model(batch).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().tolist()
                if not isinstance(probs, list):
                    probs = [probs]
                for k, prob in zip(valid_idx, probs):
                    scores[i + k] = round(prob, 4)
    return scores


def score_crop(model, device, crop_bgr):
    """Score a BGR numpy array (from cv2). Returns probability [0, 1]."""
    try:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        tensor = _transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(tensor).squeeze()
            return round(torch.sigmoid(logit).item(), 4)
    except Exception:
        return -1.0


# ══════════════════════════════════════════════════════════════════════════════
# SCAN
# ══════════════════════════════════════════════════════════════════════════════

def scan_dataset(model, device, dataset_dir, folders=None, threshold=0.5):
    # Build set of filenames already in police_confirmed/ so we can skip them
    confirmed_dir = os.path.join(dataset_dir, "police_confirmed")
    confirmed_names = set()
    if os.path.isdir(confirmed_dir):
        confirmed_names = {f for f in os.listdir(confirmed_dir)
                           if f.lower().endswith((".jpg", ".jpeg", ".png"))}
    if confirmed_names:
        print(f"  [exclude] {len(confirmed_names)} already-confirmed files from police_confirmed/")

    if folders is None:
        folders = sorted([
            e for e in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, e))
            and e not in ("police_confirmed")
        ])

    results = []
    total = 0
    hits = 0

    for folder in folders:
        folder_path = os.path.join(dataset_dir, folder)
        if not os.path.isdir(folder_path):
            print(f"  [skip] {folder}")
            continue

        all_files = [f for f in os.listdir(folder_path)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))
                     and f not in confirmed_names]
        all_fpaths = [os.path.join(folder_path, f) for f in all_files]
        deduped    = dedup_5min(all_fpaths)
        files      = [os.path.basename(p) for p in deduped]

        # Delete files that didn't survive dedup
        deduped_set = set(deduped)
        deleted_n   = 0
        for p in all_fpaths:
            if p not in deduped_set:
                try:
                    os.remove(p)
                    deleted_n += 1
                except OSError:
                    pass

        print(f"  [{folder}] {len(files)}/{len(all_files)} after dedup"
              f" ({deleted_n} deleted)...", end=" ", flush=True)

        fpaths = [os.path.join(folder_path, f) for f in files]
        scores = score_files_batch(model, device, fpaths)

        folder_hits = 0
        for fname, s in zip(files, scores):
            fpath = os.path.join(folder_path, fname)
            total += 1
            results.append((s, folder, fname, fpath))
            if s >= threshold:
                hits += 1
                folder_hits += 1

        print(f"{folder_hits} above {threshold:.2f}")

    results.sort(key=lambda x: -x[0])
    print(f"\n  Total: {total} scanned, {hits} above {threshold:.2f}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# HTML REVIEW
# ══════════════════════════════════════════════════════════════════════════════

def _img_to_data_uri(path, max_dim=200):
    img = cv2.imread(path)
    if img is None:
        return ""
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        s = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode()}"


def generate_review_html(results, output_path, threshold=0.5, max_items=500, shown_fnames=None, confirm_url=None):
    above = [(s, fld, fn, fp) for s, fld, fn, fp in results if s >= threshold][:max_items]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    if shown_fnames is None:
        shown_fnames = [fn for _, _, fn, _ in above]
    shown_js = json.dumps(shown_fnames)
    confirm_url_js = json.dumps(confirm_url)  # "http://..." or null

    cards = []
    for s, folder, fname, fpath in above:
        uri = _img_to_data_uri(fpath)
        border = "#00ff88" if s >= 0.8 else "#ffaa00" if s >= 0.5 else "#666"
        cards.append(f"""
        <div class="card" data-fname="{fname}" data-score="{s}" style="border-color:{border}">
          <img src="{uri}" />
          <div class="info">
            <span class="score">{s:.3f}</span>
            <span class="folder">{folder}</span>
          </div>
          <div class="fname">{fname}</div>
          <label class="check"><input type="checkbox" value="{fname}" /> police</label>
        </div>""")

    all_scores = [s for s, _, _, _ in results]
    bins = {
        ">=0.9": sum(1 for s in all_scores if s >= 0.9),
        "0.7-0.9": sum(1 for s in all_scores if 0.7 <= s < 0.9),
        "0.5-0.7": sum(1 for s in all_scores if 0.5 <= s < 0.7),
        "0.3-0.5": sum(1 for s in all_scores if 0.3 <= s < 0.5),
        "<0.3": sum(1 for s in all_scores if s < 0.3),
    }
    stats_html = " | ".join(f"{k}: <b>{v}</b>" for k, v in bins.items())

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Police Review</title>
<style>
* {{ box-sizing: border-box; }}
body {{ margin:0; padding:20px; background:#0a0e14; color:#c8d8c0;
       font-family:'Courier New',monospace; font-size:13px; }}
h1 {{ color:#00ff88; font-size:18px; }}
.stats {{ color:#6a8a6a; margin-bottom:16px; font-size:12px; }}
.controls {{ position:sticky; top:0; z-index:10; background:#0a0e14;
             padding:10px 0 6px; border-bottom:1px solid #1a2e1f;
             display:flex; gap:10px; align-items:center; flex-wrap:wrap; }}
.controls button {{ background:#1a2e1f; border:1px solid #2a4a2f; color:#00ff88;
                    padding:6px 14px; cursor:pointer; font-family:inherit; font-size:12px; }}
.controls button:hover {{ background:#2a4a2f; }}
#output {{ width:100%; background:#111820; border:1px solid #1a2e1f; padding:8px;
           font-size:11px; color:#ffaa00; max-height:80px; overflow-y:auto;
           display:none; white-space:pre-wrap; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr));
         gap:10px; margin-top:14px; }}
.card {{ background:#111820; border:2px solid #333; border-radius:6px; overflow:hidden; }}
.card.selected {{ border-color:#00ff88 !important; box-shadow:0 0 12px rgba(0,255,136,0.3); }}
.card img {{ width:100%; display:block; cursor:pointer; image-rendering:pixelated; }}
.card .info {{ display:flex; justify-content:space-between; padding:4px 8px; }}
.card .score {{ color:#00ff88; font-weight:bold; }}
.card .folder {{ color:#6a8a6a; font-size:11px; }}
.card .fname {{ padding:0 8px 2px; font-size:9px; color:#4a6a4a;
                white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.card .check {{ display:block; padding:6px 8px; font-size:11px; color:#6a8a6a;
                cursor:pointer; border-top:1px solid #1a2e1f; }}
</style></head><body>
<h1>Police Vehicle Review</h1>
<div class="stats">{now} | {len(results)} total | {len(above)} shown (>={threshold:.2f}) | {stats_html}</div>
<div class="controls">
  <button onclick="selectAbove(0.8)">Select >=0.8</button>
  <button onclick="selectAbove(0.5)">Select >=0.5</button>
  <button onclick="selectAll()">Select all</button>
  <button onclick="selectNone()">Clear</button>
  <button onclick="exportSelected()">Export filenames</button>
  <span id="countLabel" style="color:#6a8a6a;font-size:12px">0 selected</span>
  <pre id="output"></pre>
</div>
<div class="grid">{"".join(cards)}</div>
<script>
function getCards(){{return document.querySelectorAll('.card')}}
function getChecked(){{return document.querySelectorAll('.card input:checked')}}
function updateCount(){{document.getElementById('countLabel').textContent=getChecked().length+' selected'}}
document.querySelectorAll('.card input').forEach(cb=>{{cb.addEventListener('change',function(){{
  this.closest('.card').classList.toggle('selected',this.checked);updateCount()}});}});
document.querySelectorAll('.card img').forEach(img=>{{img.addEventListener('click',function(){{
  const cb=this.closest('.card').querySelector('input');cb.checked=!cb.checked;
  cb.dispatchEvent(new Event('change'))}});}});
function selectAbove(t){{getCards().forEach(c=>{{const s=parseFloat(c.dataset.score);
  const cb=c.querySelector('input');cb.checked=s>=t;
  c.classList.toggle('selected',cb.checked)}});updateCount()}}
function selectAll(){{getCards().forEach(c=>{{c.querySelector('input').checked=true;
  c.classList.add('selected')}});updateCount()}}
function selectNone(){{getCards().forEach(c=>{{c.querySelector('input').checked=false;
  c.classList.remove('selected')}});updateCount()}}
const SHOWN={shown_js};
const CONFIRM_URL={confirm_url_js};
function _download(confirmed){{
  const blob=new Blob([confirmed.join('\\n')],{{type:'text/plain'}});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);
  a.download='confirmed.txt';document.body.appendChild(a);a.click();document.body.removeChild(a);
}}
async function exportSelected(){{
  const confirmed=[...getChecked()].map(cb=>cb.value);
  const out=document.getElementById('output');
  out.style.display='block';
  if(confirmed.length===0){{out.textContent='Nothing selected.';return;}}
  if(!CONFIRM_URL){{
    out.textContent='Downloaded confirmed.txt -- run: python sweep.py --confirm confirmed.txt (or re-run with --serve)';
    _download(confirmed);return;
  }}
  out.textContent='Sending to server...';
  try{{
    const r=await fetch(CONFIRM_URL,{{method:'POST',
      headers:{{'Content-Type':'application/json'}},
      body:JSON.stringify({{confirmed,shown:SHOWN}})}});
    if(!r.ok)throw new Error('HTTP '+r.status);
    const d=await r.json();
    out.textContent='confirmed='+d.confirmed+'  new='+d.appended+'  rejected='+d.rejected;
  }}catch(e){{
    out.textContent='Server error: '+e.message+' -- downloaded confirmed.txt';
    _download(confirmed);
  }}
}}
</script></body></html>"""

    with open(output_path, "w") as f:
        f.write(html)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIRM
# ══════════════════════════════════════════════════════════════════════════════

def confirm_crops(dataset_dir, confirmed_file, dest_folder="police_confirmed"):
    dest = os.path.join(dataset_dir, dest_folder)
    os.makedirs(dest, exist_ok=True)

    with open(confirmed_file) as f:
        fnames = {line.strip() for line in f if line.strip()}

    skip = {os.path.abspath(os.path.join(dataset_dir, d))
            for d in (dest_folder, "police_rejected")}
    moved = 0
    for root, _, files in os.walk(dataset_dir):
        if os.path.abspath(root) in skip:
            continue
        for fname in files:
            if fname in fnames:
                shutil.move(os.path.join(root, fname), os.path.join(dest, fname))
                moved += 1
                print(f"  + {fname}")

    print(f"\nMoved {moved}/{len(fnames)} to {dest}/")


def reject_crops(dataset_dir, shown_fnames, confirmed_fnames, dest_folder="police_rejected"):
    """Move everything shown in review but not confirmed to police_rejected/."""
    rejected = set(shown_fnames) - set(confirmed_fnames)
    if not rejected:
        print("  (no rejects)")
        return
    dest = os.path.join(dataset_dir, dest_folder)
    os.makedirs(dest, exist_ok=True)

    skip = {os.path.abspath(os.path.join(dataset_dir, d))
            for d in ("police_confirmed","police_rejected", dest_folder)}
    moved = 0
    for root, _, files in os.walk(dataset_dir):
        if os.path.abspath(root) in skip:
            continue
        for fname in files:
            if fname in rejected:
                shutil.move(os.path.join(root, fname), os.path.join(dest, fname))
                moved += 1
                print(f"  - {fname}")

    print(f"\nMoved {moved}/{len(rejected)} to {dest}/")


# ══════════════════════════════════════════════════════════════════════════════
# SERVE
# ══════════════════════════════════════════════════════════════════════════════

class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def serve_review(dataset_dir, html_path, confirmed_path, shown_fnames, port=5001, path_map=None):
    """Serve the review page and handle confirm/reject actions via HTTP."""
    shown_set = set(shown_fnames)

    def make_handler():
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass  # suppress access log noise

            def do_GET(self):
                if self.path == "/":
                    with open(html_path, "rb") as f:
                        body = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", len(body))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path == "/confirm":
                    try:
                        data = json.loads(
                            self.rfile.read(int(self.headers["Content-Length"]))
                        )
                        confirmed_fnames = set(data.get("confirmed", []))

                        # Append new names to confirmed.txt
                        existing = set()
                        if os.path.exists(confirmed_path):
                            with open(confirmed_path) as f:
                                existing = {l.strip() for l in f if l.strip()}
                        new_fnames = confirmed_fnames - existing

                        # Hard limit: 10 police images per camera per day in confirmed.txt
                        cam_date_counts = {}
                        for fname in existing:
                            key = _cam_date_key(fname)
                            if key:
                                cam_date_counts[key] = cam_date_counts.get(key, 0) + 1

                        allowed = []
                        capped  = 0
                        for fname in sorted(new_fnames):
                            key = _cam_date_key(fname)
                            if key:
                                if cam_date_counts.get(key, 0) >= 10:
                                    capped += 1
                                    continue
                                cam_date_counts[key] = cam_date_counts.get(key, 0) + 1
                            allowed.append(fname)

                        if capped:
                            print(f"  [cap] {capped} names dropped (10/camera/day limit)")

                        with open(confirmed_path, "a") as f:
                            for fname in allowed:
                                f.write(fname + "\n")
                        new_fnames = set(allowed)

                        # Move files — fast path uses pre-built path_map,
                        # slow path falls back to os.walk for --confirm mode
                        if path_map is not None:
                            confirmed_dest = os.path.join(dataset_dir, "police_confirmed")
                            rejected_dest  = os.path.join(dataset_dir, "police_rejected")
                            os.makedirs(confirmed_dest, exist_ok=True)
                            for fname in confirmed_fnames:
                                src = path_map.get(fname)
                                if src and os.path.exists(src):
                                    shutil.move(src, os.path.join(confirmed_dest, fname))
                            rejected = shown_set - confirmed_fnames
                            if rejected:
                                os.makedirs(rejected_dest, exist_ok=True)
                                for fname in rejected:
                                    src = path_map.get(fname)
                                    if src and os.path.exists(src):
                                        shutil.move(src, os.path.join(rejected_dest, fname))
                        else:
                            confirm_crops(dataset_dir, confirmed_path)
                            reject_crops(dataset_dir, shown_set, confirmed_fnames)

                        result = json.dumps({
                            "appended":  len(new_fnames),
                            "confirmed": len(confirmed_fnames),
                            "rejected":  len(shown_set - confirmed_fnames),
                        }).encode()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", len(result))
                        self.end_headers()
                        self.wfile.write(result)
                    except Exception as exc:
                        import traceback
                        traceback.print_exc()
                        result = json.dumps({"error": str(exc)}).encode()
                        self.send_response(500)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", len(result))
                        self.end_headers()
                        self.wfile.write(result)
                else:
                    self.send_response(404)
                    self.end_headers()

        return Handler

    httpd = _ThreadedHTTPServer(("", port), make_handler())
    print(f"\n[sweep] Serving review at http://localhost:{port}/")
    print("[sweep] Click 'Export filenames' in the page to confirm & reject.")
    print("[sweep] Press Ctrl+C to stop.\n")
    webbrowser.open(f"http://localhost:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    type=str, default="./dataset")
    parser.add_argument("--model",      type=str, default="police_classifier.pt")
    parser.add_argument("--folders",    type=str, nargs="*")
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--max-review", type=int, default=500)
    parser.add_argument("--confirm",    type=str, help="File of confirmed filenames to move")
    parser.add_argument("--serve",      action="store_true",
                        help="Start local review server after scanning (opens browser)")
    parser.add_argument("--port",       type=int, default=5001)
    args = parser.parse_args()

    if args.confirm:
        confirm_crops(args.dataset, args.confirm)
        # Also reject unconfirmed images from the last review session
        shown_path = os.path.join(args.dataset, "review_shown.txt")
        if os.path.exists(shown_path):
            with open(shown_path) as f:
                shown = {l.strip() for l in f if l.strip()}
            with open(args.confirm) as f:
                confirmed = {l.strip() for l in f if l.strip()}
            reject_crops(args.dataset, shown, confirmed)
        else:
            print("  [no review_shown.txt — skipping reject step]")
    else:
        print(f"[sweep] Loading {args.model}...")
        model, device = load_model(args.model)

        print(f"[sweep] Scanning {args.dataset}...\n")
        results = scan_dataset(model, device, args.dataset, args.folders, args.threshold)

        csv_path   = os.path.join(args.dataset, "scores.csv")
        html_path  = os.path.join(args.dataset, "review.html")
        shown_path = os.path.join(args.dataset, "review_shown.txt")

        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["score", "folder", "filename", "path"])
            for s, fld, fn, fp in results:
                w.writerow([s, fld, fn, fp])
        print(f"\n  Scores -> {csv_path}")

        shown_fnames = [fn for s, fld, fn, fp in results if s >= args.threshold][:args.max_review]
        with open(shown_path, "w") as f:
            f.write("\n".join(shown_fnames))

        confirm_url = f"http://localhost:{args.port}/confirm" if args.serve else None
        generate_review_html(results, html_path, args.threshold, args.max_review,
                             shown_fnames=shown_fnames, confirm_url=confirm_url)
        print(f"  Review -> {html_path}")

        top = [(s, fld, fn) for s, fld, fn, _ in results if s >= args.threshold][:20]
        if top:
            print(f"\n  Top candidates:")
            for s, fld, fn in top:
                print(f"    {s:.3f}  [{fld}]  {fn}")

        if args.serve:
            confirmed_path = os.path.join(args.dataset, "..", "confirmed.txt")
            confirmed_path = os.path.normpath(confirmed_path)
            path_map = {fn: fp for _, _, fn, fp in results}
            serve_review(args.dataset, html_path, confirmed_path, shown_fnames, args.port,
                         path_map=path_map)
