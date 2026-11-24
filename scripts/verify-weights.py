#!/usr/bin/env python3
"""Check a local copy of the ERNIE-5.5-Flash-Preview weights against the Hugging Face Hub's own file list.

For every file the Hub lists at the pinned revision: the file exists, its size matches, and (unless --size-only) its
sha256 matches the hash the Hub records for LFS files. Reads only; exits 1 on any mismatch.

    python3 -I scripts/verify-weights.py /path/to/ERNIE-5.5-Flash-Preview            # sizes + sha256 (166 GB: minutes)
    python3 -I scripts/verify-weights.py /path/to/ERNIE-5.5-Flash-Preview --size-only
    python3 -I scripts/verify-weights.py DIR --repo baidu/ERNIE-5.5-Flash-Preview --rev <commit>

Standard library only. -I keeps the current directory out of the import path, so a file in the weights directory
cannot shadow a module.
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request

DEFAULT_REPO = "baidu/ERNIE-5.5-Flash-Preview"
DEFAULT_REV = "de3ffd3e8d55ec0f68ca51cbcec52d2a6490cf3b"


def hub_files(repo, rev):
    url = f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as r:
        info = json.load(r)
    return info["sha"], info["siblings"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--rev", default=DEFAULT_REV)
    ap.add_argument("--size-only", action="store_true", help="skip the sha256 pass")
    a = ap.parse_args()

    sha, files = hub_files(a.repo, a.rev)
    print(f"{a.repo} @ {sha}: {len(files)} files, {sum(f.get('size') or 0 for f in files) / 1e9:.1f} GB on the Hub")
    bad, ok = [], 0
    for f in files:
        p = os.path.join(a.directory, f["rfilename"])
        if not os.path.isfile(p):
            bad.append(("missing", f["rfilename"]))
            continue
        if f.get("size") is not None and os.path.getsize(p) != f["size"]:
            bad.append(("size", f["rfilename"]))
            continue
        want = (f.get("lfs") or {}).get("sha256")
        if want and not a.size_only and sha256(p) != want:
            bad.append(("sha256", f["rfilename"]))
            continue
        ok += 1
    mode = "sizes" if a.size_only else "sizes + sha256"
    print(f"VERDICT {ok} of {len(files)} files ok ({mode}); bad={bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
