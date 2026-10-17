"""Downloads the PhysioNet/CinC 2019 training sets from PhysioNet's open-data
mirror on AWS S3 (s3://physionet-open, no credentials needed), verifying each
file's MD5 against its S3 ETag. Safe to re-run: verified files are skipped.

  python scripts/download_physionet2019.py
"""
import hashlib
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET = "https://physionet-open.s3.amazonaws.com"
PREFIX = "challenge-2019/1.0.0/training/"
DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "physionet2019"


def list_keys() -> list[tuple[str, str]]:
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": PREFIX}
        if token:
            q["continuation-token"] = token
        xml = urllib.request.urlopen(f"{BUCKET}/?{urllib.parse.urlencode(q)}", timeout=60).read().decode()
        keys += re.findall(r"<Key>([^<]+\.psv)</Key>.*?<ETag>&quot;([0-9a-f]{32})&quot;</ETag>", xml)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not m:
            return keys
        token = m.group(1)


def fetch(item: tuple[str, str]) -> str:
    key, md5 = item
    out = DEST / key[len(PREFIX):]
    if out.exists() and hashlib.md5(out.read_bytes()).hexdigest() == md5:
        return "skip"
    for attempt in range(8):
        try:
            data = urllib.request.urlopen(f"{BUCKET}/{key}", timeout=60).read()
        except OSError:                      # includes HTTP 5xx / throttling: back off and retry
            time.sleep(min(2 ** attempt, 30))
            continue
        if hashlib.md5(data).hexdigest() == md5:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            return "ok"
    return f"FAILED {key}"


def main() -> None:
    keys = list_keys()
    print(f"{len(keys)} files listed", flush=True)
    with ThreadPoolExecutor(64) as ex:
        results = list(ex.map(fetch, keys))
    failed = [r for r in results if r.startswith("FAILED")]
    print(f"downloaded {results.count('ok')}, already present {results.count('skip')}, failed {len(failed)}")
    if failed:
        print("\n".join(failed[:20]))
        sys.exit(1)


if __name__ == "__main__":
    main()
