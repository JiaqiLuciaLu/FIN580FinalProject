"""
Download data, reference code, and benchmark portfolios from Dropbox.
Run this script to fetch all project files into the correct directories.

Usage:
    python notebooks/download_data.py
"""

import os
import zipfile
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dropbox links — changing dl=0 to dl=1 forces direct download as zip
LINKS = {
    "data": {
        "url": "https://www.dropbox.com/scl/fo/z5r2qm0lsi1rnc3sop6mx/h?rlkey=zis71edhnmygr3a4ftqzeupco&e=1&dl=1",
        "dest": os.path.join(PROJECT_ROOT, "data", "raw"),
        "zip_name": "data_raw.zip",
    },
    "reference_code": {
        "url": "https://www.dropbox.com/scl/fo/6exb7jb9kctgxnosejwu4/h?rlkey=1t89bhr0pj1fwyhlvx6d9yw9c&e=1&dl=1",
        "dest": os.path.join(PROJECT_ROOT, "original"),
        "zip_name": "reference_code.zip",
    },
    "benchmark_portfolios": {
        "url": "https://www.dropbox.com/scl/fo/0c6j15c75a8va9cercl3r/h?dl=1&e=1",
        "dest": os.path.join(PROJECT_ROOT, "data", "benchmark"),
        "zip_name": "benchmark.zip",
    },
}


def download_and_extract(name, url, dest, zip_name):
    """Download a Dropbox folder as zip and extract to dest."""
    os.makedirs(dest, exist_ok=True)
    zip_path = os.path.join(dest, zip_name)

    print(f"\n[{name}] Downloading from Dropbox...")
    print(f"  URL: {url}")
    print(f"  Destination: {dest}")

    try:
        urllib.request.urlretrieve(url, zip_path)
        print(f"  Downloaded: {os.path.getsize(zip_path) / 1024:.1f} KB")
    except Exception as e:
        print(f"  ERROR downloading: {e}")
        return False

    # Try to extract if it's a valid zip
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            print(f"  Extracting {len(zf.namelist())} files...")
            zf.extractall(dest)
            for f in zf.namelist()[:20]:  # show first 20 files
                print(f"    - {f}")
            if len(zf.namelist()) > 20:
                print(f"    ... and {len(zf.namelist()) - 20} more")
        os.remove(zip_path)  # clean up zip
        print(f"  Done!")
        return True
    except zipfile.BadZipFile:
        print(f"  Not a zip file — keeping as-is at {zip_path}")
        # Might be a single file or HTML error page, check size
        size = os.path.getsize(zip_path)
        if size < 1000:
            with open(zip_path, "r", errors="ignore") as f:
                content = f.read(500)
            print(f"  Content preview: {content[:200]}")
            print("  WARNING: This looks like an error page, not actual data.")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Downloading project files from Dropbox")
    print("=" * 60)

    for name, info in LINKS.items():
        download_and_extract(name, info["url"], info["dest"], info["zip_name"])

    # Summary: list what ended up where
    print("\n" + "=" * 60)
    print("Summary of downloaded files:")
    print("=" * 60)
    for name, info in LINKS.items():
        dest = info["dest"]
        if os.path.exists(dest):
            files = []
            for root, dirs, filenames in os.walk(dest):
                for fn in filenames:
                    rel = os.path.relpath(os.path.join(root, fn), PROJECT_ROOT)
                    files.append(rel)
            print(f"\n[{name}] {dest}:")
            for f in sorted(files):
                print(f"  {f}")
            if not files:
                print("  (empty)")
