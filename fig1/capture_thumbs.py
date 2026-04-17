#!/usr/bin/env python3
"""
Capture platform page screenshots for fig1c card thumbnails.
Output: ./thumbs/<name>.png
"""
import os
import time
from playwright.sync_api import sync_playwright

BASE_URL = "https://gutbiomedb.online"
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thumbs")
os.makedirs(OUT_DIR, exist_ok=True)

SHOTS = [
    ("browse_phenotype_real",   f"{BASE_URL}/phenotype"),
    ("browse_disease_real",     f"{BASE_URL}/disease"),
    ("browse_search_real",      f"{BASE_URL}/search"),
    ("browse_studies_real",     f"{BASE_URL}/projects"),
    ("browse_similarity_v7",    f"{BASE_URL}/similarity"),
    ("browse_metabolism_real",  f"{BASE_URL}/metabolic"),
    ("analytics_differential_v11", f"{BASE_URL}/differential"),
    ("analytics_meta_v11",      f"{BASE_URL}/meta-analysis"),
    ("analytics_network_v11",   f"{BASE_URL}/network"),
    ("analytics_lifecycle_v11", f"{BASE_URL}/lifecycle"),
    ("analytics_gbhi_v11",      f"{BASE_URL}/health-index"),
    ("export_download_v7",      f"{BASE_URL}/download"),
    ("export_about_v7",         f"{BASE_URL}/about"),
]

VIEWPORT = {"width": 1440, "height": 900}
CLIP     = {"x": 0, "y": 0, "width": 1440, "height": 900}

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport=VIEWPORT)
        page = ctx.new_page()

        for name, url in SHOTS:
            out = os.path.join(OUT_DIR, f"{name}.png")
            print(f"  -> {name} ...", end=" ", flush=True)
            try:
                page.goto(url, wait_until="networkidle", timeout=30000)
                time.sleep(1.5)
                page.screenshot(path=out, clip=CLIP)
                print("OK")
            except Exception as e:
                print(f"FAIL: {e}")

        browser.close()
    print(f"\nDone. Saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
