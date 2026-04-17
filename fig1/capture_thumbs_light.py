#!/usr/bin/env python3
"""
Capture platform UI screenshots with forced light-mode CSS injection.
Saves to ./thumbs/
"""
import os, time
from playwright.sync_api import sync_playwright

BASE  = "https://gutbiomedb.online"
OUT   = r"E:\tasks\screenshots\fig1\thumbs"
os.makedirs(OUT, exist_ok=True)

LIGHT_CSS = """
* { background-color: #ffffff !important; color: #1a1a2e !important;
    border-color: #dde3ec !important; }
body, #root, .app, main, .page-container, .container {
    background: #ffffff !important; }
.card, .panel, [class*="card"], [class*="panel"], [class*="box"] {
    background: #f4f7fb !important; border: 1px solid #dde3ec !important; }
nav, header, .navbar, [class*="nav"], [class*="header"] {
    background: #253d6e !important; color: white !important; }
nav *, header *, .navbar * { color: white !important; }
button { background: #253d6e !important; color: white !important; }
input, select { background: #f4f7fb !important; color: #1a1a2e !important; }
svg text { fill: #1a1a2e !important; }
"""

PAGES = [
    ("browse_phenotype_real",      f"{BASE}/phenotype",    {"x":0,"y":60,"width":1400,"height":760}),
    ("browse_disease_real",        f"{BASE}/disease",      {"x":0,"y":60,"width":1400,"height":760}),
    ("browse_search_real",         f"{BASE}/search",       {"x":0,"y":60,"width":1400,"height":760}),
    ("browse_studies_real",        f"{BASE}/projects",     {"x":0,"y":60,"width":1400,"height":760}),
    ("browse_similarity_v7",       f"{BASE}/similarity",   {"x":0,"y":60,"width":1400,"height":760}),
    ("browse_metabolism_real",     f"{BASE}/metabolic",    {"x":0,"y":60,"width":1400,"height":760}),
    ("analytics_differential_v11", f"{BASE}/differential", {"x":0,"y":60,"width":1400,"height":760}),
    ("analytics_meta_v11",         f"{BASE}/meta-analysis",{"x":0,"y":60,"width":1400,"height":760}),
    ("analytics_network_v11",      f"{BASE}/network",      {"x":0,"y":60,"width":1400,"height":760}),
    ("analytics_lifecycle_v11",    f"{BASE}/lifecycle",    {"x":0,"y":60,"width":1400,"height":760}),
    ("analytics_gbhi_v11",         f"{BASE}/health-index", {"x":0,"y":60,"width":1400,"height":760}),
    ("export_download_v7",         f"{BASE}/download",     {"x":0,"y":60,"width":1400,"height":760}),
    ("export_about_v7",            f"{BASE}/about",        {"x":0,"y":60,"width":1400,"height":760}),
]

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width":1400,"height":860},
                                   device_scale_factor=1.5)
        page = ctx.new_page()

        for name, url, clip in PAGES:
            out = os.path.join(OUT, f"{name}.png")
            print(f"  {name} ...", end=" ", flush=True)
            try:
                page.goto(url, wait_until="networkidle", timeout=35000)
                page.add_style_tag(content=LIGHT_CSS)
                time.sleep(2.5)
                page.screenshot(path=out, clip=clip)
                print("OK")
            except Exception as e:
                print(f"FAIL: {e}")

        browser.close()
    print(f"\nDone → {OUT}")

if __name__ == "__main__":
    main()
