#!/usr/bin/env python3
import argparse
import os
import re
import requests
import difflib
import torch
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from sentence_transformers import SentenceTransformer, util
from io import StringIO


# ======================================================
# Command-line argument parsing
# ======================================================
parser = argparse.ArgumentParser(description="Jenkins TAP Comparison Tool")
parser.add_argument("--base-url", required=True, help="Old Jenkins build URL")
parser.add_argument("--current-url", required=True, help="New Jenkins build URL")
parser.add_argument("--output", required=True, help="Output file path for comparison results")

args = parser.parse_args()
BASE_URL = args.base_url
CURRENT_URL = args.current_url
OUTPUT_FILE = args.output


# ======================================================
# Load Transformer Model (cached)
# ======================================================
print("[INFO] Loading ML model (MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')


# ======================================================
# Helper: Download TAP files from Jenkins
# ======================================================
def download_files(url, extension, download_folder):
    print(f"[INFO] Looking for '*.{extension}' files at {url}")

    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    try:
        response = requests.get(url)
        response.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    links = soup.find_all("a", href=True)

    downloaded = []

    for link in links:
        file_url = urljoin(url, link["href"])

        if file_url.lower().endswith(f".{extension.lower()}"):
            filename = os.path.basename(file_url)
            file_path = os.path.join(download_folder, filename)

            print(f"[INFO] Downloading {filename}")

            try:
                with requests.get(file_url, stream=True) as r:
                    r.raise_for_status()
                    with open(file_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

                downloaded.append(file_path)

            except Exception as e:
                print(f"[ERROR] Failed to download {file_url}: {e}")

    return downloaded


# ======================================================
# Noise filter
# ======================================================
def is_noise_change(line):
    line = line.strip()
    if not line:
        return True
    if "TEST:" in line:
        return False
    return True


# ======================================================
# Log Parsing Helpers
# ======================================================
def parse_log_sections(text):
    sections = {}
    lines = text.splitlines()
    current_suite = None
    current_lines = []

    for line in lines:
        line = line.strip()

        m_old = re.search(r'(\S+)\s*-\s*Test results:', line)
        m_notok = re.match(r'not ok\s+\d+\s*-\s*(\S+)', line)

        if "- Test results:" in line:
            suite = m_old.group(1).strip() if m_old else line.split("- Test results:")[0].strip()

        elif m_notok:
            suite = m_notok.group(1).strip()

        else:
            if current_suite is not None:
                current_lines.append(line)
            continue

        if current_suite is not None:
            sections[current_suite] = current_lines

        current_suite = suite
        current_lines = [line] if m_notok else []

    if current_suite is not None:
        sections[current_suite] = current_lines

    return sections


def filter_ok_tests(lines):
    filtered = []
    skip_mode = False

    for line in lines:
        stripped = line.strip()

        if re.match(r'ok\s+\d+\s*-', stripped):
            skip_mode = True
            continue

        if re.match(r'not ok\s+\d+\s*-', stripped):
            skip_mode = False
            filtered.append(line)
            continue

        if not skip_mode:
            filtered.append(line)

    return filtered


def compare_sections(old_lines, new_lines, threshold_main=0.88, threshold_fallback=0.85, difflib_cutoff=0.87):
    old_lines = [ln.strip() for ln in old_lines if ln.strip()]
    new_lines = [ln.strip() for ln in new_lines if ln.strip()]

    if not old_lines or not new_lines:
        return new_lines.copy()

    emb_old = model.encode(old_lines, convert_to_tensor=True)
    emb_new = model.encode(new_lines, convert_to_tensor=True)

    sims = util.cos_sim(emb_new, emb_old)
    max_scores, max_indices = sims.max(dim=1)

    max_scores = max_scores.detach().cpu().numpy()
    max_indices = max_indices.detach().cpu().numpy()

    changes = []

    for i, score in enumerate(max_scores):
        new_line = new_lines[i]
        old_line = old_lines[max_indices[i]]

        if score < threshold_main:
            changes.append(new_line)

        elif score > threshold_fallback:
            ratio = difflib.SequenceMatcher(None, new_line, old_line).ratio()

            if ratio < difflib_cutoff:
                changes.append(f"{new_line} (difflib={ratio:.2f})")

    return changes


# ======================================================
# MAIN WORKFLOW
# ======================================================
print("[INFO] Downloading base TAP files...")
old_files = download_files(BASE_URL, "tap", "base_tap")

print("[INFO] Downloading current TAP files...")
new_files = download_files(CURRENT_URL, "tap", "current_tap")

if not old_files or not new_files:
    print("[ERROR] Missing TAP files. Aborting.")
    exit(1)

print("[INFO] Parsing and comparing logs...")
output = StringIO()

old_sections = {}
old_suite_sources = {}

for file_path in old_files:
    with open(file_path, "r") as f:
        text = f.read()

    parsed = parse_log_sections(text)

    for suite, lines in parsed.items():
        old_sections.setdefault(suite, []).extend(lines)
        old_suite_sources.setdefault(suite, set()).add(os.path.basename(file_path))

for file_path in new_files:
    output.write(f"===== NEW LOG: {os.path.basename(file_path)} =====\n")

    with open(file_path, "r") as f:
        text = f.read()

    new_sections = parse_log_sections(text)
    new_sections = {s: filter_ok_tests(l) for s, l in new_sections.items()}
    new_sections = {s: l for s, l in new_sections.items() if l}

    output.write("Parsed suites: " + ", ".join(new_sections.keys()) + "\n\n")

    for suite_name in new_sections:
        matched = None

        if suite_name in old_sections:
            matched = suite_name

        else:
            close = difflib.get_close_matches(suite_name, old_sections.keys(), n=1, cutoff=0.6)
            if close:
                matched = close[0]
                output.write(f"Fuzzy match: '{suite_name}' → '{matched}'\n")
            else:
                output.write(f"[NEW] Suite without match: {suite_name}\n")
                for ln in new_sections[suite_name]:
                    output.write(f"   + {ln}\n")
                output.write("\n")
                continue

        old_sources = ", ".join(sorted(old_suite_sources.get(matched, [])))
        output.write(f"Comparing suite '{suite_name}' (from {old_sources})\n")

        changes = compare_sections(old_sections[matched], new_sections[suite_name])

        noise = [c for c in changes if is_noise_change(c)]
        real = [c for c in changes if not is_noise_change(c)]

        if real:
            output.write("Real semantic changes:\n")
            for c in real:
                output.write(f"   - {c}\n")
        else:
            output.write("No meaningful test differences.\n")

        output.write("\n")


# ======================================================
# Write output
# ======================================================
print(f"[INFO] Writing results to {OUTPUT_FILE}")
with open(OUTPUT_FILE, "w") as f:
    f.write(output.getvalue())

print("[INFO] Comparison complete.")
