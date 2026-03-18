
import os
import re
import json
import time
import base64
import typing as t

import requests
import pandas as pd
from tenacity import retry, wait_random_exponential, stop_after_attempt
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
from openai import OpenAI
#set open ai key
OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxx"

INPUT_XLSX = "sunset_project_schema_with_flickr_cols.xlsx"
OUTPUT_XLSX = "sunset_scored.xlsx"
SHEET_NAME = 0  # can be sheet index (0) or name ("Sheet1")
SCORE_COL = 'beauty_score'
LINK_COL = 'flickr_url'

MODEL = "gpt-4o-mini"   # vision-capable model
SAVE_EVERY = 25         # save progress every N newly-processed rows
SLEEP_BETWEEN = 0.2     # pacing to reduce rate limits / bans
HEADLESS = True         # set False to watch the browser
PAGE_TIMEOUT_MS = 45_000
EXTRA_WAIT_MS = 1200    # after load, wait a bit more for final rendering

# Scoring rules:
MIN_SCORE = -1
MAX_SCORE = 10

# ======================
# Helpers
# ======================

OG_IMAGE_RE = re.compile(r'<meta\s+property="og:image"\s+content="([^"]+)"', re.I)

def looks_like_url(x: t.Any) -> bool:
    return isinstance(x, str) and x.startswith(("http://", "https://"))

def safe_str(x: t.Any) -> str:
    return "" if x is None else str(x)

def detect_mime_from_url(url: str) -> str:
    u = url.lower()
    if u.endswith(".png"):
        return "image/png"
    if u.endswith(".webp"):
        return "image/webp"
    if u.endswith(".gif"):
        return "image/gif"
    return "image/jpeg"

def bytes_to_data_url(img_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def fetch_html(url: str, timeout_s: int = 30) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=timeout_s)
    r.raise_for_status()
    return r.text

def extract_og_image_from_html(html: str) -> t.Optional[str]:
    m = OG_IMAGE_RE.search(html)
    return m.group(1) if m else None

def fetch_image_bytes(url: str, timeout_s: int = 30) -> bytes:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=timeout_s)
    r.raise_for_status()
    return r.content

def build_vision_instructions() -> str:
    return (
        "You are rating sunset photos.\n"
        "Return ONLY valid JSON: {\"score\": <int -1..10>, \"reason\": \"short\"}\n"
        "Rules:\n"
        "- If the photo does NOT feature a sunset / sky colors (e.g., indoor, no sky), score = -1.\n"
        "- Otherwise score 1..10 based on beauty: colors, clouds, visibility of the sky.\n"
        "- Be consistent. Keep reason short.\n"
    )

def extract_output_text_robust(resp) -> str:
    """
    The OpenAI SDK sometimes provides resp.output_text as a convenience,
    but it may not exist depending on SDK version.
    This tries output_text first, then falls back to scanning resp.output blocks.
    """
    if hasattr(resp, "output_text") and isinstance(getattr(resp, "output_text"), str):
        return resp.output_text

    # Fallback: scan output blocks for text
    try:
        out = getattr(resp, "output", None)
        if not out:
            return ""
        parts = []
        for item in out:
            # many items have .content which is a list of blocks
            content = getattr(item, "content", None)
            if not content:
                continue
            for block in content:
                # text blocks typically have .text
                txt = getattr(block, "text", None)
                if isinstance(txt, str) and txt.strip():
                    parts.append(txt)
        return "\n".join(parts).strip()
    except Exception:
        return ""

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def score_image_with_openai(image_data_url: str) -> dict:
    """
    Calls the model with image input and asks for strict JSON back.
    Retries with exponential backoff on transient failures.
    """
    client = OpenAI()
    instructions = build_vision_instructions()

    resp = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instructions},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }
        ],
    )

    text = extract_output_text_robust(resp).strip()
    return json.loads(text)

def clamp_score(score: t.Any) -> int:
    try:
        s = int(score)
    except Exception:
        return -1
    if s < MIN_SCORE:
        return MIN_SCORE
    if s > MAX_SCORE:
        return MAX_SCORE
    return s

# ======================
# Browser extraction (fallback)
# ======================

def playwright_get_og_image_url(page, url: str) -> t.Optional[str]:
    """
    Open page, wait, then read <meta property="og:image">.
    """
    try:
        page.goto(url, wait_until="networkidle", timeout=PAGE_TIMEOUT_MS)
    except PWTimeout:
        # networkidle can be strict; try load
        page.goto(url, wait_until="load", timeout=PAGE_TIMEOUT_MS)

    page.wait_for_timeout(EXTRA_WAIT_MS)

    try:
        og = page.eval_on_selector('meta[property="og:image"]', "el => el.content")
        if looks_like_url(og):
            return og
    except Exception:
        pass

    # Extra fallback: pick largest visible-ish <img>
    try:
        imgs = page.query_selector_all("img")
        best_src = None
        best_area = 0.0
        for img in imgs:
            try:
                box = img.bounding_box()
                if not box:
                    continue
                area = float(box["width"]) * float(box["height"])
                src = img.get_attribute("src") or ""
                if area > best_area and looks_like_url(src):
                    best_area = area
                    best_src = src
            except Exception:
                continue
        if best_src:
            return best_src
    except Exception:
        pass

    return None

# ======================
# Main
# ======================

def main():
    df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)

    if LINK_COL not in df.columns:
        raise ValueError(f"LINK_COL='{LINK_COL}' not found in Excel headers. Columns are: {list(df.columns)}")

    if SCORE_COL not in df.columns:
        df[SCORE_COL] = pd.NA

    newly_processed = 0
    total = len(df)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        context = browser.new_context()
        page = context.new_page()

        for i in range(total):
            existing = df.at[i, SCORE_COL]
            if pd.notna(existing):
                continue  # resume-safe

            link = safe_str(df.at[i, LINK_COL]).strip()
            if not looks_like_url(link):
                df.at[i, SCORE_COL] = -1
                continue

            print(f"[{i+1}/{total}] opening: {link}")

            image_url = None

            # 1) Try fast path: requests GET HTML -> og:image
            try:
                html = fetch_html(link)
                image_url = extract_og_image_from_html(html)
            except Exception:
                image_url = None

            # 2) Fallback: real browser load -> og:image
            if not image_url or not looks_like_url(image_url):
                try:
                    image_url = playwright_get_og_image_url(page, link)
                except Exception as e:
                    print(f"   browser fallback failed: {e}")
                    image_url = None

            if not image_url or not looks_like_url(image_url):
                print("   could not find an image on page -> score -1")
                df.at[i, SCORE_COL] = -1
                newly_processed += 1
                continue

            # Download image and score it
            try:
                img_bytes = fetch_image_bytes(image_url)
                mime = detect_mime_from_url(image_url)
                data_url = bytes_to_data_url(img_bytes, mime)

                result = score_image_with_openai(data_url)
                score = clamp_score(result.get("score", -1))

                df.at[i, SCORE_COL] = score
                print(f"   image: {image_url}")
                print(f"   score: {score} | reason: {safe_str(result.get('reason','')).strip()[:120]}")

            except Exception as e:
                print(f"   scoring failed: {e} -> score -1")
                df.at[i, SCORE_COL] = -1

            newly_processed += 1

            if newly_processed % SAVE_EVERY == 0:
                df.to_excel(OUTPUT_XLSX, index=False)
                print(f"Saved progress ({newly_processed} new rows) -> {OUTPUT_XLSX}")

            time.sleep(SLEEP_BETWEEN)

        df.to_excel(OUTPUT_XLSX, index=False)
        print(f"Done. Final saved -> {OUTPUT_XLSX}")

        context.close()
        browser.close()

if __name__ == "__main__":
    main()

