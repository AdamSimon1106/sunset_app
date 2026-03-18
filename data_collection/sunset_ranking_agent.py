from __future__ import annotations

import pandas as pd
import requests
from bs4 import BeautifulSoup
import base64
import os
from datetime import datetime

from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback


# ================= CONFIG =================

PATH_TO_EXCEL = "sunset_project_schema_with_flickr_cols_rated.xlsx"
OUTPUT_EXCEL = "raw_sunsets_ranked.xlsx"
LINK_COLUMN = "flickr_url"
RANKING_COLUMN = "beauty_score"
OPENAI_MODEL = "gpt-4.1-mini"

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0,
    max_tokens=5
)

sunset_ranking_prompt = PromptTemplate(
    input_variables=[],
    template="""
You are an objective image evaluator.

Determine whether the image clearly shows a sunset or colorful sky.
If yes, rate its beauty from 1 to 10.
If not, return -1.

Rules:
- 10 = exceptional sunset
- 7-9 = clearly beautiful
- 4-6 = average
- 1-3 = weak
- -1 = no visible sunset

Return ONLY one integer.
No explanation.
"""
)


# ================= UTILITIES =================

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def image_bytes_to_data_url(image_bytes: bytes, mime_type="image/jpeg") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def extract_image_url_from_flickr(link: str) -> str | None:
    res = requests.get(link, headers={"User-Agent": "Mozilla/5.0"})
    if res.status_code != 200:
        return None

    soup = BeautifulSoup(res.content, "html.parser")
    img_tag = soup.find("img", class_="main-photo")

    if not img_tag:
        return None

    src = img_tag.get("src")
    if not src:
        return None

    if src.startswith("//"):
        src = "https:" + src

    return src


def fetch_image_bytes(image_url: str) -> bytes | None:
    res = requests.get(image_url)
    if res.status_code != 200:
        return None
    return res.content


def rank_image_with_llm(image_bytes: bytes):
    image_data = image_bytes_to_data_url(image_bytes)
    formatted_prompt = sunset_ranking_prompt.format()

    message = HumanMessage(content=[
        {"type": "text", "text": formatted_prompt},
        {"type": "image_url", "image_url": {"url": image_data}},
    ])

    with get_openai_callback() as cb:
        response = llm.invoke([message])

    score_text = response.content.strip()

    try:
        score = int(score_text)
    except:
        import re
        match = re.search(r"-?\d+", score_text)
        score = int(match.group(0)) if match else -1

    return score, cb.prompt_tokens, cb.completion_tokens, cb.total_cost


# ================= PROCESSING =================

def process_dataframe(df: pd.DataFrame):

    if RANKING_COLUMN not in df.columns:
        df[RANKING_COLUMN] = None

    total_cost = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_ranked = 0

    for idx, link in enumerate(df[LINK_COLUMN]):

        log(f"Processing row {idx}")

        try:
            image_url = extract_image_url_from_flickr(link)
            if not image_url:
                df.at[idx, RANKING_COLUMN] = -1
                log("No image URL found.")
                continue

            image_bytes = fetch_image_bytes(image_url)
            if not image_bytes:
                df.at[idx, RANKING_COLUMN] = -1
                log("Failed to fetch image bytes.")
                continue

            score, pt, ct, cost = rank_image_with_llm(image_bytes)

            df.at[idx, RANKING_COLUMN] = score

            total_cost += cost
            total_prompt_tokens += pt
            total_completion_tokens += ct
            total_ranked += 1

            log(f"Score={score} | request_cost=${cost:.6f} | session_total=${total_cost:.4f}")

        except Exception as e:
            df.at[idx, RANKING_COLUMN] = -1
            log(f"Error: {e}")

    return df, {
        "total_images_ranked": total_ranked,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_per_image": round(total_cost / total_ranked, 6) if total_ranked else 0
    }


# ================= ENTRY =================

def main():
    log("Starting ranking session")

    df = pd.read_excel(PATH_TO_EXCEL)

    df, tracking = process_dataframe(df)

    df.to_excel(OUTPUT_EXCEL, index=False)

    log("Finished ranking session")
    log(f"Total images ranked: {tracking['total_images_ranked']}")
    log(f"Total prompt tokens: {tracking['total_prompt_tokens']}")
    log(f"Total completion tokens: {tracking['total_completion_tokens']}")
    log(f"Total session cost: ${tracking['total_cost_usd']}")
    log(f"Average cost per image: ${tracking['avg_cost_per_image']}")
    log(f"Saved to: {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()