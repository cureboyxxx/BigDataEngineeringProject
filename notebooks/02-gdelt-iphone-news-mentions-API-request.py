# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 02 GDELT iPhone News Mentions API Request

# %% [markdown]
# Query the GDELT DOC API for a fixed set of iPhone models and save the 2023 mention counts as a CSV file for the analysis notebook.

# %%
from pathlib import Path
import time

import pandas as pd
import requests

# %% [markdown]
# ## Request 2023 News Mentions

# %%
iphone_models_for_gdelt = [
    "iPhone 12",
    "iPhone 13",
    "iPhone 14",
    "iPhone 14 Plus",
    "iPhone 14 Pro",
    "iPhone 14 Pro Max",
]

gdelt_start_datetime = "20230101000000"
gdelt_end_datetime = "20231231235959"


def extract_gdelt_timeline_total(payload):
    total_mentions = 0.0
    for timeline in payload.get("timeline", []):
        for point in timeline.get("data", []):
            value = point.get("value")
            if value is not None:
                total_mentions += float(value)
    return int(round(total_mentions))


def fetch_gdelt_mentions_2023(model_name, sleep_seconds=6):
    time.sleep(sleep_seconds)
    gdelt_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": f'"{model_name}"',
        "mode": "timelinevolraw",
        "format": "json",
        "startdatetime": gdelt_start_datetime,
        "enddatetime": gdelt_end_datetime,
        "timelinesmooth": "0",
    }

    try:
        response = requests.get(gdelt_url, params=params, timeout=20)
    except requests.RequestException as error:
        return {
            "model_name": model_name,
            "gdelt_mentions_2023": None,
            "gdelt_query_url": gdelt_url,
            "gdelt_fetch_status": f"request_error_{type(error).__name__}",
        }

    if response.status_code != 200:
        return {
            "model_name": model_name,
            "gdelt_mentions_2023": None,
            "gdelt_query_url": response.url,
            "gdelt_fetch_status": f"http_{response.status_code}",
        }

    try:
        payload = response.json()
    except ValueError:
        return {
            "model_name": model_name,
            "gdelt_mentions_2023": None,
            "gdelt_query_url": response.url,
            "gdelt_fetch_status": "invalid_json",
        }

    return {
        "model_name": model_name,
        "gdelt_mentions_2023": extract_gdelt_timeline_total(payload),
        "gdelt_query_url": response.url,
        "gdelt_fetch_status": "ok",
    }


gdelt_mentions_pd = pd.DataFrame([
    fetch_gdelt_mentions_2023(model_name)
    for model_name in iphone_models_for_gdelt
])

display(gdelt_mentions_pd)

# %% [markdown]
# ## Save GDELT Mentions as CSV

# %%
output_file = Path("gdelt-iphone-news-mentions-2023.csv")
gdelt_mentions_pd.to_csv(output_file, index=False)

print(f"GDELT mentions saved to {output_file}")
