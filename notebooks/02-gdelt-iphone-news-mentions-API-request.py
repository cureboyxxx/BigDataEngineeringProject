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
output_file = Path("data/gdelt-iphone-news-mentions-2023.csv")
request_sleep_seconds = 30
max_request_attempts = 4
retry_sleep_seconds = 60


def extract_gdelt_timeline_total(payload):
    total_mentions = 0.0
    for timeline in payload.get("timeline", []):
        for point in timeline.get("data", []):
            value = point.get("value")
            if value is not None:
                total_mentions += float(value)
    return int(round(total_mentions))


def get_retry_wait_seconds(response, attempt_number):
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return max(int(retry_after), retry_sleep_seconds)
        except ValueError:
            pass

    return retry_sleep_seconds * attempt_number


def fetch_gdelt_mentions_2023(model_name):
    gdelt_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": f'"{model_name}"',
        "mode": "timelinevolraw",
        "format": "json",
        "startdatetime": gdelt_start_datetime,
        "enddatetime": gdelt_end_datetime,
        "timelinesmooth": "0",
    }

    last_error_status = None

    for attempt_number in range(1, max_request_attempts + 1):
        time.sleep(request_sleep_seconds)

        try:
            response = requests.get(gdelt_url, params=params, timeout=20)
        except requests.RequestException as error:
            last_error_status = f"request_error_{type(error).__name__}"
            if attempt_number == max_request_attempts:
                return {
                    "model_name": model_name,
                    "gdelt_mentions_2023": None,
                    "gdelt_query_url": gdelt_url,
                    "gdelt_fetch_status": last_error_status,
                }
            continue

        if response.status_code == 429:
            last_error_status = "http_429"
            if attempt_number == max_request_attempts:
                break

            wait_seconds = get_retry_wait_seconds(response, attempt_number)
            print(
                f"GDELT returned HTTP 429 for {model_name}; "
                f"waiting {wait_seconds} seconds before retry {attempt_number + 1}."
            )
            time.sleep(wait_seconds)
            continue

        if response.status_code >= 500:
            last_error_status = f"http_{response.status_code}"
            if attempt_number == max_request_attempts:
                break

            wait_seconds = retry_sleep_seconds * attempt_number
            print(
                f"GDELT returned HTTP {response.status_code} for {model_name}; "
                f"waiting {wait_seconds} seconds before retry {attempt_number + 1}."
            )
            time.sleep(wait_seconds)
            continue

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

    return {
        "model_name": model_name,
        "gdelt_mentions_2023": None,
        "gdelt_query_url": response.url if "response" in locals() else gdelt_url,
        "gdelt_fetch_status": f"{last_error_status}_after_{max_request_attempts}_attempts",
    }


if output_file.exists():
    existing_gdelt_mentions_pd = pd.read_csv(output_file)
else:
    existing_gdelt_mentions_pd = pd.DataFrame(
        columns=["model_name", "gdelt_mentions_2023", "gdelt_query_url", "gdelt_fetch_status"]
    )

existing_gdelt_mentions_pd["gdelt_mentions_2023"] = pd.to_numeric(
    existing_gdelt_mentions_pd["gdelt_mentions_2023"],
    errors="coerce",
)

usable_existing_rows = existing_gdelt_mentions_pd[
    (existing_gdelt_mentions_pd["model_name"].isin(iphone_models_for_gdelt))
    & (existing_gdelt_mentions_pd["gdelt_fetch_status"] == "ok")
    & (existing_gdelt_mentions_pd["gdelt_mentions_2023"].notna())
]
usable_existing_models = set(usable_existing_rows["model_name"])
models_to_fetch = [
    model_name
    for model_name in iphone_models_for_gdelt
    if model_name not in usable_existing_models
]

print(f"Reusing successful cached GDELT rows: {len(usable_existing_models)}")
print(f"Fetching GDELT rows: {models_to_fetch}")

fresh_gdelt_mentions_pd = pd.DataFrame([
    fetch_gdelt_mentions_2023(model_name)
    for model_name in models_to_fetch
])

gdelt_mentions_pd = (
    pd.concat([usable_existing_rows, fresh_gdelt_mentions_pd], ignore_index=True)
    .assign(
        model_name=lambda dataframe: pd.Categorical(
            dataframe["model_name"],
            categories=iphone_models_for_gdelt,
            ordered=True,
        )
    )
    .sort_values("model_name")
)
gdelt_mentions_pd["model_name"] = gdelt_mentions_pd["model_name"].astype(str)

display(gdelt_mentions_pd)

# %% [markdown]
# ## Save GDELT Mentions as CSV

# %%
gdelt_mentions_pd.to_csv(output_file, index=False)

print(f"GDELT mentions saved to {output_file}")
