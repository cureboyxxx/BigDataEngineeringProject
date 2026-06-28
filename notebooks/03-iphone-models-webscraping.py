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
# # 03 iPhone Models Web Scraping

# %% [markdown]
# Scrape iPhone generation metadata from Wikipedia, normalize split generation names, and save the result as a CSV file for the analysis notebook.

# %%
import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# %% [markdown]
# ## Scrape iPhone Generations

# %%
url = "https://en.wikipedia.org/wiki/IPhone"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TechShop_UniProject_Scraper/1.0"
}

response = requests.get(url, headers=headers, timeout=30)
response.raise_for_status()
print(f"Downloaded iPhone reference page: HTTP {response.status_code}")

soup = BeautifulSoup(response.text, "html.parser")
tables = soup.find_all("table", class_="wikitable")
target_table = tables[1]

iphone_model_rows = []
for table_row in target_table.find_all("tr")[3:]:
    row_values = [
        cell.get_text(" ", strip=True)
        for cell in table_row.find_all(["th", "td"])
    ]
    if len(row_values) >= 4 and row_values[0].lower().startswith("iphone"):
        iphone_model_rows.append(row_values[:9])

iphone_model_columns = [
    "model_name",
    "initial_os",
    "release_date",
    "discontinued_date",
    "support_ended",
    "final_os",
    "support_lifespan_max",
    "support_lifespan_min",
    "support_status",
]

normalized_iphone_model_rows = [
    row + [None] * (len(iphone_model_columns) - len(row))
    for row in iphone_model_rows
]

iphone_models_pd = pd.DataFrame(normalized_iphone_model_rows, columns=iphone_model_columns)
iphone_models_pd["release_year"] = (
    iphone_models_pd["release_date"]
    .str.extract(r"(\d{4})-\d{2}-\d{2}", expand=False)
    .astype("Int64")
)
iphone_models_pd["release_date_iso"] = (
    iphone_models_pd["release_date"]
    .str.extract(r"(\d{4}-\d{2}-\d{2})", expand=False)
)
iphone_models_pd = iphone_models_pd.dropna(subset=["release_year"])
iphone_models_pd["release_year"] = iphone_models_pd["release_year"].astype(int)
iphone_models_pd["model_name"] = iphone_models_pd["model_name"].str.replace(r"\s+", " ", regex=True).str.strip()

# %% [markdown]
# ## Normalize Split Model Names

# %%
def expand_iphone_model_name(model_name):
    parts = [part.strip() for part in re.split(r"\s*/\s*", model_name)]
    if len(parts) == 1:
        return parts

    expanded_names = []
    for part in parts:
        if part.lower().startswith("iphone"):
            expanded_names.append(part)
        else:
            expanded_names.append(f"iPhone {part}")
    return expanded_names


expanded_iphone_model_rows = []
for _, row in iphone_models_pd.iterrows():
    for expanded_model_name in expand_iphone_model_name(row["model_name"]):
        expanded_row = row.copy()
        expanded_row["model_name"] = expanded_model_name
        expanded_iphone_model_rows.append(expanded_row)

iphone_models_pd = pd.DataFrame(expanded_iphone_model_rows)
iphone_models_pd = iphone_models_pd.drop_duplicates(subset=["model_name", "release_year"])

print(f"iPhone generations scraped: {len(iphone_models_pd)}")
display(iphone_models_pd.head(10))

# %% [markdown]
# ## Save iPhone Model Data

# %%
output_file = Path("wikipedia-iphone-models.csv")
iphone_models_pd.to_csv(output_file, index=False)

print(f"iPhone model data saved to {output_file}")
