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
# # 01 Frankfurter Exchange Rate API Request

# %% [markdown]
# Fetch the 2023 INR to EUR historical exchange rates from Frankfurter and save them as a CSV file for the analysis notebook.

# %%
import csv
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

# %% [markdown]
# ## Request 2023 INR to EUR Rates

# %%
params = {
    "from": "2023-01-01",
    "to": "2023-12-31",
    "base": "INR",
    "quotes": "EUR",
}

frankfurter_url = f"https://api.frankfurter.dev/v2/rates?{urlencode(params)}"
request = Request(
    frankfurter_url,
    headers={
        "Accept": "application/json",
        "User-Agent": "big-data-engineering-project/1.0",
    },
)

with urlopen(request, timeout=30) as response:
    payload = json.load(response)

print(f"Frankfurter URL: {frankfurter_url}")

# %% [markdown]
# ## Normalize API Response

# %%
if not isinstance(payload, list):
    raise ValueError(f"Unexpected Frankfurter response shape: {type(payload).__name__}")

exchange_rate_rows = [
    {
        "date": row["date"],
        "base": row["base"],
        "quote": row["quote"],
        "rate": float(row["rate"]),
    }
    for row in payload
    if row.get("base") == "INR"
    and row.get("quote") == "EUR"
    and row.get("rate") is not None
    and "2023-01-01" <= row.get("date", "") <= "2023-12-31"
]

if not exchange_rate_rows:
    raise ValueError("No INR -> EUR exchange rates were returned by Frankfurter for 2023.")

exchange_rate_rows = sorted(exchange_rate_rows, key=lambda row: row["date"])
average_rate = sum(row["rate"] for row in exchange_rate_rows) / len(exchange_rate_rows)

print(f"2023 INR -> EUR observations: {len(exchange_rate_rows)}")
print(f"Average 2023 INR -> EUR rate: {average_rate:.8f}")

# %% [markdown]
# ## Save Exchange Rates as CSV

# %%
output_file = Path("data/frankfurter-inr-eur-exchange-rates-2023.csv")

with output_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["date", "base", "quote", "rate"])
    writer.writeheader()
    writer.writerows(exchange_rate_rows)

print(f"Exchange rates saved to {output_file}")
