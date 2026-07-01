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
# #### **The notebooks should be executed in order. Alternatively the notebooks 01 - 04 can be skipped because the data from the API requests is also provided in the csv files in the folder "data".**

# %% [markdown]
# # Do iPhone Models with More News Attention Have Worse Amazon Rating Scores?

# %% [markdown]
# This notebook tells a simple data story with three sources:
#
# - Amazon Electronics listings provide product titles, star ratings, review counts, and prices.
# - GDELT provides a 2023 news-attention signal for each iPhone model.
# - Wikipedia provides model release years, which help separate newer launch-cycle phones from older long-tail phones.
#
# Research question: Do iPhone models with more news attention have worse Amazon rating scores?
#
# The purpose of this university informatics project is to show a clear technology pipeline and a presentable story. The score below is intentionally transparent instead of statistically complex.
#
# `rating_score = weighted_average_rating * review_confidence`
#
# where:
#
# - `weighted_average_rating` gives more influence to listings with more reviews: `log(1 + no_of_ratings)`.
# - `review_confidence` penalizes model groups with very little review evidence: `1 - exp(-total_reviews / 500)`.
#
# A model can therefore have a high raw rating but a lower rating score if it appears in only a few weakly reviewed listings. GDELT mentions are treated as media attention, not as a direct measure of product quality.

# %% [markdown]
# ## Import Libraries

# %%
import math
import os
import re
import sys

spark_home = os.environ.get("SPARK_HOME", "/usr/local/spark")
spark_python_path = os.path.join(spark_home, "python")
py4j_zip_path = os.path.join(spark_python_path, "lib", "py4j-0.10.9.7-src.zip")

for path in [spark_python_path, py4j_zip_path]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# %% [markdown]
# ## Spark Context and Session

# %%
spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("iphone-amazon-rating-news-attention")
    .config("spark.ui.showConsoleProgress", "false")
    .getOrCreate()
)

sc = spark.sparkContext
sc.setLogLevel("ERROR")

print(sc.version)
print(sc.master)

# %% [markdown]
# ## Chart Style

# %%
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.figsize": (11, 6),
    "axes.titlesize": 15,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 120,
})

def add_bar_labels(ax, fmt="{:.2f}", padding=3):
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, padding=padding, fontsize=8)

# %% [markdown]
# ## Load Input Data with PySpark

# %%
cleaned_data_file = "data/cleaned-amazon-electronics-sales-2023.csv"
gdelt_file = "data/gdelt-iphone-news-mentions-2023.csv"
wiki_file = "data/wikipedia-iphone-models.csv"

for required_file in [cleaned_data_file, gdelt_file, wiki_file]:
    if not os.path.exists(required_file):
        raise FileNotFoundError(f"{required_file} was not found.")

cleaned_schema = StructType([
    StructField("name", StringType(), True),
    StructField("main_category", StringType(), True),
    StructField("sub_category", StringType(), True),
    StructField("image", StringType(), True),
    StructField("link", StringType(), True),
    StructField("ratings", DoubleType(), True),
    StructField("no_of_ratings", IntegerType(), True),
    StructField("discount_price", DoubleType(), True),
    StructField("actual_price", DoubleType(), True),
])

gdelt_schema = StructType([
    StructField("model_name", StringType(), True),
    StructField("gdelt_mentions_2023", DoubleType(), True),
    StructField("gdelt_query_url", StringType(), True),
    StructField("gdelt_fetch_status", StringType(), True),
])

wiki_schema = StructType([
    StructField("model_name", StringType(), True),
    StructField("initial_os", StringType(), True),
    StructField("release_date", StringType(), True),
    StructField("discontinued_date", StringType(), True),
    StructField("support_ended", StringType(), True),
    StructField("final_os", StringType(), True),
    StructField("support_lifespan_max", StringType(), True),
    StructField("support_lifespan_min", StringType(), True),
    StructField("support_status", StringType(), True),
    StructField("release_year", IntegerType(), True),
])

amazon_df = spark.read.option("header", "true").schema(cleaned_schema).csv(cleaned_data_file)
gdelt_df = spark.read.option("header", "true").schema(gdelt_schema).csv(gdelt_file)
wiki_df = spark.read.option("header", "true").schema(wiki_schema).csv(wiki_file)

print("Amazon rows:", amazon_df.count())
print("GDELT rows:", gdelt_df.count())
print("Wikipedia rows:", wiki_df.count())

# %% [markdown]
# ## Extract Real iPhone Handset Listings

# %% [markdown]
# The Amazon Electronics file contains many accessories. The extraction below keeps titles that look like actual iPhone handsets and rejects obvious cases such as cases, covers, cables, chargers, protectors, adapters, and renewed-store accessories.

# %%
wiki_models = (
    wiki_df
    .select("model_name", "release_year", "release_date", "support_status")
    .where(F.col("model_name").rlike("^iPhone"))
    .toPandas()
    .sort_values(["release_year", "model_name"], ascending=[False, False])
)

model_names = wiki_models["model_name"].tolist()
model_names[:10]

# %%
accessory_terms = [
    "case", "cover", "protector", "tempered", "glass", "guard", "screen", "film",
    "charger", "charging", "cable", "adapter", "earphone", "headphone", "buds",
    "strap", "stand", "holder", "mount", "ring", "lens", "camera protector",
    "skin", "wallet", "sleeve", "back cover", "battery", "power bank",
]

condition_terms = ["renewed", "refurbished", "used", "pre-owned", "unlocked"]


def title_matches_model(title, model):
    title_norm = re.sub(r"[^a-z0-9]+", " ", str(title).lower()).strip()
    model_norm = re.sub(r"[^a-z0-9]+", " ", str(model).lower()).strip()
    return bool(re.search(rf"\b{re.escape(model_norm)}\b", title_norm))


def looks_like_accessory(title):
    title_norm = re.sub(r"[^a-z0-9]+", " ", str(title).lower()).strip()
    return any(re.search(rf"\b{re.escape(term)}\b", title_norm) for term in accessory_terms)


def looks_like_handset(title):
    title_norm = str(title).lower()
    has_storage = bool(re.search(r"\b(16|32|64|128|256|512)\s*gb\b|\b1\s*tb\b", title_norm))
    has_phone_word = "iphone" in title_norm
    return has_phone_word and (has_storage or any(term in title_norm for term in condition_terms))


def extract_model(title):
    if looks_like_accessory(title) or not looks_like_handset(title):
        return None
    for model in model_names:
        if title_matches_model(title, model):
            return model
    return None


amazon_pdf = amazon_df.toPandas()
amazon_pdf["model_name"] = amazon_pdf["name"].apply(extract_model)

iphone_listings = (
    amazon_pdf
    .dropna(subset=["model_name", "ratings", "no_of_ratings"])
    .query("ratings > 0 and no_of_ratings > 0")
    .copy()
)

iphone_listings["review_weight"] = np.log1p(iphone_listings["no_of_ratings"])
iphone_listings["rating_weighted"] = iphone_listings["ratings"] * iphone_listings["review_weight"]

print("Matched handset listings:", len(iphone_listings))
iphone_listings[["model_name", "ratings", "no_of_ratings", "discount_price", "name"]].head(10)

# %% [markdown]
# ## Build the Model-Level Rating Score

# %%
model_summary = (
    iphone_listings
    .groupby("model_name", as_index=False)
    .agg(
        listing_count=("name", "count"),
        average_rating=("ratings", "mean"),
        median_rating=("ratings", "median"),
        total_reviews=("no_of_ratings", "sum"),
        median_reviews=("no_of_ratings", "median"),
        median_discount_price=("discount_price", "median"),
        weighted_rating_sum=("rating_weighted", "sum"),
        review_weight_sum=("review_weight", "sum"),
    )
)

model_summary["weighted_average_rating"] = (
    model_summary["weighted_rating_sum"] / model_summary["review_weight_sum"]
)
model_summary["review_confidence"] = 1 - np.exp(-model_summary["total_reviews"] / 500)
model_summary["rating_score"] = (
    model_summary["weighted_average_rating"] * model_summary["review_confidence"]
)

analysis_df = (
    model_summary
    .merge(gdelt_df.toPandas()[["model_name", "gdelt_mentions_2023"]], on="model_name", how="left")
    .merge(wiki_models[["model_name", "release_year", "support_status"]], on="model_name", how="left")
)

analysis_df["gdelt_mentions_2023"] = analysis_df["gdelt_mentions_2023"].fillna(0)
analysis_df["log_news_attention"] = np.log1p(analysis_df["gdelt_mentions_2023"])
analysis_df["news_rank"] = analysis_df["gdelt_mentions_2023"].rank(ascending=False, method="dense")
analysis_df["rating_score_rank"] = analysis_df["rating_score"].rank(ascending=False, method="dense")
analysis_df["attention_group"] = pd.qcut(
    analysis_df["gdelt_mentions_2023"].rank(method="first"),
    q=min(3, len(analysis_df)),
    labels=["Low attention", "Medium attention", "High attention"],
)

analysis_df = analysis_df.sort_values("gdelt_mentions_2023", ascending=False).reset_index(drop=True)
analysis_df.round(3)

# %% [markdown]
# ## How the Attention Groups Are Defined

# %% [markdown]
# The attention groups are based on the 2023 GDELT mention count. The models are ranked from lowest to highest news attention and then split into three similarly sized groups with `pd.qcut`: Low attention, Medium attention, and High attention. These groups are only a simple storytelling aid for the charts, not fixed industry categories.
#
# pd.qcut is a pandas function that splits data into groups with roughly the same number of rows.
#
# We sort the values and divide them into 3 equal-sized groups:
#
# lowest third -> Low
#
# middle third -> Medium
#
# highest third -> High

# %% [markdown]
# ## Story Result in One Table

# %%
story_table = analysis_df[[
    "model_name",
    "release_year",
    "gdelt_mentions_2023",
    "listing_count",
    "total_reviews",
    "weighted_average_rating",
    "review_confidence",
    "rating_score",
    "news_rank",
    "rating_score_rank",
]].copy()

story_table["gdelt_mentions_2023"] = story_table["gdelt_mentions_2023"].astype(int)
story_table.round(3)

# %% [markdown]
# ## Correlation Snapshot

# %%
pearson_corr = analysis_df[["gdelt_mentions_2023", "log_news_attention", "rating_score", "weighted_average_rating"]].corr()
pearson_corr.round(3)

# %% [markdown]
# The story is easier to present as an attention-versus-rating contrast than as a formal hypothesis test. Do the models that dominate 2023 news attention land lower on the Amazon rating score axis?

# %% [markdown]
# ## Chart 1: What builds the rating score?

# %% [markdown]
# This chart breaks the rating score into its main parts. It makes clear that the final score depends on both rating quality and review confidence.

# %%
component_df = analysis_df.sort_values("rating_score", ascending=False).melt(
    id_vars=["model_name"],
    value_vars=["weighted_average_rating", "review_confidence", "rating_score"],
    var_name="metric",
    value_name="value",
)

plt.figure(figsize=(12, 7))
ax = sns.barplot(
    data=component_df,
    x="model_name",
    y="value",
    hue="metric",
    palette=["#4C78A8", "#F2C14E", "#E4572E"],
)
ax.set_title("What Builds the Rating Score?")
ax.set_xlabel("")
ax.set_ylabel("Metric value")
ax.tick_params(axis="x", rotation=45)
ax.legend(title="")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Chart 2: Rating Score vs Price

# %% [markdown]
# This chart adds price context to the story. It helps check whether the rating-score pattern is mainly about expensive or cheaper listings.

# %%
price_df = analysis_df.dropna(subset=["median_discount_price"]).copy()

plt.figure(figsize=(11, 7))
ax = sns.scatterplot(
    data=price_df,
    x="median_discount_price",
    y="rating_score",
    size="gdelt_mentions_2023",
    hue="attention_group",
    palette=["#5B8E7D", "#F2C14E", "#E4572E"],
    sizes=(100, 700),
    edgecolor="white",
    linewidth=0.8,
)
for _, row in price_df.iterrows():
    ax.text(row["median_discount_price"] + 2, row["rating_score"] + 0.02, row["model_name"], fontsize=8)
ax.set_title("Price Context: Rating Score Is Not Just a Cheap-Phone Story")
ax.set_xlabel("Median discount price in the Amazon data")
ax.set_ylabel("Amazon rating score")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Chart 3: News Attention by iPhone Model
#

# %% [markdown]
# This chart shows which iPhone models were mentioned most often in 2023 news data. It sets up the "attention" side of the research question.

# %%
attention_order = analysis_df.sort_values("gdelt_mentions_2023", ascending=True)

plt.figure(figsize=(11, 7))
ax = sns.barplot(
    data=attention_order,
    x="gdelt_mentions_2023",
    y="model_name",
    hue="attention_group",
    dodge=False,
    palette=["#5B8E7D", "#F2C14E", "#E4572E"],
)
ax.set_title("2023 News Attention by iPhone Model")
ax.set_xlabel("GDELT mentions in 2023")
ax.set_ylabel("")
ax.legend(title="")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Chart 4: Rating Score and News Attention by iPhone Model
#

# %% [markdown]
# This chart compares the calculated Amazon rating score for each model. It shows which models look strongest or weakest after review evidence is included.

# %%
rating_order = analysis_df.sort_values("rating_score", ascending=True)

plt.figure(figsize=(11, 7))
ax = sns.barplot(
    data=rating_order,
    x="rating_score",
    y="model_name",
    hue="attention_group",
    dodge=False,
    palette=["#5B8E7D", "#F2C14E", "#E4572E"],
)
ax.set_title("Amazon Rating Score by iPhone Model")
ax.set_xlabel("Rating score, 0 to 5")
ax.set_ylabel("")
ax.set_xlim(0, 5)
ax.legend(title="")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Results

# %% [markdown]
# The analysis shows that news attention and Amazon rating scores do not move together in a simple way. The most mentioned iPhone models are not automatically the models with the best Amazon rating scores.
#
# The main pattern is that newer and highly discussed models receive more news attention, but their Amazon rating score can still be weaker when the review evidence is less strong. Older or less discussed models can look better because they have more stable review histories and more accumulated customer feedback.
#
# For the research question, the simple answer is: more news attention does not mean better Amazon ratings. In this dataset, the story points slightly in the opposite direction: very visible iPhone models can have more mixed Amazon rating results.
#
# This does not prove that news attention causes worse ratings. It only shows that attention and customer satisfaction are different signals. News attention mostly reflects public discussion, launches, and media interest, while the Amazon rating score reflects customer reviews in the available product listings.

# %%
spark.stop()
