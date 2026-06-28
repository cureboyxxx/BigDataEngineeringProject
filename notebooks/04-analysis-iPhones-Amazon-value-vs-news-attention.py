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
# # 01 Cleaning the Amazon Electronics Sales Dataset with Spark

# %% [markdown]
# ## Import Libraries

# %%
import glob
import os
import re
import shutil
import sys
import time

spark_home = os.environ.get("SPARK_HOME", "/usr/local/spark")
spark_python_path = os.path.join(spark_home, "python")
py4j_zip_path = os.path.join(spark_python_path, "lib", "py4j-0.10.9.7-src.zip")

for path in [spark_python_path, py4j_zip_path]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    DecimalType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


# %% [markdown]
# ## Spark Context and Session
# Initialize Spark Context and Spark Session.

# %%
spark = (
    SparkSession
    .builder
    .master("local[*]")
    .appName("amazon-electronics-cleaning")
    .config("spark.ui.showConsoleProgress", "false")
    .getOrCreate()
)

sc = spark.sparkContext
sc.setLogLevel("ERROR")

print(sc.version)
print(sc.master)

# %% [markdown]
# ## Load Data from Amazon Electronics Products Sales CSV
# Load the raw CSV as strings first. This prevents Spark from guessing incorrect types before cleaning.

# %%
input_candidates = [
    "All Electronics.csv",
    "All Electronics(1).csv",
]

input_file = next((path for path in input_candidates if os.path.exists(path)), input_candidates[0])
print(f"Reading input file: {input_file}")

raw_schema = StructType([
    StructField("name", StringType(), True),
    StructField("main_category", StringType(), True),
    StructField("sub_category", StringType(), True),
    StructField("image", StringType(), True),
    StructField("link", StringType(), True),
    StructField("ratings", StringType(), True),
    StructField("no_of_ratings", StringType(), True),
    StructField("discount_price", StringType(), True),
    StructField("actual_price", StringType(), True),
])

raw_data = (
    spark.read
    .option("header", "true")
    .option("quote", '"')
    .option("escape", '"')
    .option("multiLine", "true")
    .option("mode", "PERMISSIVE")
    .schema(raw_schema)
    .csv(input_file)
)

raw_data.cache()
raw_data.show(5, truncate=False)

# %% [markdown]
# ## Understand the Data
# View the first few rows, check the shape, and inspect the raw schema.

# %%
print((raw_data.count(), len(raw_data.columns)))
raw_data.printSchema()


# %% [markdown]
# ## Handle Missing Data

# %%
def show_missing_counts(input_df, include_blank_strings=True):
    expressions = []
    for column in input_df.columns:
        is_missing = F.col(column).isNull()
        if include_blank_strings:
            is_missing = is_missing | (F.trim(F.col(column)) == "")
        expressions.append(F.count(F.when(is_missing, column)).alias(column))

    input_df.select(expressions).show(truncate=False)

show_missing_counts(raw_data)

# %% [markdown]
# Rows with missing `ratings` or `actual_price` are removed. Missing `discount_price` values are replaced by `actual_price` because those products are treated as not discounted.

# %%
data = raw_data.replace("", None)

data = data.dropna(subset=["ratings", "actual_price"])
data = data.withColumn("discount_price", F.coalesce(F.col("discount_price"), F.col("actual_price")))

show_missing_counts(data)
print((data.count(), len(data.columns)))

# %% [markdown]
# ## Handle Duplicates
# Count fully duplicated rows and remove them if any exist.

# %%
duplicate_count = data.count() - data.dropDuplicates().count()
print(f"Duplicate rows: {duplicate_count}")

data = data.dropDuplicates()

# %% [markdown]
# ## Handle Incorrect Data Types

# %% [markdown]
# ### Price
# Remove currency symbols and thousands separators, then convert prices to numeric values.

# %%
for column in ["discount_price", "actual_price"]:
    data = data.withColumn(
        column,
        F.regexp_replace(F.col(column), r"[^0-9.]", "").cast(DoubleType())
    )

# %% [markdown]
# ### Ratings
# Invalid values such as `Get` are converted to null by the numeric cast.

# %%
data.select("ratings").distinct().orderBy("ratings").show(50, truncate=False)

data = data.withColumn("ratings", F.col("ratings").cast(DoubleType()))

# %% [markdown]
# ### Number of Ratings
# Remove thousands separators before converting review counts to integers.

# %%
data = data.withColumn(
    "no_of_ratings",
    F.regexp_replace(F.col("no_of_ratings"), ",", "").cast(IntegerType())
)

# %% [markdown]
# Rows that still contain invalid numeric values in essential columns are removed.

# %%
data = data.dropna(subset=["ratings", "no_of_ratings", "discount_price", "actual_price"])

data.printSchema()
print((data.count(), len(data.columns)))


# %% [markdown]
# ## Convert INR prices to EUR
#
# Load the Frankfurter INR -> EUR historical rates created by `01-frankfurter-exchange-rate-API-request.ipynb`, calculate the average 2023 exchange rate, and convert both price columns to EUR.
#

# %%
exchange_rates_file = "frankfurter-inr-eur-exchange-rates-2023.csv"

if not os.path.exists(exchange_rates_file):
    raise FileNotFoundError(
        f"{exchange_rates_file} was not found. Run "
        "01-frankfurter-exchange-rate-API-request.ipynb before this analysis notebook."
    )

exchange_rate_schema = StructType([
    StructField("date", StringType(), True),
    StructField("base", StringType(), True),
    StructField("quote", StringType(), True),
    StructField("rate", DoubleType(), True),
])

exchange_rates = (
    spark.read
    .option("header", "true")
    .schema(exchange_rate_schema)
    .csv(exchange_rates_file)
    .filter(
        (F.col("base") == "INR")
        & (F.col("quote") == "EUR")
        & (F.col("rate").isNotNull())
        & (F.col("date").between("2023-01-01", "2023-12-31"))
    )
)

exchange_rate_summary = exchange_rates.agg(
    F.avg("rate").alias("average_rate"),
    F.count("rate").alias("observation_count"),
).first()

if exchange_rate_summary["observation_count"] == 0:
    raise ValueError(f"No valid INR -> EUR exchange rates found in {exchange_rates_file}.")

inr_to_eur_rate = float(exchange_rate_summary["average_rate"])
exchange_rate_observation_count = int(exchange_rate_summary["observation_count"])

print(f"Frankfurter CSV: {exchange_rates_file}")
print(f"2023 INR -> EUR observations: {exchange_rate_observation_count}")
print(f"Average 2023 INR -> EUR rate: {inr_to_eur_rate:.8f}")

price_columns = ["discount_price", "actual_price"]

for column in price_columns:
    data = data.withColumn(
        column,
        F.round(F.col(column) * F.lit(inr_to_eur_rate), 2)
    )

data.select(price_columns).show(5)


# %% [markdown]
# ## Handle Outliers

# %%
outlier_columns = ["discount_price", "actual_price", "no_of_ratings"]

data.select(outlier_columns + ["ratings"]).summary().show()

# %% [markdown]
# ### Print Outliers According to IQR Rule
# The outlier counts are reported, but the rows are not removed because electronics prices can vary widely across product types.

# %%
for column in outlier_columns:
    q1, q3 = data.approxQuantile(column, [0.25, 0.75], 0.01)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_count = data.filter(
        (F.col(column) < lower_bound) | (F.col(column) > upper_bound)
    ).count()

    print(f"{column}: {outlier_count} outliers")

# %% [markdown]
# ### Remove Impossible Values
# Ratings are valid only on the Amazon 1-5 scale. A discounted price should not be higher than the actual price; if this happens after cleaning, align the discount price with the actual price.

# %%
data = data.filter((F.col("ratings") >= 1) & (F.col("ratings") <= 5))

data = data.withColumn(
    "discount_price",
    F.when(F.col("discount_price") > F.col("actual_price"), F.col("actual_price"))
    .otherwise(F.col("discount_price"))
)

data.select(outlier_columns + ["ratings"]).summary().show()

# %% [markdown]
# ## Final Data Formatting
# Set the final column order and data types before writing the Parquet file. Price columns use `decimal(10,2)` so EUR values keep exactly two decimal places in the output schema.

# %%
string_columns = ["name", "main_category", "sub_category", "image", "link"]

for column in string_columns:
    data = data.withColumn(column, F.trim(F.col(column)))

data = data.select(
    F.col("name").cast(StringType()).alias("name"),
    F.col("main_category").cast(StringType()).alias("main_category"),
    F.col("sub_category").cast(StringType()).alias("sub_category"),
    F.col("image").cast(StringType()).alias("image"),
    F.col("link").cast(StringType()).alias("link"),
    F.round(F.col("ratings"), 1).cast(DoubleType()).alias("ratings"),
    F.col("no_of_ratings").cast(IntegerType()).alias("no_of_ratings"),
    F.round(F.col("discount_price"), 2).cast(DecimalType(10, 2)).alias("discount_price"),
    F.round(F.col("actual_price"), 2).cast(DecimalType(10, 2)).alias("actual_price"),
)

data.printSchema()
data.show(5, truncate=False)
print((data.count(), len(data.columns)))

# %% [markdown]
# ## Save Cleaned Data as Parquet and CSV
# Spark normally writes Parquet and CSV datasets as folders. This cell writes one part for each format and moves each part to a single output file.

# %%
parquet_output_file = "cleaned-amazon-electronics-sales-2023.parquet"
csv_output_file = "cleaned-amazon-electronics-sales-2023.csv"
parquet_temporary_output_dir = f"{parquet_output_file}.tmp"
csv_temporary_output_dir = f"{csv_output_file}.tmp"

for path in [
    parquet_output_file,
    csv_output_file,
    parquet_temporary_output_dir,
    csv_temporary_output_dir,
]:
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)

(
    data
    .coalesce(1)
    .write
    .mode("overwrite")
    .parquet(parquet_temporary_output_dir)
)

parquet_part_files = glob.glob(os.path.join(parquet_temporary_output_dir, "part-*.parquet"))
if not parquet_part_files:
    raise FileNotFoundError("Spark did not create a Parquet part file.")

shutil.move(parquet_part_files[0], parquet_output_file)
shutil.rmtree(parquet_temporary_output_dir)

(
    data
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("header", "true")
    .csv(csv_temporary_output_dir)
)

csv_part_files = glob.glob(os.path.join(csv_temporary_output_dir, "part-*.csv"))
if not csv_part_files:
    raise FileNotFoundError("Spark did not create a CSV part file.")

shutil.move(csv_part_files[0], csv_output_file)
shutil.rmtree(csv_temporary_output_dir)

print(f"Cleaned data saved to {parquet_output_file}")
print(f"Cleaned data saved to {csv_output_file}")

# %% [markdown]
# ## Validate Saved Parquet Output
# Read the saved file back and verify the schema, null counts, and basic business rules.

# %%
saved_data = spark.read.parquet(parquet_output_file)

saved_data.printSchema()
print((saved_data.count(), len(saved_data.columns)))
saved_data.show(5, truncate=False)

saved_data.select([
    F.count(F.when(F.col(column).isNull(), column)).alias(column)
    for column in saved_data.columns
]).show(truncate=False)

invalid_row_count = saved_data.filter(
    (F.col("ratings") < 1) |
    (F.col("ratings") > 5) |
    (F.col("discount_price") > F.col("actual_price")) |
    F.col("ratings").isNull() |
    F.col("no_of_ratings").isNull() |
    F.col("discount_price").isNull() |
    F.col("actual_price").isNull()
).count()

print(f"Invalid rows in saved Parquet file: {invalid_row_count}")
price_range = saved_data.select(
    F.min("discount_price").alias("min_discount_price_eur"),
    F.max("discount_price").alias("max_discount_price_eur"),
    F.min("actual_price").alias("min_actual_price_eur"),
    F.max("actual_price").alias("max_actual_price_eur"),
)

print("Saved EUR price range:")
price_range.show(truncate=False)


# %% [markdown]
# ## Validate Saved CSV Output
# Read the saved CSV file back and verify the row and column counts.

# %%
saved_csv_data = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(csv_output_file)
)

saved_csv_data.printSchema()
print((saved_csv_data.count(), len(saved_csv_data.columns)))
saved_csv_data.show(5, truncate=False)

print(f"Saved CSV row count matches Parquet: {saved_csv_data.count() == saved_data.count()}")
print(f"Saved CSV columns match Parquet: {saved_csv_data.columns == saved_data.columns}")


# %% [markdown]
# # Do iPhone Models with More News Attention Have Worse Amazon Value Scores?
#
# The cleaned Amazon Electronics data gives price, ratings, and review volume. Wikipedia gives release dates and support context for every iPhone generation. GDELT adds a third angle: how often each iPhone model was mentioned in online news during 2023.
#
# The research question for this project is:
#
# **Do iPhone models with more news attention have worse Amazon value scores?**
#
# The Amazon Electronics dataset contains many iPhone-related accessories, but this analysis excludes them. Only listings that look like actual iPhone handsets are used in the research result.
#
# The Amazon dataset does not contain listing creation dates. To avoid giving older models an automatic advantage from having more time to collect reviews, this notebook makes one explicit assumption: every actual iPhone handset listing started on Amazon on the release date of the iPhone model named in the title and was observed through `2023-12-31`.
#
# For this notebook, "value" means a listing has a high rating, enough reviews per assumed listing year to make that rating credible, and a low price. The score is intentionally simple and transparent:
#
# `value_score = rating * log(1 + reviews_per_assumed_listing_year) / discount_price_eur`
#
# The score is not a financial recommendation. It is a data-driven way to compare observed Amazon listings in this dataset.
#
# GDELT news mentions are treated as a rough media-attention signal, not as a direct measure of sales or demand.

# %% [markdown]
# ## Project Setup: Analysis DataFrame
#
# The analysis starts from the validated parquet-backed Spark DataFrame created above. This avoids brittle CSV parsing and keeps the rest of the project in Spark.

# %%
products_df = saved_data.select(
    F.monotonically_increasing_id().alias("product_id"),
    F.col("name").cast(StringType()).alias("name"),
    F.col("main_category").cast(StringType()).alias("main_category"),
    F.col("sub_category").cast(StringType()).alias("sub_category"),
    F.col("image").cast(StringType()).alias("image"),
    F.col("link").cast(StringType()).alias("link"),
    F.col("ratings").cast(DoubleType()).alias("ratings"),
    F.col("no_of_ratings").cast(IntegerType()).alias("no_of_ratings"),
    F.col("discount_price").cast(DoubleType()).alias("discount_price_eur"),
    F.col("actual_price").cast(DoubleType()).alias("actual_price_eur"),
)

products_df.cache()
products_df.printSchema()
print(f"Cleaned Amazon products available for analysis: {products_df.count()}")

# %% [markdown]
# ## Scrape iPhone Generations
#
# The product titles in the Amazon data mention model names, but not release years. To add that context, the notebook scrapes the iPhone model table from Wikipedia and normalizes it into one row per generation.

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

iphone_models_pd.to_csv("wikipedia_iphone_models.csv", index=False)
print(f"iPhone generations scraped: {len(iphone_models_pd)}")
display(iphone_models_pd.head(10))

# %% [markdown]
# ## Move the iPhone Reference Data Into Spark
#
# Spark is used for the matching, scoring, aggregation, and output. The scraped table is small, so it is safe to broadcast during the join.

# %%
iphone_models_spark = spark.createDataFrame(iphone_models_pd)

iphone_models_spark = (
    iphone_models_spark
    .filter(F.col("model_name") != "iPhone")
    .withColumn("model_name_lower", F.lower(F.col("model_name")))
    .withColumn("model_name_length", F.length(F.col("model_name")))
    .withColumn("release_date_parsed", F.to_date(F.col("release_date_iso")))
    .withColumn(
        "assumed_listing_age_years",
        F.round(
            F.datediff(F.lit("2023-12-31").cast("date"), F.col("release_date_parsed")) / F.lit(365.25),
            4,
        )
    )
    .withColumn("age_in_2023", F.lit(2023) - F.col("release_year"))
    .withColumn(
        "age_group",
        F.when(F.col("age_in_2023") >= 5, F.lit("older_5_plus_years"))
        .when(F.col("age_in_2023") >= 2, F.lit("middle_2_to_4_years"))
        .otherwise(F.lit("newer_0_to_1_years"))
    )
)

iphone_models_spark.orderBy("release_year").show(50, truncate=False)

# %% [markdown]
# ## Actual iPhones in the Dataset
#
# This section narrows the data to product titles that look like actual iPhone handsets: titles starting with `Apple iPhone` and including a storage size such as `64GB`, `128GB`, or `256 GB`.
#
# This stricter filter is intentionally conservative. It will miss some possible phone listings, but the rows it keeps are much more likely to be real iPhones rather than cases, chargers, or other accessories. These rows are the only Amazon listings used in the GDELT/news-attention analysis.

# %%
actual_iphones_df = (
    products_df
    .withColumn("product_name_lower", F.lower(F.col("name")))
    .filter(F.col("product_name_lower").rlike(r"^apple iphone"))
    .filter(F.col("product_name_lower").rlike(r"(\([0-9]+\s*gb\)|[0-9]+\s*gb)"))
    .withColumn(
        "matched_model_name",
        F.regexp_extract(F.col("name"), r"Apple\s+(iPhone\s+[^(]+?)\s*\(", 1)
    )
    .withColumn("matched_model_name", F.trim(F.col("matched_model_name")))
    .withColumn("matched_model_name_lower", F.lower(F.col("matched_model_name")))
    .join(
        F.broadcast(
            iphone_models_spark.select(
                "model_name",
                "model_name_lower",
                "release_year",
                "release_date_iso",
                "assumed_listing_age_years",
                "age_in_2023",
                "age_group",
            )
        ),
        F.col("matched_model_name_lower") == F.col("model_name_lower"),
        "left",
    )
    .withColumn(
        "storage_gb",
        F.regexp_extract(F.col("name"), r"([0-9]+)\s*GB", 1).cast(IntegerType())
    )
    .withColumn(
        "discount_pct",
        F.when(
            F.col("actual_price_eur") > 0,
            F.round((F.col("actual_price_eur") - F.col("discount_price_eur")) / F.col("actual_price_eur") * 100, 2)
        )
    )
    .withColumn(
        "reviews_per_assumed_listing_year",
        F.round(F.col("no_of_ratings") / F.col("assumed_listing_age_years"), 2)
    )
    .withColumn(
        "value_score",
        F.round(
            F.col("ratings") * F.log1p(F.col("reviews_per_assumed_listing_year")) / F.col("discount_price_eur"),
            4,
        )
    )
    .select(
        "model_name",
        "release_year",
        "release_date_iso",
        "assumed_listing_age_years",
        "age_in_2023",
        "age_group",
        "storage_gb",
        "name",
        "ratings",
        "no_of_ratings",
        "reviews_per_assumed_listing_year",
        "discount_price_eur",
        "actual_price_eur",
        "discount_pct",
        "value_score",
        "link",
    )
    .orderBy("model_name", "storage_gb", "discount_price_eur")
)

actual_iphones_df.cache()
print(f"Likely actual iPhone handset listings: {actual_iphones_df.count()}")
actual_iphones_df.show(50, truncate=100)

# %% [markdown]
# ## iPhone SE Check
#
# The dataset mentions `iPhone SE`, but those rows are not actual phone listings. They are accessories such as cases, covers, and screen protectors. This check documents that distinction explicitly.

# %%
iphone_se_mentions_df = (
    products_df
    .withColumn("product_name_lower", F.lower(F.col("name")))
    .filter(F.col("product_name_lower").rlike(r"iphone\s+se"))
    .select(
        "name",
        "ratings",
        "no_of_ratings",
        "discount_price_eur",
        "actual_price_eur",
        "link",
    )
    .orderBy("discount_price_eur")
)

actual_iphone_se_df = (
    actual_iphones_df
    .filter(F.lower(F.col("model_name")).rlike(r"iphone\s+se"))
)

print(f"Listings mentioning iPhone SE: {iphone_se_mentions_df.count()}")
iphone_se_mentions_df.show(50, truncate=100)

print(f"Likely actual iPhone SE handset listings: {actual_iphone_se_df.count()}")
actual_iphone_se_df.show(50, truncate=100)

# %% [markdown]
# ## Coverage Check
#
# Before answering the research question, check which actual iPhone generations appear in the Amazon dataset. If a model is missing, the notebook does not infer its value.

# %%
model_coverage_df = (
    actual_iphones_df
    .groupBy("model_name", "release_year", "age_in_2023", "age_group")
    .agg(
        F.count("*").alias("amazon_listing_count"),
        F.countDistinct("storage_gb").alias("storage_option_count"),
        F.round(F.avg("storage_gb"), 2).alias("avg_storage_gb"),
        F.round(F.avg("assumed_listing_age_years"), 2).alias("avg_assumed_listing_age_years"),
        F.round(F.avg("discount_price_eur"), 2).alias("avg_discount_price_eur"),
        F.round(F.avg("ratings"), 2).alias("avg_rating"),
        F.round(F.avg("reviews_per_assumed_listing_year"), 2).alias("avg_reviews_per_assumed_listing_year"),
        F.round(F.avg("value_score"), 4).alias("avg_value_score"),
    )
    .orderBy("release_year", "model_name")
)

model_coverage_df.show(100, truncate=False)

# %% [markdown]
# ## GDELT News Attention in 2023
#
# GDELT is used as a media-attention dataset. For every iPhone model that appears in the Amazon analysis, the notebook queries the GDELT DOC API and counts 2023 news mentions for the model name. The query uses `timelinevolraw`, then sums the raw timeline values across the year.
#
# The results are cached in `gdelt-iphone-news-mentions-2023.csv` so rerunning the notebook does not repeatedly query the public API.

# %%
gdelt_output_file = "gdelt-iphone-news-mentions-2023.csv"
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


model_names_for_gdelt = [
    row["model_name"]
    for row in model_coverage_df.select("model_name").distinct().orderBy("model_name").collect()
]

if os.path.exists(gdelt_output_file):
    cached_gdelt_mentions_pd = pd.read_csv(gdelt_output_file)
    print(f"Loaded cached GDELT mentions from {gdelt_output_file}")
else:
    cached_gdelt_mentions_pd = pd.DataFrame(
        columns=["model_name", "gdelt_mentions_2023", "gdelt_query_url", "gdelt_fetch_status"]
    )

cached_gdelt_mentions_pd["gdelt_mentions_2023"] = pd.to_numeric(
    cached_gdelt_mentions_pd["gdelt_mentions_2023"],
    errors="coerce",
)

usable_cached_models = set(
    cached_gdelt_mentions_pd[
        (cached_gdelt_mentions_pd["gdelt_fetch_status"] == "ok") &
        (cached_gdelt_mentions_pd["gdelt_mentions_2023"].notna())
    ]["model_name"]
)
models_to_fetch_from_gdelt = [
    model_name
    for model_name in model_names_for_gdelt
    if model_name not in usable_cached_models
]
max_live_gdelt_fetches_per_run = 12
models_to_fetch_now = models_to_fetch_from_gdelt[:max_live_gdelt_fetches_per_run]
models_deferred = models_to_fetch_from_gdelt[max_live_gdelt_fetches_per_run:]

fresh_gdelt_mentions_pd = pd.DataFrame([
    fetch_gdelt_mentions_2023(model_name)
    for model_name in models_to_fetch_now
])

deferred_gdelt_mentions_pd = pd.DataFrame([
    {
        "model_name": model_name,
        "gdelt_mentions_2023": None,
        "gdelt_query_url": None,
        "gdelt_fetch_status": "deferred_fetch_limit",
    }
    for model_name in models_deferred
])

gdelt_mentions_pd = (
    pd.concat([cached_gdelt_mentions_pd, fresh_gdelt_mentions_pd, deferred_gdelt_mentions_pd], ignore_index=True)
    .sort_values(["model_name", "gdelt_fetch_status"])
    .drop_duplicates(subset=["model_name"], keep="last")
)
gdelt_mentions_pd = gdelt_mentions_pd[gdelt_mentions_pd["model_name"].isin(model_names_for_gdelt)]
gdelt_mentions_pd.to_csv(gdelt_output_file, index=False)
print(f"Saved GDELT mentions to {gdelt_output_file}")

gdelt_mentions_pd["gdelt_mentions_2023"] = pd.to_numeric(
    gdelt_mentions_pd["gdelt_mentions_2023"],
    errors="coerce",
)
gdelt_mentions_pd["gdelt_fetch_status"] = gdelt_mentions_pd["gdelt_fetch_status"].fillna("unknown")
gdelt_mentions_pd = gdelt_mentions_pd.where(pd.notnull(gdelt_mentions_pd), None)

display(gdelt_mentions_pd)

gdelt_mentions_schema = StructType([
    StructField("model_name", StringType(), True),
    StructField("gdelt_mentions_2023", DoubleType(), True),
    StructField("gdelt_query_url", StringType(), True),
    StructField("gdelt_fetch_status", StringType(), True),
])

gdelt_mentions_df = spark.createDataFrame(gdelt_mentions_pd, schema=gdelt_mentions_schema)

gdelt_mentions_df.show(100, truncate=False)

# %% [markdown]
# ## News Attention vs Amazon Value
#
# This is the main project join: model-level Amazon value scores are joined with GDELT 2023 news mentions. A negative correlation would support the idea that heavily covered iPhone models are worse Amazon value. A positive correlation would suggest that news attention and Amazon value move together in this dataset.

# %%
hype_value_df = (
    model_coverage_df
    .join(gdelt_mentions_df, on="model_name", how="left")
    .withColumn("gdelt_mentions_2023", F.col("gdelt_mentions_2023").cast(DoubleType()))
    .withColumn("log_gdelt_mentions_2023", F.round(F.log1p(F.col("gdelt_mentions_2023")), 4))
    .orderBy(F.col("gdelt_mentions_2023").desc_nulls_last())
)

hype_value_df.select(
    "model_name",
    "release_year",
    "amazon_listing_count",
    "avg_discount_price_eur",
    "avg_reviews_per_assumed_listing_year",
    "avg_value_score",
    "gdelt_mentions_2023",
    "log_gdelt_mentions_2023",
    "gdelt_fetch_status",
).show(100, truncate=False)

hype_value_with_mentions_df = hype_value_df.filter(
    F.col("gdelt_mentions_2023").isNotNull() &
    ~F.isnan(F.col("gdelt_mentions_2023"))
)
hype_value_rows_with_mentions = hype_value_with_mentions_df.count()

if hype_value_rows_with_mentions >= 2:
    news_value_correlation = hype_value_with_mentions_df.stat.corr("log_gdelt_mentions_2023", "avg_value_score")
    news_price_correlation = hype_value_with_mentions_df.stat.corr("log_gdelt_mentions_2023", "avg_discount_price_eur")
else:
    news_value_correlation = None
    news_price_correlation = None

print(f"Models with GDELT mention counts: {hype_value_rows_with_mentions}")
print(f"Correlation log(GDELT 2023 mentions) vs average Amazon value score: {news_value_correlation}")
print(f"Correlation log(GDELT 2023 mentions) vs average discount price: {news_price_correlation}")

# %% [markdown]
# ## Actual iPhone Storage and Price Context
#
# This table summarizes the actual handset listings by model and storage size. It is supporting context for the GDELT comparison and keeps accessories out of the analysis.

# %%
handset_storage_summary_df = (
    actual_iphones_df
    .groupBy("model_name", "storage_gb")
    .agg(
        F.count("*").alias("listing_count"),
        F.round(F.avg("actual_price_eur"), 2).alias("avg_actual_price_eur"),
        F.round(F.avg("discount_price_eur"), 2).alias("avg_discount_price_eur"),
        F.round(F.avg("ratings"), 2).alias("avg_rating"),
        F.round(F.avg("no_of_ratings"), 0).alias("avg_review_count"),
        F.round(F.avg("reviews_per_assumed_listing_year"), 2).alias("avg_reviews_per_assumed_listing_year"),
        F.round(F.avg("value_score"), 4).alias("avg_value_score"),
    )
    .orderBy("model_name", "storage_gb")
)

handset_storage_summary_df.show(truncate=False)

# %% [markdown]
# ## Supporting Context: Older and Newer Actual iPhone Listings
#
# This table is no longer the main answer, but it helps interpret the GDELT result. News attention, model age, price, and review history are related, so the notebook still shows actual handset listings by release-age group:
#
# - `older_5_plus_years`: released in 2018 or earlier
# - `middle_2_to_4_years`: released from 2019 to 2021
# - `newer_0_to_1_years`: released from 2022 to 2023
#
# The most important column is `avg_value_score`. Higher values mean a better combination of rating, review evidence, and low observed price.
# Review evidence is corrected for assumed listing age before the score is calculated.

# %%
age_group_summary_df = (
    actual_iphones_df
    .groupBy("age_group")
    .agg(
        F.count("*").alias("listing_count"),
        F.countDistinct("model_name").alias("model_count"),
        F.round(F.avg("age_in_2023"), 2).alias("avg_model_age_years"),
        F.round(F.avg("discount_price_eur"), 2).alias("avg_discount_price_eur"),
        F.round(F.expr("percentile_approx(discount_price_eur, 0.5)"), 2).alias("median_discount_price_eur"),
        F.round(F.avg("ratings"), 2).alias("avg_rating"),
        F.round(F.avg("no_of_ratings"), 0).alias("avg_review_count"),
        F.round(F.avg("reviews_per_assumed_listing_year"), 2).alias("avg_reviews_per_assumed_listing_year"),
        F.round(F.avg("discount_pct"), 2).alias("avg_discount_pct"),
        F.round(F.avg("value_score"), 4).alias("avg_value_score"),
        F.round(F.expr("percentile_approx(value_score, 0.5)"), 4).alias("median_value_score"),
    )
    .orderBy(
        F.when(F.col("age_group") == "older_5_plus_years", 1)
        .when(F.col("age_group") == "middle_2_to_4_years", 2)
        .otherwise(3)
    )
)

age_group_summary_df.show(truncate=False)

# %% [markdown]
# ## The Best-Value Actual iPhone Listings
#
# This table shows the highest-scoring actual iPhone handset listings.

# %%
top_value_iphone_products_df = (
    actual_iphones_df
    .select(
        "model_name",
        "release_year",
        "age_in_2023",
        "age_group",
        "storage_gb",
        "name",
        "ratings",
        "no_of_ratings",
        "reviews_per_assumed_listing_year",
        "discount_price_eur",
        "actual_price_eur",
        "discount_pct",
        "value_score",
        "link",
    )
    .orderBy(F.col("value_score").desc(), F.col("no_of_ratings").desc())
)

top_value_iphone_products_df.show(25, truncate=100)

# %% [markdown]
# ## Answer the Research Question
#
# The conclusion is computed from the Spark joined table rather than written by hand. The notebook answers whether iPhone models with more 2023 GDELT news mentions tend to have worse Amazon value scores.

# %%
if news_value_correlation is None or news_value_correlation != news_value_correlation:
    conclusion = (
        "GDELT did not return enough usable model-level mention counts to answer whether "
        "news attention is associated with worse Amazon value scores."
    )
elif news_value_correlation < -0.1:
    conclusion = (
        "In this dataset, iPhone models with more 2023 news attention tend to have worse "
        "Amazon value scores. The correlation between log GDELT mentions and average value score is negative."
    )
elif news_value_correlation > 0.1:
    conclusion = (
        "In this dataset, iPhone models with more 2023 news attention do not have worse "
        "Amazon value scores. The correlation between log GDELT mentions and average value score is positive."
    )
else:
    conclusion = (
        "In this dataset, the relationship between 2023 GDELT news attention and Amazon value score "
        "is weak. Media attention does not clearly explain Amazon value."
    )

print(conclusion)

# %% [markdown]
# ## Save Project Outputs
#
# The final project outputs are saved so the analysis can be reused without rerunning the full notebook.

# %%
project_output_dirs = [
    "actual-iphone-listings",
    "iphone-se-mentions",
    "gdelt-iphone-news-mentions-2023",
    "iphone-news-attention-value-summary",
    "iphone-value-model-summary",
    "iphone-handset-storage-summary",
    "iphone-value-age-group-summary",
    "iphone-value-top-listings",
]

for output_dir in project_output_dirs:
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

actual_iphones_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(project_output_dirs[0])
iphone_se_mentions_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(project_output_dirs[1])
gdelt_mentions_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(project_output_dirs[2])
hype_value_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(project_output_dirs[3])
model_coverage_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(project_output_dirs[4])
handset_storage_summary_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(project_output_dirs[5])
age_group_summary_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(project_output_dirs[6])
top_value_iphone_products_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(project_output_dirs[7])

print("Saved iPhone value analysis outputs.")

# %%
spark.stop()
