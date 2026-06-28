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
# # 05 Do iPhone Models with More News Attention Have Worse Amazon Value Scores?

# %% [markdown]
# ## Import Libraries

# %%
import os
import sys

spark_home = os.environ.get("SPARK_HOME", "/usr/local/spark")
spark_python_path = os.path.join(spark_home, "python")
py4j_zip_path = os.path.join(spark_python_path, "lib", "py4j-0.10.9.7-src.zip")

for path in [spark_python_path, py4j_zip_path]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

import pandas as pd
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
# Initialize Spark Context and Spark Session.

# %%
spark = (
    SparkSession
    .builder
    .master("local[*]")
    .appName("iphone-amazon-value-news-attention")
    .config("spark.ui.showConsoleProgress", "false")
    .getOrCreate()
)

sc = spark.sparkContext
sc.setLogLevel("ERROR")

print(sc.version)
print(sc.master)

# %% [markdown]
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
# The analysis starts from the cleaned CSV created by `04-amazon-electronics-dataset-cleaning.ipynb` and moves it into Spark for the rest of the project.

# %%
cleaned_data_file = "cleaned-amazon-electronics-sales-2023.csv"

if not os.path.exists(cleaned_data_file):
    raise FileNotFoundError(
        f"{cleaned_data_file} was not found. Run "
        "04-amazon-electronics-dataset-cleaning.ipynb before this analysis notebook."
    )

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

cleaned_data = (
    spark.read
    .option("header", "true")
    .schema(cleaned_schema)
    .csv(cleaned_data_file)
)

products_df = cleaned_data.select(
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
print(f"Cleaned Amazon products loaded from {cleaned_data_file}: {products_df.count()}")

# %% [markdown]
# ## Load iPhone Generations
#
# The product titles in the Amazon data mention model names, but not release years. To add that context, this notebook loads the iPhone model table created by `03-iphone-models-webscraping.ipynb`.

# %%
iphone_models_file = "wikipedia-iphone-models.csv"

if not os.path.exists(iphone_models_file):
    raise FileNotFoundError(
        f"{iphone_models_file} was not found. Run "
        "03-iphone-models-webscraping.ipynb before this analysis notebook."
    )

iphone_models_pd = pd.read_csv(iphone_models_file)
required_iphone_model_columns = [
    "model_name",
    "release_year",
    "release_date_iso",
]
missing_iphone_model_columns = [
    column
    for column in required_iphone_model_columns
    if column not in iphone_models_pd.columns
]

if missing_iphone_model_columns:
    raise ValueError(
        f"{iphone_models_file} is missing required columns: {missing_iphone_model_columns}"
    )

iphone_models_pd = iphone_models_pd.dropna(subset=["model_name", "release_year", "release_date_iso"])
iphone_models_pd["release_year"] = iphone_models_pd["release_year"].astype(int)
iphone_models_pd["model_name"] = iphone_models_pd["model_name"].astype(str).str.strip()

print(f"iPhone generations loaded: {len(iphone_models_pd)}")
display(iphone_models_pd.head(10))

# %% [markdown]
# ## Move the iPhone Reference Data Into Spark
#
# Spark is used for the matching, scoring, aggregation, and output. The iPhone reference table is small, so it is safe to broadcast during the join.

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
# GDELT is used as a media-attention dataset. This notebook loads the 2023 news mention counts created by `02-gdelt-iphone-news-mentions-API-request.ipynb`.

# %%
gdelt_output_file = "gdelt-iphone-news-mentions-2023.csv"

if os.path.exists(gdelt_output_file):
    gdelt_mentions_pd = pd.read_csv(gdelt_output_file)
    print(f"Loaded GDELT mentions from {gdelt_output_file}")
else:
    raise FileNotFoundError(
        f"{gdelt_output_file} was not found. Run "
        "02-gdelt-iphone-news-mentions-API-request.ipynb before this analysis notebook."
    )

required_gdelt_columns = [
    "model_name",
    "gdelt_mentions_2023",
    "gdelt_query_url",
    "gdelt_fetch_status",
]
missing_gdelt_columns = [
    column
    for column in required_gdelt_columns
    if column not in gdelt_mentions_pd.columns
]

if missing_gdelt_columns:
    raise ValueError(f"{gdelt_output_file} is missing required columns: {missing_gdelt_columns}")

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
# ## Review Project Outputs
#
# The final project outputs are kept in memory and displayed in the notebook.

# %%
print(f"Actual iPhone listings: {actual_iphones_df.count()}")
print(f"iPhone SE mention rows: {iphone_se_mentions_df.count()}")
print(f"GDELT mention rows: {gdelt_mentions_df.count()}")
print(f"News attention value rows: {hype_value_df.count()}")
print(f"Model coverage rows: {model_coverage_df.count()}")
print(f"Handset storage summary rows: {handset_storage_summary_df.count()}")
print(f"Age group summary rows: {age_group_summary_df.count()}")
print(f"Top value listing rows: {top_value_iphone_products_df.count()}")

# %%
spark.stop()
