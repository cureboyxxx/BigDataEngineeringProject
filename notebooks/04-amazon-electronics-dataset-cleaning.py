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
# # 04 Cleaning the Amazon Electronics Sales Dataset with Spark

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

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
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
    "data/All Electronics.csv",
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
exchange_rates_file = "data/frankfurter-inr-eur-exchange-rates-2023.csv"

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
# Set the final column order and data types before writing the cleaned CSV file. Price columns use `decimal(10,2)` so EUR values keep exactly two decimal places in the output schema.

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
# ## Validate Cleaned DataFrame
# Verify the schema, null counts, and basic business rules before writing the cleaned CSV file.

# %%
saved_data = data.cache()

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

print(f"Invalid rows in cleaned DataFrame: {invalid_row_count}")
price_range = saved_data.select(
    F.min("discount_price").alias("min_discount_price_eur"),
    F.max("discount_price").alias("max_discount_price_eur"),
    F.min("actual_price").alias("min_actual_price_eur"),
    F.max("actual_price").alias("max_actual_price_eur"),
)

print("Cleaned EUR price range:")
price_range.show(truncate=False)



# %% [markdown]
# ## Save Cleaned Dataset as CSV
# Write the cleaned Amazon Electronics dataset to a CSV file for the analysis notebook.

# %%
cleaned_output_file = "data/cleaned-amazon-electronics-sales-2023.csv"
saved_data.toPandas().to_csv(cleaned_output_file, index=False)

print(f"Cleaned Amazon Electronics dataset saved to {cleaned_output_file}")

# %%
spark.stop()
