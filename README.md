# INF6000 Dissertation Project – Retail Sales Forecasting and Fairness Evaluation

This repository hosts the MSc Dissertation Project (INF6000), which focuses on forecasting retail sales revenue and evaluating fairness in predictive models using a multi-year retail transaction dataset (2020–2025). It contains scripts for data cleaning, exploratory data analysis (EDA), statistical testing, forecasting with SARIMAX and ensemble methods, and fairness calibration via ridge regression.

---

## Table of Contents
1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation and Setup](#installation-and-setup)
4. [Instructions to process the original data](#instructions)
5. [Running the Scripts](#running-the-scripts)
6. [Project Structure](#project-structure)
7. [License](#license)

---

## Overview
- **Course**: INF6000
- **Introduction**:Consumer purchasing behaviour is increasingly shaped by discounts, order priorities, and purchase platforms, all of which play a critical role in driving short-term demand and influencing long-term retail performance. Forecasting models are essential to uncover these dynamics and support strategic decision-making in retail. This dissertation examines how discount strategies affect sales revenue trends and whether predictive models behave consistently and fairly across product categories. The study uses SARIMAX forecasting with exogenous regressors, XGBoost machine learning models, and post-hoc ridge calibration to balance accuracy and fairness.

- **Goal**: To analyse the impact of discounts, order priorities, and purchase platforms on retail sales and evaluate the fairness and robustness of forecasting models across product categories.
- **Aims and Objectives**: a) Analyse the impact of discounts, order priority, and purchase methods on purchasing behaviour.
b) Forecast sales revenue trends with discounts as an exogenous variable.
c) Evaluate model fairness in predicting order returns and purchase likelihood.
d) Assess whether predictive models behave consistently across product categories and transaction volumes.

- **Primary Research Question**: How do discounts, order priority, and purchase methods influence consumer behaviour, and how can predictive models be used to forecast sales and assess fairness across retail?

- **Secondary Research Questions**:

a)	What is the effect of discounts, order priority, and purchase methods on purchasing frequency and sales revenue?
b)	How accurately can models with discounts as an exogenous factor forecast sales trends?
c)	How fair are predictive models in forecasting order returns and purchase likelihood based on discount levels?
d)	Do predictive sales models perform consistently across product categories and transaction volumes, or do they have biases?

- **Research Methodology**: This project followed a quantitative, data-driven methodology consisting of four major phases:

1. Data Preparation and Cleaning
      Dataset: Online Sales Dataset (Kaggle)
      Period: 2020–2025
      Records: ~45,500 transactions (reduced to 44,136 valid rows after cleaning)
      Steps: handling missing/inconsistent values, recoding returns, converting InvoiceDate into quarterly periods, standardising continuous variables, encoding categorical fields.

2. Exploratory Data Analysis (EDA)
      Visualisations (histograms, boxplots, correlation heatmaps)
      Statistical testing: Kolmogorov–Smirnov for normality, Kruskal–Wallis for group differences
      Insights: category-wise discount variability, country-level heterogeneity, and revenue-discount relationships.

3. Forecasting Framework
      SARIMAX with discount as an exogenous regressor (2020–2024 training, 2025 testing)
      XGBoost & Ensemble Forecasting to integrate statistical and machine learning methods
      Evaluation metrics: RMSE, MAE, R²

4. Fairness Assessment and Calibration
      Post-hoc ridge regression calibration with leave-one-out validation (LOO-CV)
      Dummy variables for product categories and quarters to capture categorical and seasonal effects
      Aim: reduce predictive bias across categories while maintaining competitive accuracy.

- **Methods Used**:
  - **Data Cleaning & Validation**
  - **Feature Engineering & Aggregation**
  - **Descriptive & Statistical Analysis**
  - **Forecasting**
  - **Fairness & Post-hoc Calibration**
  - **Evaluation Metrics**
  - **Reproducibility & Outputs**

---

## System Requirements
- **Operating System**: Windows, macOS, or Linux
- **Python Version:**: 3.9 or later
- **Python Packages**:
  - pip install pandas numpy matplotlib seaborn scikit-learn statsmodels xgboost openpyxl
  - Others (as needed and specified in the code)

---

## Installation and Setup
1. **Clone the Repository**:
   - Click the green "**Code**" button in this GitHub repository.
   - Copy the HTTPS or SSH link provided (e.g., `https://github.com/rabraham2/INF6000-Dissertation.git`).
   - Open your terminal or command prompt and run:
     ```bash
     git clone https://github.com/rabraham2/INF6000-Dissertation.git
     ```

2. **Install Required Python Packages**:
   Open **PyCharm** or **Other IDE-Enabled Coding Platform** and install the necessary packages:

```python
   pip install pandas numpy matplotlib seaborn scikit-learn statsmodels xgboost openpyxl
```

## Instructions

Step 1: Load the Original Dataset
File: online_sales_dataset.xlsx

Tool Used: Microsoft Excel / Python 
Load the dataset to examine its structure, variable descriptions, and overall data quality.

Step 2: Data Cleaning
Removal of 1,428 invalid or inconsistent rows.
Conversion of InvoiceDate into quarterly periods.
Encoding of categorical attributes (Category, OrderPriority, etc.).
Standardisation of continuous variables (SalesRevenue, Discount).

Step 3: EDA
Descriptive statistics (mean, median, IQR).
Boxplots and histograms for discounts, revenues, and categories.
Correlation analysis between discounts and sales revenues.

Step 4: Forecasting Models
SARIMAX with exogenous regressors.
Ensemble model combining SARIMAX + XGBoost.
Training: 2020–2024; Testing: 2025.

Step 5: Fairness Calibration
Ridge regression with LOO-CV.
Dummy variables for categories and quarters.


## Running the Scripts

```Python Code

## 1. Data Preparation Script ##

import pandas as pd
from tabulate import tabulate

# Load dataset
df = pd.read_csv('online_sales_dataset.csv')

# Track row counts
row_counts = [("Initial", len(df))]

# Convert InvoiceDate to datetime and extract time-based features
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')  # Safe parse
df['Month'] = df['InvoiceDate'].dt.month
df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
df['Hour'] = df['InvoiceDate'].dt.hour
row_counts.append(("After datetime features", len(df)))

# Coerce key numeric fields BEFORE any comparisons/filters
for col in ['Quantity', 'UnitPrice', 'Discount', 'ShippingCost', 'CustomerID']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
row_counts.append(("After numeric coercion", len(df)))

discount_info = {}

if 'Discount' in df.columns:
    # Before scaling stats
    discount_info['raw_count'] = int(df['Discount'].shape[0])
    discount_info['raw_prop_gt_1'] = float((df['Discount'] > 1).mean(skipna=True))
    discount_info['raw_prop_lt_0'] = float((df['Discount'] < 0).mean(skipna=True))

    # Auto-detect percentage scale
    if (df['Discount'] > 1).mean(skipna=True) > 0.20:
        df['Discount'] = df['Discount'] / 100.0
        discount_info['scaled'] = True
    else:
        discount_info['scaled'] = False

    # Drop negative & >100% discounts (strict rule)
    before_rows = len(df)
    df = df[(df['Discount'] >= 0) & (df['Discount'] <= 1)]
    discount_info['dropped_bad_discounts'] = before_rows - len(df)
    row_counts.append(("After discount bounds (0..1)", len(df)))

# Remove invalid Quantity and UnitPrice
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
row_counts.append(("After qty/price > 0", len(df)))

# Drop rows with missing essential fields
df = df.dropna(subset=["CustomerID", "ShippingCost", "WarehouseLocation", "InvoiceDate"])
row_counts.append(("After essential fields dropna", len(df)))

# Reset index after filtering
df.reset_index(drop=True, inplace=True)

# Calculate revenue (Net Sales Revenue = Gross - Shipping)
df['GrossSales'] = df['Quantity'] * df['UnitPrice'] * (1 - df['Discount'])
df['SalesRevenue'] = df['GrossSales'] - df['ShippingCost']
row_counts.append(("After revenue calc", len(df)))

# Remove rows where SalesRevenue is 0
df = df[df['SalesRevenue'] > 0]
row_counts.append(("After remove zero SalesRevenue", len(df)))

# Final check: Remove any rows with negative numeric values
id_like = [c for c in ['CustomerID', 'InvoiceNo', 'StockCode'] if c in df.columns]
numeric_cols = df.select_dtypes(include='number').columns.difference(id_like)
df = df[(df[numeric_cols] >= 0).all(axis=1)]
row_counts.append(("After non-negative numeric check", len(df)))

# Final drop of any remaining missing values
df = df.dropna()
row_counts.append(("After final dropna", len(df)))

# Save cleaned output
out_path = "processed_output_cleaned.csv"
df.to_csv(out_path, index=False)

# Build row count changes DataFrame
row_changes_df = pd.DataFrame(row_counts, columns=["Step", "Row count"])

# Print outputs instead of display_dataframe_to_user
print("\nRow counts after each step:")
print(tabulate(row_changes_df, headers='keys', tablefmt='pretty'))

if discount_info:
    print("\nDiscount summary:")
    discount_df = pd.DataFrame(list(discount_info.items()), columns=["Metric", "Value"])
    print(tabulate(discount_df, headers='keys', tablefmt='pretty'))

print("\nCleaned dataset preview (first 50 rows):")
print(tabulate(df.head(50), headers='keys', tablefmt='pretty'))

summary = {
    "row_counts": row_counts,
    "discount_info": discount_info,
    "final_shape": df.shape,
    "output_path": out_path
}
print("\nSummary:", summary)

# Basic checks
print("\nCleaned Data Overview")
print("Shape:", df.shape)
print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())


## 2. Data Distribution Script ##

# Descriptive statistics of sales revenue

import pandas as pd
import matplotlib.pyplot as plt

# Load processed dataset
df = pd.read_csv("processed_output_cleaned.csv")

# Calculate descriptive statistics
total_valid_transactions = df['SalesRevenue'].count()
mean_sales = df['SalesRevenue'].mean()
median_sales = df['SalesRevenue'].median()
std_sales = df['SalesRevenue'].std()
iqr_sales = df['SalesRevenue'].quantile(0.75) - df['SalesRevenue'].quantile(0.25)
min_sales = df['SalesRevenue'].min()
max_sales = df['SalesRevenue'].max()

# Create a summary table
summary_table = pd.DataFrame({
    "Metric": [
        "Total Valid Transactions",
        "Mean Sales Revenue (£)",
        "Median Sales Revenue (£)",
        "Standard Deviation (£)",
        "Interquartile Range (£)",
        "Minimum Observed Transaction (£)",
        "Maximum Observed Transaction (£)"
    ],
    "Value": [
        total_valid_transactions,
        round(mean_sales, 2),
        round(median_sales, 2),
        round(std_sales, 2),
        round(iqr_sales, 2),
        round(min_sales, 2),
        round(max_sales, 2)
    ]
})

# Display the summary table
print("\nDescriptive Statistics Summary:")
print(summary_table.to_string(index=False))

# Plot histogram for SalesRevenue
plt.figure(figsize=(10, 6))
plt.hist(df['SalesRevenue'], bins=50, color='skyblue', edgecolor='black')
plt.title('Histogram of Sales Revenue', fontsize=14)
plt.xlabel('Sales Revenue (£)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the histogram
histogram_path = "histogram_of_sales_revenue.png"
plt.savefig(histogram_path)

print(f"\nHistogram saved to: {histogram_path}")


# Descriptive statistics of quartiles per year

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Load data
    clean_path = "processed_output_cleaned.csv"
    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"Could not find {clean_path}.")

    df = pd.read_csv(clean_path)

    # Ensure dates & year, and valid SalesRevenue
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["SalesRevenue"] = pd.to_numeric(df["SalesRevenue"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate", "SalesRevenue"])
    df["Year"] = df["InvoiceDate"].dt.year.astype(int)

    # MAIN BODY: Overall summary table
    total_valid = int(df["SalesRevenue"].count())
    mean_sales = df["SalesRevenue"].mean()
    median_sales = df["SalesRevenue"].median()
    std_sales = df["SalesRevenue"].std()
    iqr_sales = df["SalesRevenue"].quantile(0.75) - df["SalesRevenue"].quantile(0.25)
    min_sales = df["SalesRevenue"].min()
    max_sales = df["SalesRevenue"].max()

    overall_summary = pd.DataFrame({
        "Metric": [
            "Total Valid Transactions",
            "Mean Sales Revenue (£)",
            "Median Sales Revenue (£)",
            "Standard Deviation (£)",
            "Interquartile Range (£)",
            "Minimum Observed Transaction (£)",
            "Maximum Observed Transaction (£)",
        ],
        "Value": [
            total_valid,
            round(mean_sales, 2),
            round(median_sales, 2),
            round(std_sales, 2),
            round(iqr_sales, 2),
            round(min_sales, 2),
            round(max_sales, 2),
        ],
    })

    print("\n OVERALL SALES REVENUE SUMMARY ")
    print(overall_summary.to_string(index=False))

    overall_csv_path = "sales_revenue_summary_overall.csv"
    overall_summary.to_csv(overall_csv_path, index=False)

    # APPENDIX: Yearly summary table
    def yr_metrics(g: pd.Series) -> pd.Series:
        q1 = g.quantile(0.25)
        q3 = g.quantile(0.75)
        return pd.Series({
            "Transactions": int(g.count()),
            "Mean (£)": round(g.mean(), 2),
            "Median (£)": round(g.median(), 2),
            "Std (£)": round(g.std(), 2),
            "IQR (£)": round(q3 - q1, 2),
            "Min (£)": round(g.min(), 2),
            "Max (£)": round(g.max(), 2),
            "Q1 (£)": round(q1, 2),
            "Q3 (£)": round(q3, 2),
        })

    by_year = (
        df.groupby("Year")["SalesRevenue"]
          .apply(yr_metrics)
          .unstack()
          .sort_index()
    )

    print("\n SALES REVENUE SUMMARY BY YEAR ")
    print(by_year.to_string())

    by_year_csv_path = "sales_revenue_summary_by_year.csv"
    by_year.to_csv(by_year_csv_path)

    # MAIN BODY: Boxplot across years
    plt.figure(figsize=(12, 6))
    df.boxplot(column="SalesRevenue", by="Year", grid=False)  # default styling
    plt.title("Sales Revenue by Year (Boxplot)")
    plt.suptitle("")  # remove default grouped-by subtitle
    plt.xlabel("Year")
    plt.ylabel("Sales Revenue (£)")
    plt.tight_layout()
    boxplot_path = "sales_revenue_boxplot_by_year.png"
    plt.savefig(boxplot_path)
    plt.close()

    # APPENDIX: Yearly histograms with quartile markers
    out_dir = Path("yearly_histograms")
    out_dir.mkdir(exist_ok=True)

    for year in sorted(df["Year"].unique()):
        y = df.loc[df["Year"] == year, "SalesRevenue"].dropna()

        if y.empty:
            continue

        q1 = y.quantile(0.25)
        med = y.median()
        q3 = y.quantile(0.75)

        plt.figure(figsize=(10, 6))
        plt.hist(y, bins=50)  # default colors
        plt.title(f"Sales Revenue Histogram — {year}")
        plt.xlabel("Sales Revenue (£)")
        plt.ylabel("Frequency")
        # Quartile markers (default style)
        for val, label in [(q1, "Q1"), (med, "Median"), (q3, "Q3")]:
            plt.axvline(val, linestyle="--")
            # place label near the top
            ymax = plt.ylim()[1]
            plt.text(val, ymax * 0.95, label, rotation=90, va='top', ha='right', fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / f"histogram_sales_revenue_{year}.png")
        plt.close()

    # Output printing
    print("\n Files created:")
    print(f"- {overall_csv_path}")
    print(f"- {by_year_csv_path}")
    print(f"- {boxplot_path}")
    print(f"- {out_dir}/ (PNG per year)")

if __name__ == "__main__":
    main()

# Proportion of categories and top 10 products sold

import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("processed_output_cleaned.csv")

# Ensure needed columns exist and are valid
for col in ["Category", "Description", "SalesRevenue"]:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")
df["SalesRevenue"] = pd.to_numeric(df["SalesRevenue"], errors="coerce")
df = df.dropna(subset=["SalesRevenue"])

# PRODUCT CATEGORY SUMMARY BASED ON TRANSACTIONS
df = df.dropna(subset=["Category"]).copy()
cat_counts = df["Category"].value_counts()
cat_total_sales = df.groupby("Category")["SalesRevenue"].sum()
cat_avg_sales = df.groupby("Category")["SalesRevenue"].mean()

category_summary = (
    pd.DataFrame({
        "Transaction Count": cat_counts,
        "Proportion (%)": (cat_counts / len(df) * 100).round(2),
        "Total SalesRevenue (£)": cat_total_sales.round(2),
        "Revenue Share (%)": (cat_total_sales / df["SalesRevenue"].sum() * 100).round(2),
        "Avg SalesRevenue per Tx (£)": cat_avg_sales.round(2),
    })
    .reset_index()
    .rename(columns={"index": "Category"})
    .sort_values(by="Transaction Count", ascending=False)
)

print("\n PRODUCT CATEGORY SUMMARY BASED ON TRANSACTIONS")
print(category_summary.to_string(index=False))
category_summary.to_csv("product_category_summary_based_on_transactions.csv", index=False)

# HISTOGRAM OF PRODUCT CATEGORIES BASED ON TRANSACTIONS
plt.figure(figsize=(12, 6))
plt.bar(category_summary["Category"].astype(str), category_summary["Transaction Count"])
plt.title("Transactions per Category")
plt.xlabel("Category")
plt.ylabel("Transaction Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("histogram_of_product_categories.png")
plt.close()

# TOP PRODUCTS IN DESCRIPTION
top_n = 10
prod_counts = df["Description"].value_counts().head(top_n)
top_products_summary = pd.DataFrame({
    "Product Description": prod_counts.index,
    "Transaction Count": prod_counts.values,
    "Proportion (%)": (prod_counts.values / len(df) * 100).round(2)
})
print("\n TOP 10 PRODUCTS BY TRANSACTION COUNT")
print(top_products_summary.to_string(index=False))
top_products_summary.to_csv("top_products_summary.csv", index=False)

# Pie chart for top products based on transactions
plt.figure(figsize=(8, 8))
plt.pie(prod_counts.values, labels=prod_counts.index, autopct="%1.1f%%")
plt.title("Top 10 Products - Transaction Share")
plt.ylabel("")  # cleaner figure
plt.tight_layout()
plt.savefig("top_products_pie_chart.png")
plt.close()

print("\n Files created:")
print("- product_category_summary_based_on_transactions.csv")
print("- histogram_of_product_categories.png")
print("- top_products_summary.csv")
print("- top_products_pie_chart.png")

# Histogram for plotting discount by different categories

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

# --- Config ---
CSV_PATH = "processed_output_cleaned.csv"
CATEGORIES = ["Furniture", "Electronics", "Stationery", "Accessories", "Apparel"]
BINS = 20
PALETTE = {
    "Furniture":   "#1f77b4",  # blue
    "Electronics": "#ff7f0e",  # orange
    "Stationery":  "#2ca02c",  # green
    "Accessories": "#d62728",  # red
    "Apparel":     "#9467bd",  # purple
}

# --- Load & prepare data ---
df = pd.read_csv(CSV_PATH)
df = df[df["Category"].isin(CATEGORIES)].copy()
df["Discount"] = pd.to_numeric(df["Discount"], errors="coerce")
df = df.dropna(subset=["Discount", "Category"])
df["Category"] = pd.Categorical(df["Category"], categories=CATEGORIES, ordered=True)

sns.set(style="whitegrid")

# --- Helpers ---
def make_legend_handles_labels():
    handles = [Patch(facecolor=PALETTE[c], label=c) for c in CATEGORIES]
    labels = list(CATEGORIES)
    return handles, labels

def format_xaxis_percent(ax):
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_xlim(0, 0.5)

# === 1) Side-by-side (dodge) ===
plt.figure(figsize=(12, 6))
ax = sns.histplot(
    data=df, x="Discount", hue="Category",
    bins=BINS, multiple="dodge", element="bars",
    shrink=0.85, kde=False, alpha=1,
    palette=PALETTE, legend=False  # we'll add the legend manually
)
format_xaxis_percent(ax)
handles, labels = make_legend_handles_labels()
plt.legend(handles=handles, labels=labels, title="Category",
           frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.title("Histogram of Discount by Category (Side-by-Side)")
plt.xlabel("Discount"); plt.ylabel("Count")
plt.tight_layout()
plt.savefig("discount_histogram_by_category_dodge.png", dpi=300)
plt.close()

# === 2) Stacked histogram ===
plt.figure(figsize=(12, 6))
ax = sns.histplot(
    data=df, x="Discount", hue="Category",
    bins=BINS, multiple="stack", element="bars",
    kde=False, alpha=0.9, palette=PALETTE, legend=False
)
format_xaxis_percent(ax)
handles, labels = make_legend_handles_labels()
plt.legend(handles=handles, labels=labels, title="Category",
           frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.title("Histogram of Discount by Category (Stacked)")
plt.xlabel("Discount"); plt.ylabel("Count")
plt.tight_layout()
plt.savefig("discount_histogram_by_category_stacked.png", dpi=300)
plt.close()

# === 3) Faceted (no legend) ===
g = sns.displot(
    data=df, x="Discount", col="Category",
    col_wrap=3, bins=BINS, kde=True,
    common_bins=True, common_norm=False,
    height=3.2, aspect=1.2,
    hue="Category", palette=PALETTE, legend=False
)
for ax in g.axes.flatten():
    format_xaxis_percent(ax)
g.set_axis_labels("Discount", "Count")
g.fig.suptitle("Histogram of Discount by Category (Faceted)", y=1.02)
g.savefig("discount_histogram_by_category_faceted.png", dpi=300)
plt.close("all")

print("Saved:")
print(" - discount_histogram_by_category_dodge.png")
print(" - discount_histogram_by_category_stacked.png")
print(" - discount_histogram_by_category_faceted.png")


# Descriptive statistics for discounts per category
desc_stats = df.groupby("Category")["Discount"].describe()
print("\nDescriptive Statistics for Discount by Category:")
print(desc_stats)

# Top products with highest discount
top_products = df.groupby("Description")["Discount"].mean().sort_values(ascending=False).head(10)
print("\nTop 10 Products with Highest Average Discount:")
print(top_products)

# Summary table for categories
category_summary = df["Category"].value_counts().reset_index()
category_summary.columns = ["Category", "Count"]
print("\nSummary Table for Categories:")
print(category_summary)

# Save tables to CSV
desc_stats.to_csv("discount_descriptive_stats.csv")
top_products.to_csv("top_products_by_discount.csv")
category_summary.to_csv("category_summary.csv", index=False)

print("\n All outputs generated and histogram saved as 'discount_histogram_by_category.png'")

# Sales Channel Descriptive Statistics & Dual Histograms

import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("processed_output_cleaned.csv")

# Ensure required columns exist
for col in ["SalesChannel", "SalesRevenue"]:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# Convert sales revenue to numeric & drop missing rows
df["SalesRevenue"] = pd.to_numeric(df["SalesRevenue"], errors="coerce")
df = df.dropna(subset=["SalesChannel", "SalesRevenue"]).copy()

# Count and proportion of transactions per channel
sales_channel_counts = df["SalesChannel"].value_counts()
sales_channel_proportion = (sales_channel_counts / len(df) * 100).round(2)

# Total revenue per channel
sales_channel_total_revenue = df.groupby("SalesChannel")["SalesRevenue"].sum().round(2)
sales_channel_revenue_share = (sales_channel_total_revenue / df["SalesRevenue"].sum() * 100).round(2)

# Create summary table
sales_channel_summary = pd.DataFrame({
    "Transaction Count": sales_channel_counts,
    "Proportion (%)": sales_channel_proportion,
    "Total SalesRevenue (£)": sales_channel_total_revenue,
    "Revenue Share (%)": sales_channel_revenue_share
}).reset_index().rename(columns={"index": "SalesChannel"})

# Display summary table
print("\n SALES CHANNEL SUMMARY")
print(sales_channel_summary.to_string(index=False))

# Save summary table
sales_channel_summary.to_csv("sales_channel_summary.csv", index=False)

# Plot histogram of transactions per sales channel
plt.figure(figsize=(8, 6))
plt.bar(sales_channel_summary["SalesChannel"], sales_channel_summary["Transaction Count"], color="skyblue")
plt.title("Transactions per Sales Channel")
plt.xlabel("Sales Channel")
plt.ylabel("Transaction Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("histogram_sales_channel_transactions.png")
plt.close()

# Plot histogram of sales revenue per sales channel
plt.figure(figsize=(8, 6))
plt.bar(sales_channel_summary["SalesChannel"], sales_channel_summary["Total SalesRevenue (£)"], color="lightgreen")
plt.title("Sales Revenue per Sales Channel")
plt.xlabel("Sales Channel")
plt.ylabel("Total Sales Revenue (£)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("histogram_sales_channel_revenue.png")
plt.close()

print("\n Files created:")
print("- sales_channel_summary.csv")
print("- histogram_sales_channel_transactions.png")
print("- histogram_sales_channel_revenue.png")

# Payment Method Descriptive Statistics & Histograms

import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Load cleaned dataset
    csv_path = "processed_output_cleaned.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path} in the working directory.")

    df = pd.read_csv(csv_path)

    # Detect payment method column
    candidate_cols = ["PaymentMethod"]
    payment_col = next((c for c in candidate_cols if c in df.columns), None)
    if payment_col is None:
        raise ValueError(
            "Could not find a payment method column. "
            f"Tried: {candidate_cols}. Please rename your column or add it to the list."
        )

    # Ensure SalesRevenue is present and usable
    if "SalesRevenue" not in df.columns:
        raise ValueError("Missing required column 'SalesRevenue' in dataset.")

    df["SalesRevenue"] = pd.to_numeric(df["SalesRevenue"], errors="coerce")

    # Drop rows missing either payment platform or revenue
    df = df.dropna(subset=[payment_col, "SalesRevenue"]).copy()

    # Summary stats by payment platform
    tx_counts = df[payment_col].value_counts()
    tx_prop = (tx_counts / len(df) * 100).round(2)

    total_rev = df.groupby(payment_col)["SalesRevenue"].sum().round(2)
    revenue_share = (total_rev / df["SalesRevenue"].sum() * 100).round(2)
    avg_rev = df.groupby(payment_col)["SalesRevenue"].mean().round(2)
    median_rev = df.groupby(payment_col)["SalesRevenue"].median().round(2)
    std_rev = df.groupby(payment_col)["SalesRevenue"].std().round(2)

    summary = (
        pd.DataFrame({
            "Transaction Count": tx_counts,
            "Proportion (%)": tx_prop,
            "Total SalesRevenue (£)": total_rev,
            "Revenue Share (%)": revenue_share,
            "Avg Revenue / Tx (£)": avg_rev,
            "Median Revenue / Tx (£)": median_rev,
            "Std Revenue / Tx (£)": std_rev,
        })
        .reset_index()
        .rename(columns={"index": "PaymentMethod"})
        .sort_values(by="Transaction Count", ascending=False)
    )

    # Print & save table
    print("\n PAYMENT METHOD SUMMARY ")
    print(summary.to_string(index=False))
    summary.to_csv("payment_method_summary.csv", index=False)

    # Histogram of transactions per payment platform
    plt.figure(figsize=(9, 6))
    plt.bar(summary["PaymentMethod"].astype(str), summary["Transaction Count"])
    plt.title("Transactions per Payment Method")
    plt.xlabel("PaymentMethod")
    plt.ylabel("Transaction Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("histogram_payment_method_transactions.png")
    plt.close()

    # Histogram of total sales revenue per payment platform
    plt.figure(figsize=(9, 6))
    plt.bar(summary["PaymentMethod"].astype(str), summary["Total SalesRevenue (£)"])
    plt.title("Total Sales Revenue per Payment Method")
    plt.xlabel("PaymentMethod")
    plt.ylabel("Total Sales Revenue (£)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("histogram_payment_method_revenue.png")
    plt.close()

    # Print Results
    print("\n Files created:")
    print("- payment_platform_summary.csv")
    print("- histogram_payment_method_transactions.png")
    print("- histogram_payment_method_revenue.png")

if __name__ == "__main__":
    main()

# Histogram for plotting quantity sold with its return status

import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    csv_path = "processed_output_cleaned.csv"  # Adjust if needed
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path} in the working directory.")

    # Load data
    df = pd.read_csv(csv_path)

    # Ensure needed columns exist
    for col in ["Quantity", "ReturnStatus"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Convert Quantity to numeric
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df = df.dropna(subset=["Quantity", "ReturnStatus"]).copy()

    # Map ReturnStatus to boolean
    df["IsReturn"] = df["ReturnStatus"].str.strip().str.lower() == "returned"

    # Use absolute quantities for descriptive stats and plots
    df["AbsQuantity"] = df["Quantity"].abs()

    # --- Function for stats ---
    def qstats(s: pd.Series) -> pd.Series:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        return pd.Series({
            "Count": int(s.count()),
            "Mean": round(s.mean(), 2),
            "Median": round(s.median(), 2),
            "Std": round(s.std(), 2),
            "IQR": round(q3 - q1, 2),
            "Min": round(s.min(), 2),
            "Max": round(s.max(), 2),
        })

    # --- Overall stats ---
    overall_df = qstats(df["AbsQuantity"]).to_frame(name="Quantity (Absolute)").reset_index().rename(columns={"index": "Metric"})
    overall_df.to_csv("quantity_summary_overall.csv", index=False)
    print("\n QUANTITY SUMMARY ")
    print(overall_df.to_string(index=False))

    # --- Stats by return status ---
    by_return = (
        df.groupby("ReturnStatus")["AbsQuantity"]
          .apply(qstats)
          .unstack()
    )
    by_return.to_csv("quantity_summary_by_return.csv")
    print("\n QUANTITY SUMMARY BY RETURN STATUS")
    print(by_return.to_string())

    # Histograms
    nonret = df.loc[df["ReturnStatus"].str.lower() == "not returned", "AbsQuantity"]
    rets   = df.loc[df["ReturnStatus"].str.lower() == "returned", "AbsQuantity"]

    # Overlay histogram
    plt.figure(figsize=(10, 6))
    plt.hist(nonret, bins=30, alpha=0.6, label="Not Returned")
    plt.hist(rets, bins=30, alpha=0.6, label="Returned")
    plt.title("Histogram of Quantity — Returned vs Not Returned")
    plt.xlabel("Quantity (Absolute Units)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("histogram_quantity_overlay.png", dpi=300)
    plt.close()

    # Separate histograms
    plt.figure(figsize=(10, 6))
    plt.hist(nonret, bins=30)
    plt.title("Histogram of Quantity — Not Returned")
    plt.xlabel("Quantity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("histogram_quantity_not_returned.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(rets, bins=30)
    plt.title("Histogram of Quantity — Returned")
    plt.xlabel("Quantity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("histogram_quantity_returned.png", dpi=300)
    plt.close()

    # Done
    print("\n Files created:")
    print("- quantity_summary_overall.csv")
    print("- quantity_summary_by_return.csv")
    print("- histogram_quantity_overlay.png")
    print("- histogram_quantity_not_returned.png")
    print("- histogram_quantity_returned.png")

if __name__ == "__main__":
    main()

# Country-level Sales & Quantity Descriptive Statistics + Charts

import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    csv_path = "processed_output_cleaned.csv"  # adjust if needed
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path} in the working directory.")

    # Load data
    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    for col in ["Country", "SalesRevenue", "Quantity"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Coerce numeric fields & drop missing essentials
    df["SalesRevenue"] = pd.to_numeric(df["SalesRevenue"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df = df.dropna(subset=["Country", "SalesRevenue", "Quantity"]).copy()

    # Descriptive summary by Country
    by_country = df.groupby("Country").agg(
        Transactions=("Country", "size"),
        Products_Sold=("Quantity", "sum"),
        Mean_Products_per_Tx=("Quantity", "mean"),
        Total_SalesRevenue=("SalesRevenue", "sum"),
        Mean_Revenue_per_Tx=("SalesRevenue", "mean"),
        Median_Revenue_per_Tx=("SalesRevenue", "median"),
        Std_Revenue_per_Tx=("SalesRevenue", "std"),
    )

    # Revenue share (%)
    grand_total_rev = by_country["Total_SalesRevenue"].sum()
    by_country["Revenue_Share_%"] = (by_country["Total_SalesRevenue"] / grand_total_rev * 100).round(2)

    # Rounding & ordering
    by_country = by_country.round({
        "Products_Sold": 2,
        "Mean_Products_per_Tx": 2,
        "Total_SalesRevenue": 2,
        "Mean_Revenue_per_Tx": 2,
        "Median_Revenue_per_Tx": 2,
        "Std_Revenue_per_Tx": 2
    }).sort_values("Total_SalesRevenue", ascending=False)

    # Print & save summary table
    print("\n COUNTRY SUMMARY (Transactions, Products Sold, Revenue, Shares)")
    print(by_country.to_string())
    summary_csv = "country_summary.csv"
    by_country.to_csv(summary_csv)

    # Charts
    # 1) Bar chart: Products sold by country
    plt.figure(figsize=(12, 6))
    plt.bar(by_country.index.astype(str), by_country["Products_Sold"])
    plt.title("Products Sold by Country (Total Quantity)")
    plt.xlabel("Country")
    plt.ylabel("Products Sold (Total Quantity)")
    plt.xticks(rotation=60, ha="right")
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.tight_layout()
    products_bar_png = "histogram_country_products_sold.png"
    plt.savefig(products_bar_png, dpi=300)
    plt.close()

    # 2) Bar chart: Total sales revenue by country
    plt.figure(figsize=(12, 6))
    plt.bar(by_country.index.astype(str), by_country["Total_SalesRevenue"])
    plt.title("Total Sales Revenue by Country")
    plt.xlabel("Country")
    plt.ylabel("Total Sales Revenue (£)")
    plt.xticks(rotation=60, ha="right")
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.tight_layout()
    revenue_bar_png = "histogram_country_sales_revenue.png"
    plt.savefig(revenue_bar_png, dpi=300)
    plt.close()

    # 3) Pie chart: Revenue share (%) by country
    plt.figure(figsize=(9, 9))
    # Optional: collapse tiny slices for readability (comment out if you want all)
    data = by_country["Revenue_Share_%"]
    labels = by_country.index.astype(str)
    plt.pie(data, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title("Revenue Share by Country (%)")
    plt.tight_layout()
    pie_png = "pie_country_revenue_share.png"
    plt.savefig(pie_png, dpi=300)
    plt.close()

    # Done
    print("\n Files created:")
    print(f"- {summary_csv}")
    print(f"- {products_bar_png}")
    print(f"- {revenue_bar_png}")
    print(f"- {pie_png}")

if __name__ == "__main__":
    main()


# Analysing How Discount Levels, Order Priority, and Purchase Platforms Influence Customer Behaviour
#Kolmogorov-Smirnov Normality Test

import pandas as pd
from scipy.stats import kstest, zscore

# Load the cleaned dataset
df = pd.read_csv("processed_output_cleaned.csv")

# Select continuous variables
continuous_cols = ['Discount', 'SalesRevenue']

# Apply K-S test after standardising with Z-score
print("Kolmogorov-Smirnov Normality Test Results:\n")

for col in continuous_cols:
    data_standardised = zscore(df[col].dropna())
    stat, p_value = kstest(data_standardised, 'norm')  # test against standard normal
    print(f"Variable: {col}")
    print(f"  D-statistic = {stat:.4f}")
    print(f"  p-value     = {p_value:.4f}")
    if p_value <= 0.05:
        print(" Reject H0: Not normally distributed (use non-parametric tests)\n")
    else:
        print(" Fail to reject H0: Normally distributed (use parametric tests)\n")

#Kruskal-Wallis Test Across Categorical Variables

import pandas as pd
from scipy.stats import kruskal

# Load cleaned dataset
df = pd.read_csv("processed_output_cleaned.csv")

# Define the categorical variables to test
categorical_vars = ['SalesChannel', 'OrderPriority', 'PaymentMethod', 'Country', 'ReturnStatus']

# Run Kruskal-Wallis test for each variable
for cat_var in categorical_vars:
    print(f"\n Kruskal-Wallis Test: Discount across '{cat_var}' groups")

    # Create groups
    groups = [group['Discount'].values for name, group in df.groupby(cat_var)]

    # Apply test (requires at least 2 groups)
    if len(groups) >= 2:
        stat, p = kruskal(*groups)
        print(f"  H-statistic = {stat:.4f}")
        print(f"  p-value     = {p:.4f}")
        if p <= 0.05:
            print(" Reject H0: At least one group differs significantly")
        else:
            print(" Fail to reject H0: No significant difference across groups")
    else:
        print("  Not enough groups to run Kruskal-Wallis test")

#Boxplot Visualisations for Discount across Groups

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("processed_output_cleaned.csv")

# Define categorical variables
categorical_vars = ['SalesChannel', 'OrderPriority', 'PaymentMethod', 'Country', 'ReturnStatus']

# Generate and save boxplots without the warning
for var in categorical_vars:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x=var, y='Discount')  # Removed palette to avoid FutureWarning
    plt.title(f"Boxplot of Discount by {var}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    file_path = f"boxplot_discount_by_{var.lower()}.png"
    plt.savefig(file_path)
    plt.close()

print("Saved boxplots without FutureWarnings.")

# Mean Discount Analysis by Different Groups

import pandas as pd

# Load cleaned dataset
df = pd.read_csv("processed_output_cleaned.csv")

# Ensure 'Discount' is numeric
df["Discount"] = pd.to_numeric(df["Discount"], errors="coerce")

# Drop missing rows in relevant fields
df = df.dropna(subset=["Discount"])

# Define categorical variables for analysis
categories = ["OrderPriority", "SalesChannel", "PaymentMethod", "Country", "ReturnStatus"]

# Loop through and save each summary as separate CSV
for cat in categories:
    summary = df.groupby(cat)["Discount"].mean().reset_index()
    summary = summary.rename(columns={"Discount": "Mean_Discount"})

    # Save to CSV
    filename = f"mean_discount_by_{cat.lower()}.csv"
    summary.to_csv(filename, index=False)

    print(f" Saved: {filename}")
    print(summary.to_string(index=False))


# Using Spearman Correlation

import pandas as pd
from scipy.stats import spearmanr

# Load your dataset
df = pd.read_csv("processed_output_cleaned.csv")

# Drop rows with missing values in the relevant columns
df = df.dropna(subset=["Discount", "SalesRevenue", "ShippingCost"])

# Calculate Spearman correlation: Discount vs SalesRevenue
corr1, pval1 = spearmanr(df['Discount'], df['SalesRevenue'])
print("Spearman Correlation - Discount vs SalesRevenue")
print(f"Correlation Coefficient: {corr1:.4f}")
print(f"P-value: {pval1:.4f}")
if pval1 <= 0.05:
    print(" Statistically Significant: Use this relationship")
else:
    print(" Not Statistically Significant")

print()

# Calculate Spearman correlation: Discount vs ShippingCost
corr2, pval2 = spearmanr(df['Discount'], df['ShippingCost'])
print("Spearman Correlation - Discount vs ShippingCost")
print(f"Correlation Coefficient: {corr2:.4f}")
print(f"P-value: {pval2:.4f}")
if pval2 <= 0.05:
    print(" Statistically Significant: Use this relationship")
else:
    print(" Not Statistically Significant")


# Distribution of Sales Revenue Data in coordination with discount

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
file_path = "processed_output_cleaned.csv"
df = pd.read_csv(file_path)

# Set plot style
sns.set(style="whitegrid")

# Create histogram plots for Discount
plt.figure(figsize=(14, 6))

# Histogram for Discount
plt.subplot(1, 2, 1)
sns.histplot(df["Discount"], bins=30, kde=True, color="skyblue")
plt.title("Distribution of Discount")
plt.xlabel("Discount")
plt.ylabel("Frequency")
file_path = f"histogram_of_discount.png"
plt.savefig(file_path)
plt.close()

# Line Plot of SalesRevenue YoY

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("processed_output_cleaned.csv")

# Use the correct date column 'InvoiceDate' instead of 'OrderDate'
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Year'] = df['InvoiceDate'].dt.year

# Calculate total sales revenue per year
yearly_sales = df.groupby('Year')['SalesRevenue'].sum().reset_index()

# Filter for years 2020 to 2024
filtered_yearly_sales = yearly_sales[(yearly_sales['Year'] >= 2020) & (yearly_sales['Year'] <= 2024)]

# Plotting
plt.figure(figsize=(10, 6))
sns.lineplot(data=filtered_yearly_sales, x='Year', y='SalesRevenue', marker='o')
plt.title('Year-on-Year Sales Revenue (2020–2024)')
plt.ylabel('Total Sales Revenue (£ Millions)')
plt.xlabel('Year')
plt.xticks(filtered_yearly_sales['Year'])
plt.grid(True)

# Format y-axis in millions
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'£{x*1e-6:.1f}M'))

# Save and display
plot_path = "yoy_sales_revenue_2020_2024.png"
plt.savefig(plot_path)
plt.show()


#Quartely Sales Trend Revenue from 2020 to 2024

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator

# Load the cleaned dataset
df = pd.read_csv("processed_output_cleaned.csv")

# Ensure InvoiceDate is datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

# Filter for years 2020–2024
df['Year'] = df['InvoiceDate'].dt.year
df['Quarter'] = df['InvoiceDate'].dt.to_period('Q').dt.quarter
df_filtered = df[df['Year'].between(2020, 2024)]

# Aggregate sales revenue per quarter per year
quarterly_sales = df_filtered.groupby(['Year', 'Quarter'])['SalesRevenue'].sum().reset_index()

# Pivot for plotting
pivot_df = quarterly_sales.pivot(index='Year', columns='Quarter', values='SalesRevenue').fillna(0)
years = pivot_df.index
quarters = [1, 2, 3, 4]
bar_width = 0.2
x = np.arange(len(years))

# Use more distinct colors for better visibility
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

# Plotting
fig, ax = plt.subplots(figsize=(14, 6))

for i, q in enumerate(quarters):
    values = pivot_df[q].values
    bars = ax.bar(x + i * bar_width, values, width=bar_width, color=colors[i], label=f'Q{q}')
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'£{height / 1e6:.2f}M',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

# Y-axis formatting
ax.set_ylabel("Sales Revenue (£ Millions)")
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'£{y/1e6:.1f}M'))
ax.yaxis.set_major_locator(MultipleLocator(0.5e6))

# X-axis setup
ax.set_xticks(x + 1.5 * bar_width)
ax.set_xticklabels(years)
ax.set_xlabel("Year")
ax.set_title("Quarterly Sales Revenue (2020–2024) – Clustered Bar Chart")

# Legend and layout
ax.legend(title="Quarter")
plt.tight_layout()

# Save and show plot
output_path = "quarterly_sales_grouped_varied_2020_2024.png"
plt.savefig(output_path)
plt.show()

#Quartely Category Sales Trend Revenue from 2020 and 2024

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = pd.read_csv("processed_output_cleaned.csv")
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
df['Year'] = df['InvoiceDate'].dt.year


def plot_doughnut_with_labels(year, save_path):
    df_year = df[df['Year'] == year]
    category_sales = df_year.groupby('Category')['SalesRevenue'].sum()
    total_sales = category_sales.sum()
    percentages = (category_sales / total_sales) * 100
    labels = [f"{cat}\n{pct:.1f}%" for cat, pct in zip(category_sales.index, percentages)]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts = ax.pie(category_sales, startangle=90, wedgeprops=dict(width=0.4), colors=plt.cm.Set3.colors)

    # Add labels inside the segments
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        ax.text(x * 0.7, y * 0.7, labels[i], ha=horizontalalignment, va='center', fontsize=10, weight='bold')

    ax.set_title(f"{year} Category-wise Sales Distribution", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# Generate for both years
plot_doughnut_with_labels(2020, "category_sales_doughnut_2020.png")
plot_doughnut_with_labels(2024, "category_sales_doughnut_2024.png")


# Correlation between purchase quantity and discount

import pandas as pd
from scipy.stats import pearsonr
from tabulate import tabulate

# Load dataset
df = pd.read_csv("processed_output_cleaned.csv")
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

# Filter data between 2020 and 2024
df_filtered = df[df['InvoiceDate'].dt.year.between(2020, 2024)]

# Group by InvoiceNo to calculate total quantity and average discount per order
invoice_summary = df_filtered.groupby('InvoiceNo', as_index=False).agg(
    Total_Quantity=('Quantity', 'sum'),
    Avg_Discount=('Discount', 'mean')
)

# Remove invalid or negative values
invoice_summary = invoice_summary[
    (invoice_summary['Total_Quantity'] > 0) &
    (invoice_summary['Avg_Discount'] >= 0)
]

# Calculate Pearson correlation
corr, p_value = pearsonr(invoice_summary['Total_Quantity'], invoice_summary['Avg_Discount'])

# Prepare results table
results = [
    ["Pearson Correlation Coefficient", f"{corr:.4f}"],
    ["P-value", f"{p_value:.6f}"],
    ["Interpretation",
     "Strong Positive" if corr > 0.7 else
     "Moderate Positive" if corr > 0.5 else
     "Weak Positive" if corr > 0.3 else
     "No/Little Correlation" if abs(corr) <= 0.3 else
     "Negative Correlation"]
]

# Display neatly
print("\nCorrelation Analysis: Purchase Quantity vs Discount\n")
print(tabulate(results, headers=["Metric", "Value"], tablefmt="grid"))

# Sales Vs Next-Year Discount

import pandas as pd
from scipy.stats import spearmanr

# Load the cleaned data
df = pd.read_csv("processed_output_cleaned.csv")

# Step 1: Parse InvoiceDate to datetime and extract Year
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
df['Year'] = df['InvoiceDate'].dt.year

# Step 2: Group by Category and Year for SalesRevenue
sales_by_year = df.groupby(['Category', 'Year'])['SalesRevenue'].mean().reset_index()
sales_by_year.rename(columns={'SalesRevenue': 'AvgSalesRevenue'}, inplace=True)

# Step 3: Group by Category and Year for Discount
discount_by_year = df.groupby(['Category', 'Year'])['Discount'].mean().reset_index()
discount_by_year.rename(columns={'Discount': 'AvgDiscount'}, inplace=True)

# Step 4: Shift the sales revenue to compare it with next year's discount
sales_by_year['Year'] += 1  # Shift revenue year forward to match with next year's discount

# Step 5: Merge the two datasets
merged = pd.merge(discount_by_year, sales_by_year, on=['Category', 'Year'])

# Step 6: Spearman correlation
correlation_result = spearmanr(merged['AvgSalesRevenue'], merged['AvgDiscount'])

# Display results
print("Merged Dataset Preview:")
print(merged.head(10))  # Show top 10 rows

print("\n Spearman Correlation between Avg Sales Revenue and Next-Year Avg Discount:")
print(f"Correlation Coefficient: {correlation_result.correlation:.4f}")
print(f"P-value: {correlation_result.pvalue:.4f}")


# Category-wise Spearman correlation between average sales revenue and next-year average discount

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Load the cleaned dataset
df = pd.read_csv("processed_output_cleaned.csv")

# Step 1: Ensure datetime parsing and extract year
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
df['Year'] = df['InvoiceDate'].dt.year

# Step 2: Group sales revenue by Category and Year
sales_by_year = df.groupby(['Category', 'Year'])['SalesRevenue'].mean().reset_index()
sales_by_year.rename(columns={'SalesRevenue': 'AvgSalesRevenue'}, inplace=True)
sales_by_year['Year'] += 1  # Shift revenue year to align with next year's discount

# Step 3: Group discount by Category and Year
discount_by_year = df.groupby(['Category', 'Year'])['Discount'].mean().reset_index()
discount_by_year.rename(columns={'Discount': 'AvgDiscount'}, inplace=True)

# Step 4: Merge to align next-year discounts with current-year sales
merged = pd.merge(discount_by_year, sales_by_year, on=['Category', 'Year'])

# Step 5: Perform category-wise Spearman correlation
category_corr_results = []

for category in merged['Category'].unique():
    sub_df = merged[merged['Category'] == category]
    if len(sub_df) > 1:  # Need at least 2 points to compute correlation
        corr, p = spearmanr(sub_df['AvgSalesRevenue'], sub_df['AvgDiscount'])
        category_corr_results.append({
            'Category': category,
            'Correlation': corr,
            'P-value': p
        })

# Convert results to DataFrame
category_corr_df = pd.DataFrame(category_corr_results)
category_corr_df['P-value'] = category_corr_df['P-value'].round(2)
category_corr_df.sort_values(by='Correlation', ascending=False, inplace=True)
category_corr_df.reset_index(drop=True, inplace=True)

category_corr_df.head(10)  # Show top 10 results sorted by correlation value

# Save to Excel
output_excel_path = "category_sales_discount_correlation.xlsx"
category_corr_df.to_excel(output_excel_path, index=False)

# Visualise correlation values
plt.figure(figsize=(10, 6))
bars = plt.barh(category_corr_df['Category'], category_corr_df['Correlation'], color='steelblue', edgecolor='black')
plt.axvline(0, color='grey', linestyle='--')
plt.xlabel('Spearman Correlation Coefficient')
plt.title('Category-wise Correlation:\nAvg Sales Revenue vs Next-Year Avg Discount')

# Annotate bars
for bar in bars:
    plt.text(bar.get_width() + 0.01 if bar.get_width() > 0 else bar.get_width() - 0.05,
             bar.get_y() + bar.get_height()/2,
             f"{bar.get_width():.2f}",
             va='center')

plt.tight_layout()
plt.savefig("category_correlation_visualisation_fixed.png")
plt.show()
print

# Category-wise Spearman correlation between average sales revenue and average discount of the same year

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Load dataset
df = pd.read_csv("processed_output_cleaned.csv")

# Ensure proper datetime conversion
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["Year"] = df["InvoiceDate"].dt.year

# Group sales revenue by Category and Year
sales_by_year = df.groupby(["Category", "Year"])["SalesRevenue"].mean().reset_index()
sales_by_year.rename(columns={"SalesRevenue": "AvgSalesRevenue"}, inplace=True)

# Group discount by Category and Year
discount_by_year = df.groupby(["Category", "Year"])["Discount"].mean().reset_index()
discount_by_year.rename(columns={"Discount": "AvgDiscount"}, inplace=True)

# Merge sales and discount data on same-year basis
merged = pd.merge(sales_by_year, discount_by_year, on=["Category", "Year"])

# Compute Spearman correlations category-wise
results = []
for cat, group in merged.groupby("Category"):
    corr, pval = spearmanr(group["AvgSalesRevenue"], group["AvgDiscount"])
    results.append([cat, round(corr, 2), round(pval, 2)])

corr_df = pd.DataFrame(results, columns=["Category", "Correlation", "P-value"])
print("Category-wise Sales Discount Correlation (Same Year):")
print(corr_df)

# Save as table
corr_df.to_csv("category_sales_discount_sameyear.csv", index=False)

# Plot correlations
plt.figure(figsize=(8,5))
plt.barh(corr_df["Category"], corr_df["Correlation"], color="steelblue")
for i, (corr, pval) in enumerate(zip(corr_df["Correlation"], corr_df["P-value"])):
    plt.text(corr, i, f"{corr:.2f}", va="center", ha="left" if corr > 0 else "right")

plt.axvline(0, color="grey", linestyle="--")
plt.xlabel("Spearman Correlation Coefficient")
plt.title("Category-wise Correlation:\nAvg Sales Revenue vs Same-Year Avg Discount")
plt.tight_layout()
plt.savefig("category_correlation_visualisation_same_year.png")
plt.show()
print


# Quarterly sales summary (2020–2024)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

# Load & prepare data
df = pd.read_csv("processed_output_cleaned.csv")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df = df.dropna(subset=["InvoiceDate", "SalesRevenue"])

# Year & Quarter
df["Year"] = df["InvoiceDate"].dt.year
df["Quarter"] = df["InvoiceDate"].dt.to_period("Q").dt.quarter

# Filter 2020–2024
df_2020_2024 = df[df["Year"].between(2020, 2024)]

# Aggregate quarterly sales
quarterly = (df_2020_2024
             .groupby(["Year", "Quarter"])["SalesRevenue"]
             .sum()
             .reset_index())

# Pivot for grouped bars
pivot_df = quarterly.pivot(index="Year", columns="Quarter", values="SalesRevenue").fillna(0.0)

# Save the table to CSV (for your appendix)
pivot_df.to_csv("quarterly_sales_summary_2020_2024.csv")
print("Quarterly table saved to: quarterly_sales_summary_2020_2024.csv")
print(pivot_df)

# Plot
years = pivot_df.index.to_numpy()
quarters = [1, 2, 3, 4]
x = np.arange(len(years), dtype=float)
bar_width = 0.20

fig, ax = plt.subplots(figsize=(14, 6))
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']  # Q1..Q4

for i, q in enumerate(quarters):
    vals = pivot_df[q].to_numpy()
    bars = ax.bar(x + i*bar_width, vals, width=bar_width, label=f"Q{q}", color=colors[i])
    for b in bars:
        h = b.get_height()
        ax.annotate(f"£{h/1e6:.2f}M",
                    xy=(b.get_x()+b.get_width()/2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

ax.set_title("Quarterly Sales Revenue (2020–2024) – Clustered Bar Chart")
ax.set_xlabel("Year")
ax.set_ylabel("Sales Revenue (£ Millions)")
ax.set_xticks(x + 1.5*bar_width)
ax.set_xticklabels(years)
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"£{y/1e6:.1f}M"))
ax.yaxis.set_major_locator(MultipleLocator(0.25e6))
ax.legend(title="Quarter")
plt.tight_layout()

# Save PNG
out_path = "quarterly_sales_grouped_varied_2020_2024.png"
plt.savefig(out_path, dpi=150)
print(f"Chart saved: {out_path}")

#Quarterly Sales Revenue Summary (2020–2024)

import pandas as pd

# Load cleaned dataset
df = pd.read_csv("processed_output_cleaned.csv")

# Ensure InvoiceDate is datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

# Extract Year and Quarter
df['Year'] = df['InvoiceDate'].dt.year
df['Quarter'] = df['InvoiceDate'].dt.quarter

# Filter for years 2020–2024
df_filtered = df[df['Year'].between(2020, 2024)]

# Group by Year and Quarter, then calculate descriptive statistics
quarterly_summary = df_filtered.groupby(['Year', 'Quarter'])['SalesRevenue'].agg(
    Mean='mean',
    Median='median',
    Std_Dev='std',
    IQR=lambda x: x.quantile(0.75) - x.quantile(0.25),
    Min='min',
    Max='max',
    Total='sum'
).reset_index()

# Round selected columns to 2 decimal places
cols_to_round = ['Mean', 'Median', 'Std_Dev', 'IQR', 'Min', 'Max']
quarterly_summary[cols_to_round] = quarterly_summary[cols_to_round].round(2)

# Save to CSV
output_path = "quarterly_sales_descriptive_sales_summary_2020_2024.csv"
quarterly_summary.to_csv(output_path, index=False)

output_path



# Total Sales Revenue By Category In 2025

import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("processed_output_cleaned.csv")

# Ensure InvoiceDate is datetime and extract year
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
df['Year'] = df['InvoiceDate'].dt.year

# Filter for 2025 and calculate total sales revenue by category
sales_2025 = df[df['Year'] == 2025].groupby('Category')['SalesRevenue'].sum().reset_index()
sales_2025 = sales_2025.sort_values(by='SalesRevenue', ascending=False)

# Save the total 2025 sales by category to Excel
output_excel_path = "total_sales_2025_by_category.xlsx"
sales_2025.to_excel(output_excel_path, index=False)
output_excel_path


#Ensemble of Time Series + ML prediction

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("processed_output_cleaned.csv")
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
df['Year'] = df['InvoiceDate'].dt.year
df['Quarter'] = df['InvoiceDate'].dt.to_period('Q')

# Filter top 10 categories by total sales
top_categories = df.groupby('Category')['SalesRevenue'].sum().nlargest(10).index
df = df[df['Category'].isin(top_categories)]

# Aggregate sales and discount quarterly
quarterly_data = df.groupby(['Category', 'Quarter']).agg({
    'SalesRevenue': 'sum',
    'Discount': 'mean'
}).reset_index()

# Extract year and quarter as integers for modeling
quarterly_data['Year'] = quarterly_data['Quarter'].dt.year
quarterly_data['Q'] = quarterly_data['Quarter'].dt.quarter

# Forecast with SARIMAX per category
sarimax_preds = []
for cat in top_categories:
    cat_data = quarterly_data[quarterly_data['Category'] == cat].copy()
    cat_data.set_index('Quarter', inplace=True)
    series = cat_data['SalesRevenue']

    if len(series) >= 8:
        try:
            model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4), enforce_stationarity=False, enforce_invertibility=False)
            result = model.fit(disp=False)
            forecast = result.forecast(steps=4)
            forecast_df = pd.DataFrame({
                'Quarter': pd.period_range(start=series.index[-1] + 1, periods=4, freq='Q'),
                'Category': cat,
                'SARIMAX_Pred': forecast.values
            })
            sarimax_preds.append(forecast_df)
        except Exception as e:
            continue

sarimax_result = pd.concat(sarimax_preds)

# Prepare XGBoost training data
quarterly_data['QuarterStr'] = quarterly_data['Quarter'].astype(str)
le = LabelEncoder()
quarterly_data['CategoryEncoded'] = le.fit_transform(quarterly_data['Category'])

xgb_features = ['Year', 'Q', 'Discount', 'CategoryEncoded']
X = quarterly_data[xgb_features]
y = quarterly_data['SalesRevenue']

# Train XGBoost with limited estimators for speed
xgb_model = XGBRegressor(random_state=42, n_estimators=50, max_depth=3)
xgb_model.fit(X, y)

# Predict for 2025 Q1 to Q4
future_quarters = pd.period_range('2025Q1', '2025Q4', freq='Q')
xgb_inputs = pd.DataFrame([
    {'Year': q.year, 'Q': q.quarter, 'Discount': quarterly_data[quarterly_data['Category'] == cat]['Discount'].mean(),
     'Category': cat, 'CategoryEncoded': le.transform([cat])[0]}
    for cat in top_categories for q in future_quarters
])

xgb_preds = xgb_model.predict(xgb_inputs[['Year', 'Q', 'Discount', 'CategoryEncoded']])
xgb_inputs['Quarter'] = future_quarters.tolist() * len(top_categories)
xgb_inputs['XGBoost_Pred'] = xgb_preds

# Merge SARIMAX and XGBoost results
merged = pd.merge(sarimax_result, xgb_inputs, on=['Category', 'Quarter'], how='outer')

# Merge with actuals for 2025
actuals = df[df['Year'] == 2025].copy()
actuals['Quarter'] = actuals['InvoiceDate'].dt.to_period('Q')
actual_quarterly = actuals.groupby(['Category', 'Quarter'])['SalesRevenue'].sum().reset_index()
actual_quarterly.rename(columns={'SalesRevenue': 'Actual'}, inplace=True)

final_df = pd.merge(merged, actual_quarterly, on=['Category', 'Quarter'], how='left')

# Ensemble
best_weight = 0
best_rmse = float('inf')
best_ensemble = []

for w in np.linspace(0, 1, 101):
    ensemble = w * final_df['SARIMAX_Pred'].fillna(0) + (1 - w) * final_df['XGBoost_Pred'].fillna(0)
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(final_df['Actual'].fillna(0), ensemble))
    if rmse < best_rmse:
        best_rmse = rmse
        best_weight = w
        best_ensemble = ensemble

final_df['Ensemble'] = best_ensemble
final_df['SARIMAX_Weight'] = best_weight
final_df['XGBoost_Weight'] = 1 - best_weight

# Compute metrics
rmse = np.sqrt(mean_squared_error(final_df['Actual'].fillna(0), final_df['Ensemble']))
mae = mean_absolute_error(final_df['Actual'].fillna(0), final_df['Ensemble'])
r2 = r2_score(final_df['Actual'].fillna(0), final_df['Ensemble'])

print(final_df.head(10))  # Or use .to_string() for full display
final_df[['SARIMAX_Pred', 'Discount', 'XGBoost_Pred', 'Actual', 'Ensemble']] = (
    final_df[['SARIMAX_Pred', 'Discount', 'XGBoost_Pred', 'Actual', 'Ensemble']].round(2).astype(float)
)
final_df.to_excel("Optimised_Quarterly_Ensemble_Forecasts.xlsx", index=False)
print("Model Evaluation Metrics:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R²): {r2:.4f}")


# Improved SARIMAX with Discount as Exogenous (Lags/Rolling Features)

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Forecasting Years
FORECAST_YEARS = ["2025Q1", "2025Q2", "2025Q3", "2025Q4"]
MIN_SARIMAX_POINTS = 8
XGB_PARAMS = dict(random_state=42, n_estimators=100, max_depth=4)  # Balances speed and accuracy
RESULTS_XLSX = "Improved_Quarterly_Ensemble_Forecasts.xlsx"

# Loading the data for preparation
df = pd.read_csv("processed_output_cleaned.csv")

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["Year"] = df["InvoiceDate"].dt.year
df["Quarter"] = df["InvoiceDate"].dt.to_period("Q")

# Keep top 10 categories by sales
top_categories = df.groupby("Category")["SalesRevenue"].sum().nlargest(10).index
df = df[df["Category"].isin(top_categories)].copy()

# 2) Quarterly aggregation
quarterly_data = (
    df.groupby(["Category", "Quarter"])
      .agg(SalesRevenue=("SalesRevenue", "sum"),
           Discount=("Discount", "mean"),
           Quantity=("Quantity", "sum"))
      .reset_index()
)
quarterly_data["Year"] = quarterly_data["Quarter"].dt.year
quarterly_data["Q"] = quarterly_data["Quarter"].dt.quarter

# Lags & rolling (per category)
quarterly_data = quarterly_data.sort_values(["Category", "Quarter"])
for lag in [1, 2]:
    quarterly_data[f"Lag{lag}"] = quarterly_data.groupby("Category")["SalesRevenue"].shift(lag)
quarterly_data["RollingMean2"] = (
    quarterly_data.groupby("Category")["SalesRevenue"]
    .shift(1).rolling(2).mean()
)

# Label encode category for XGB
le = LabelEncoder()
quarterly_data["CategoryEncoded"] = le.fit_transform(quarterly_data["Category"])

# SARIMAX per category (with Discount as exogenous variable)
sarimax_rows = []
for cat in top_categories:
    cat_df = quarterly_data[quarterly_data["Category"] == cat].copy()
    cat_df = cat_df.sort_values("Quarter")
    # align series & exog and drop NaNs together
    tmp = cat_df[["Quarter", "SalesRevenue", "Discount"]].dropna()
    tmp = tmp.set_index("Quarter")
    series = tmp["SalesRevenue"]
    exog = tmp[["Discount"]]

    if len(series) >= MIN_SARIMAX_POINTS:
        try:
            model = SARIMAX(
                series,
                exog=exog,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 4),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            res = model.fit(disp=False)

            # future horizon
            future_quarters = pd.period_range(
                start=series.index[-1] + 1, periods=len(FORECAST_YEARS), freq="Q"
            )
            # simple exog forecast: use last known discount mean
            future_exog = pd.DataFrame(
                {"Discount": [exog["Discount"].iloc[-min(4, len(exog)):].mean()] * len(future_quarters)},
                index=future_quarters
            )
            fc = res.forecast(steps=len(future_quarters), exog=future_exog)

            sarimax_rows.append(pd.DataFrame({
                "Category": cat,
                "Quarter": future_quarters.astype("period[Q]"),
                "SARIMAX_Pred": fc.values
            }))
        except Exception as e:
            # print(f"SARIMAX failed for {cat}: {e}")
            pass

sarimax_result = pd.concat(sarimax_rows, ignore_index=True) if sarimax_rows else pd.DataFrame(columns=["Category","Quarter","SARIMAX_Pred"])
sarimax_result["Quarter"] = sarimax_result["Quarter"].astype("period[Q]")

# XGBoost fit & future inference
xgb_features = ["Year", "Q", "Discount", "CategoryEncoded", "Lag1", "Lag2", "RollingMean2"]
train = quarterly_data.dropna(subset=xgb_features + ["SalesRevenue"]).copy()

X = train[xgb_features]
y = train["SalesRevenue"]

xgb = XGBRegressor(**XGB_PARAMS)
xgb.fit(X, y)

# Build future input rows for 2025Q1-Q4 for each category
future_quarters = pd.PeriodIndex(FORECAST_YEARS, freq="Q")
xgb_rows = []
for cat in top_categories:
    hist = quarterly_data[quarterly_data["Category"] == cat].copy().sort_values("Quarter")
    # derive simple recent stats for lags when constructing future features
    lag1 = hist["SalesRevenue"].iloc[-1] if len(hist) >= 1 else 0.0
    lag2 = hist["SalesRevenue"].iloc[-2] if len(hist) >= 2 else 0.0
    roll2 = hist["SalesRevenue"].iloc[-2:].mean() if len(hist) >= 2 else lag1

    mean_disc = hist["Discount"].dropna().mean()
    cat_code = le.transform([cat])[0]

    for q in future_quarters:
        xgb_rows.append({
            "Category": cat,
            "Quarter": q,  # ensure present for merge
            "Year": q.year,
            "Q": q.quarter,
            "Discount": mean_disc,
            "CategoryEncoded": cat_code,
            "Lag1": lag1,
            "Lag2": lag2,
            "RollingMean2": roll2
        })

xgb_inputs = pd.DataFrame(xgb_rows)
xgb_inputs["Quarter"] = xgb_inputs["Quarter"].astype("period[Q]")
xgb_inputs["XGBoost_Pred"] = xgb.predict(xgb_inputs[xgb_features])

# Merge model forecasts
merged = pd.merge(sarimax_result, xgb_inputs, on=["Category", "Quarter"], how="outer")

# Actuals for evaluation
actuals = df[df["Year"] == 2025].copy()
actuals["Quarter"] = actuals["InvoiceDate"].dt.to_period("Q")
actual_quarterly = (
    actuals.groupby(["Category", "Quarter"])["SalesRevenue"].sum().reset_index()
)
actual_quarterly.rename(columns={"SalesRevenue": "Actual"}, inplace=True)
actual_quarterly["Quarter"] = actual_quarterly["Quarter"].astype("period[Q]")

final_df = pd.merge(merged, actual_quarterly, on=["Category", "Quarter"], how="left")

# Category-level blending (linear regression)

final_df["Ensemble"] = np.nan
weights = []  # store learned weights for debug/inspection

for cat in top_categories:
    mask_cat = final_df["Category"] == cat
    cat_df = final_df.loc[mask_cat, ["SARIMAX_Pred", "XGBoost_Pred", "Actual"]].dropna()

    if len(cat_df) >= 3:
        # Learn blend weights to map preds -> actual
        lr = LinearRegression()
        lr.fit(cat_df[["SARIMAX_Pred", "XGBoost_Pred"]], cat_df["Actual"])
        final_df.loc[mask_cat, "Ensemble"] = lr.predict(
            final_df.loc[mask_cat, ["SARIMAX_Pred", "XGBoost_Pred"]]
        )
        weights.append((cat, float(lr.coef_[0]), float(lr.coef_[1]), float(lr.intercept_)))
    else:
        # fallback: simple average where available
        final_df.loc[mask_cat, "Ensemble"] = (
            final_df.loc[mask_cat, ["SARIMAX_Pred", "XGBoost_Pred"]]
            .mean(axis=1, skipna=True)
        )
        weights.append((cat, 0.5, 0.5, 0.0))

# Evaluation
# Evaluating only on rows where Actual exists (2025 quarters)
eval_df = final_df.dropna(subset=["Actual"]).copy()
eval_df["PredUsed"] = eval_df["Ensemble"].fillna(
    eval_df["XGBoost_Pred"].fillna(eval_df["SARIMAX_Pred"])
)

rmse = np.sqrt(mean_squared_error(eval_df["Actual"], eval_df["PredUsed"]))
mae = mean_absolute_error(eval_df["Actual"], eval_df["PredUsed"])
r2 = r2_score(eval_df["Actual"], eval_df["PredUsed"])

print(f"\n Improved SARIMAX Model Evaluation Metrics")
print(f"Improved R²: {r2:.4f}")
print(f"RMSE: {rmse:,.2f}")
print(f"MAE: {mae:,.2f}")

# Per-category R²
print("\n Product Category R² Values:")
for cat in top_categories:
    c = eval_df[eval_df["Category"] == cat]
    if c["Actual"].notna().sum() >= 2:
        cr2 = r2_score(c["Actual"], c["PredUsed"])
        print(f"  {cat:<20} R² = {cr2: .4f}")

# Save results
final_df = final_df.sort_values(["Category", "Quarter"])
final_df[['SARIMAX_Pred', 'Discount', 'Lag1', 'Lag2', 'RollingMean2', 'XGBoost_Pred', 'Actual', 'Ensemble']] = (
    final_df[['SARIMAX_Pred', 'Discount', 'Lag1', 'Lag2', 'RollingMean2', 'XGBoost_Pred', 'Actual', 'Ensemble']].round(2).astype(float)
)
final_df.to_excel(RESULTS_XLSX, index=False)
print(f"\nSaved forecasts to: {RESULTS_XLSX}")


# Global Ridge with Category & Quarterly Data
import pandas as pd, numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import Ridge
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def metrics(y, yhat):
    mask = (~pd.isna(y)) & (~pd.isna(yhat))
    if mask.sum()==0:
        return np.nan, np.nan, np.nan
    y, yhat = y[mask], yhat[mask]
    rmse = float(np.sqrt(mean_squared_error(y, yhat)))
    mae  = float(mean_absolute_error(y, yhat))
    r2   = float(r2_score(y, yhat))
    return rmse, mae, r2

# Loading data & preparing aggregate
df = pd.read_csv("processed_output_cleaned.csv")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["Quarter"] = df["InvoiceDate"].dt.to_period("Q")
df["Year"] = df["Quarter"].dt.year
df["Q"] = df["Quarter"].dt.quarter

# Focus on top 10 categories
top_categories = df.groupby("Category")["SalesRevenue"].sum().nlargest(10).index
df = df[df["Category"].isin(top_categories)].copy()

q = (df.groupby(["Category","Quarter"])
       .agg(SalesRevenue=("SalesRevenue","sum"),
            Discount=("Discount","mean"))
       .reset_index())
q["Year"] = q["Quarter"].dt.year
q["Q"]    = q["Quarter"].dt.quarter
q = q.sort_values(["Category","Quarter"]).reset_index(drop=True)

# Label Encoding
le = LabelEncoder()
q["CategoryEncoded"] = le.fit_transform(q["Category"])

# Base features for XGB model
q["lag_1"] = q.groupby("Category")["SalesRevenue"].shift(1)
q["lag_4"] = q.groupby("Category")["SalesRevenue"].shift(4)
q["rolling_mean_4"] = q.groupby("Category")["SalesRevenue"].shift(1).rolling(4).mean().reset_index(level=0, drop=True)
ang = 2*np.pi*(q["Q"]-1)/4
q["sin_q"], q["cos_q"] = np.sin(ang), np.cos(ang)
q["YearCtr"] = q["Year"] - q["Year"].median()
q["disc_x_lag1"] = q["Discount"]*q["lag_1"]
q["disc_x_lag4"] = q["Discount"]*q["lag_4"]

FEATS = ["Year","Q","YearCtr","Discount","CategoryEncoded",
         "lag_1","lag_4","rolling_mean_4","sin_q","cos_q",
         "disc_x_lag1","disc_x_lag4"]

qm = q.dropna(subset=["lag_1","lag_4","rolling_mean_4"]).copy()
train = qm[(qm["Year"]>=2020)&(qm["Year"]<=2024)]
test  = qm[qm["Year"]==2025]

# Per-category XGB predictions for 2025
test_xgb = []
for cat, gtr in train.groupby("Category"):
    gte = test[test["Category"]==cat]
    if gte.empty:
        continue
    mdl = XGBRegressor(random_state=42, n_estimators=500, max_depth=6, learning_rate=0.05,
                       subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0)
    mdl.fit(gtr[FEATS], gtr["SalesRevenue"])
    preds = mdl.predict(gte[FEATS])
    test_xgb.append(pd.DataFrame({"Category": cat, "Quarter": gte["Quarter"].values, "XGB": preds}))
xgb_df = pd.concat(test_xgb, ignore_index=True) if test_xgb else pd.DataFrame(columns=["Category","Quarter","XGB"])

# SARIMAX Predictions for 2025
sar_list = []
for cat, g in q.groupby("Category"):
    hist = g[g["Year"]<=2024].sort_values("Quarter").set_index("Quarter")
    if len(hist) < 6:
        continue
    steps = g[g["Year"]==2025]["Quarter"].nunique()
    if steps == 0:
        continue
    try:
        res = SARIMAX(hist["SalesRevenue"], order=(1,1,1), seasonal_order=(1,1,1,4),
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = res.forecast(steps=steps)
        sar_list.append(pd.DataFrame({"Category": cat, "Quarter": pd.period_range(hist.index[-1]+1, periods=steps, freq="Q"),
                                      "SARIMAX": fc.values}))
    except Exception:
        pass
sar_df = pd.concat(sar_list, ignore_index=True) if sar_list else pd.DataFrame(columns=["Category","Quarter","SARIMAX"])

# Seasonal Naïve 2025 + Actual Data Values
nav_rows = []
for cat, g in q.groupby("Category"):
    for k in [1,2,3,4]:
        a25 = g[(g["Year"]==2025)&(g["Q"]==k)]["SalesRevenue"]
        if a25.empty: continue
        p24 = g[(g["Year"]==2024)&(g["Q"]==k)]["SalesRevenue"]
        nav_rows.append({
            "Category": cat,
            "Quarter": pd.Period(f"2025Q{k}","Q"),
            "Naive": float(p24.iloc[0]) if not p24.empty else np.nan,
            "Actual": float(a25.iloc[0]),
            "Q": k
        })
nav_df = pd.DataFrame(nav_rows)

# Assemble eval_df
eval_df = nav_df.merge(xgb_df, on=["Category","Quarter"], how="left")
eval_df = eval_df.merge(sar_df, on=["Category","Quarter"], how="left")
eval_df["Mean"] = eval_df["Actual"].mean()

# Base model metrics
rmse_x, mae_x, r2_x = metrics(eval_df["Actual"], eval_df["XGB"])
rmse_s, mae_s, r2_s = metrics(eval_df["Actual"], eval_df["SARIMAX"])
rmse_n, mae_n, r2_n = metrics(eval_df["Actual"], eval_df["Naive"])
rmse_m, mae_m, r2_m = metrics(eval_df["Actual"], eval_df["Mean"])

# LOO-CV: Global Ridge with Category & Quarterly Data
cat_ohe = OneHotEncoder(drop="first", sparse_output=False).fit(eval_df[["Category"]])
qtr_ohe = OneHotEncoder(drop="first", sparse_output=False).fit(eval_df[["Q"]])

X_base = eval_df[["XGB","SARIMAX","Naive","Mean"]].fillna(0.0).values
X_cat  = cat_ohe.transform(eval_df[["Category"]])
X_qtr  = qtr_ohe.transform(eval_df[["Q"]])
X_full = np.hstack([X_base, X_cat, X_qtr])
y_full = eval_df["Actual"].values

alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
pred_loo = np.zeros(len(eval_df))
for i in range(len(eval_df)):
    mask = np.ones(len(eval_df), dtype=bool); mask[i]=False
    # choose alpha by minimizing train RMSE on the (n-1) points
    best_a, best_rmse = None, 1e18
    for a in alphas:
        r = Ridge(alpha=a, fit_intercept=True)
        r.fit(X_full[mask], y_full[mask])
        yhat_tr = r.predict(X_full[mask])
        rmse_tr = np.sqrt(mean_squared_error(y_full[mask], yhat_tr))
        if rmse_tr < best_rmse:
            best_rmse, best_a = rmse_tr, a
    r = Ridge(alpha=best_a, fit_intercept=True)
    r.fit(X_full[mask], y_full[mask])
    pred_loo[i] = r.predict(X_full[i:i+1])[0]

rmse_cv, mae_cv, r2_cv = metrics(pd.Series(y_full), pd.Series(pred_loo))

print("Post-hoc 2025 LOO-CV — Global Ridge")
print(f"Base XGB            → R²: {r2_x:.4f}")
print(f"Base SARIMAX        → R²: {r2_s:.4f}")
print(f"Seasonal Naïve      → R²: {r2_n:.4f}")
print(f"Constant Mean       → R²: {r2_m:.4f}")
print(f"LOO-CV Ridge (final)→ RMSE: {rmse_cv:,.2f} | MAE: {mae_cv:,.2f} | R²: {r2_cv:.4f}")

# Saving Results
eval_df[["Naive","Actual","XGB","SARIMAX","Mean"]] = eval_df[["Naive","Actual","XGB","SARIMAX","Mean"]].round(2)
out_path = Path("PostHoc_2025_LOOCV_RidgeClean.xlsx")
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    eval_df.to_excel(writer, index=False, sheet_name="Eval_2025_Base")
    pd.DataFrame({"Pred_LOO":pred_loo, "Actual":y_full}).to_excel(writer, index=False, sheet_name="LOO_Preds")
    pd.DataFrame({
        "Method":[ "XGB","SARIMAX","Naive","Mean","LOO_Ridge_Dummies" ],
        "R2":[ r2_x, r2_s, r2_n, r2_m, r2_cv ],
        "RMSE":[ None, None, None, None, rmse_cv ],
        "MAE":[ None, None, None, None, mae_cv ],
    }).to_excel(writer, index=False, sheet_name="Summary")

print("Saved:", str(out_path))


```

## Project Structure

Dissertation-Project/
├─ Data/
│  ├─ processed_output_cleaned.csv
├─ Scripts/
│  ├─ data_preparation.py
│  ├─ data_distribution.py
├─ Results
├─ README.md
├─ LICENSE


## License

MIT License

Copyright (c) 2025 rabraham2

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

