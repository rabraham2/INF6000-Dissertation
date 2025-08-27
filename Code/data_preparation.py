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
