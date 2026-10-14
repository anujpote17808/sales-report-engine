from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Sales Data Filter & Report Engine",
    page_icon="📊",
    layout="wide",
)
css_path = Path(__file__).with_name("style.css")

st.markdown(
    "<style>" + css_path.read_text(encoding="utf-8") + "</style>",
    unsafe_allow_html=True,
)

# 1. Create month labels and a 10-store × 12-month sales array.
months = np.arange(1, 13)
stores = np.arange(1, 11)

rng = np.random.default_rng(42)
sales = rng.integers(20000, 100001, size=120).reshape(10, 12)

month_labels = [f"Month {month}" for month in months]
store_labels = [f"Store {store}" for store in stores]


def sales_table(values, row_labels, column_labels):
    return pd.DataFrame(
        values,
        index=row_labels,
        columns=column_labels,
    )


def query_table(mask):
    store_indices, month_indices = np.where(mask)
    return pd.DataFrame(
        {
            "Store": stores[store_indices],
            "Month": months[month_indices],
            "Sales (₹)": sales[store_indices, month_indices],
        }
    )


st.title("Overview")
st.caption(
    "Sales Data Filter & Report Engine | "
    "Simulated sales for 10 stores across 12 months."
)

column1, column2, column3 = st.columns(3)
column1.metric("Annual Sales", f"₹{sales.sum():,.0f}")
column2.metric("Stores", sales.shape[0])
column3.metric("Months", sales.shape[1])

st.subheader("Annual Sales Data")
st.dataframe(
    sales_table(sales, store_labels, month_labels),
    use_container_width=True,
)

# 2. Slice quarters and individual stores.
st.subheader("Quarter and Store Slicing")

column1, column2 = st.columns(2)

with column1:
    quarter = st.selectbox("Select quarter", ["Q1", "Q2", "Q3", "Q4"])

with column2:
    store_number = st.selectbox("Select store", stores.tolist())

quarter_number = int(quarter[1]) - 1
start = quarter_number * 3
end = start + 3

quarter_sales = sales[:, start:end]
individual_store_sales = sales[store_number - 1, :]

st.write(f"{quarter} sales for all stores")
st.dataframe(
    sales_table(
        quarter_sales,
        store_labels,
        month_labels[start:end],
    ),
    use_container_width=True,
)

st.write(f"Store {store_number}: sales across all 12 months")
st.dataframe(
    pd.DataFrame(
        {
            "Month": months,
            "Sales (₹)": individual_store_sales,
        }
    ),
    hide_index=True,
    use_container_width=True,
)

# 3. Boolean masking using &, | and ~.
st.subheader("Sales Query Engine")

target = st.number_input(
    "Monthly sales target (₹)",
    min_value=0,
    value=60000,
    step=5000,
)

store_averages = sales.mean(axis=1)
overall_average = sales.mean()

above_target = sales > target
below_average_stores = store_averages < overall_average
below_average_mask = below_average_stores[:, None]

queries = {
    "Sales above target": above_target,
    "Sales at or below target": ~above_target,
    "Above target AND store below average": (
        above_target & below_average_mask
    ),
    "Above target OR store below average": (
        above_target | below_average_mask
    ),
}

selected_query = st.selectbox("Choose query", list(queries))
results = query_table(queries[selected_query])

st.write(f"Matching records: {len(results)}")
st.dataframe(
    results,
    hide_index=True,
    use_container_width=True,
)

st.write(
    f"Stores below the overall monthly average "
    f"of ₹{overall_average:,.2f}"
)
st.dataframe(
    pd.DataFrame(
        {
            "Store": stores[below_average_stores],
            "Average Monthly Sales (₹)": (
                store_averages[below_average_stores].round(2)
            ),
        }
    ),
    hide_index=True,
    use_container_width=True,
)

# 4. Fancy indexing: exact indices [1, 4, 7] from the PDF.
st.subheader("Fancy Indexing")

selected_store_indices = np.array([1, 4, 7])
selected_store_sales = sales[selected_store_indices, :]

st.write(
    "Array indices [1, 4, 7] select Stores 2, 5 and 8 "
    "because NumPy indexing starts at zero."
)
st.dataframe(
    sales_table(
        selected_store_sales,
        [store_labels[index] for index in selected_store_indices],
        month_labels,
    ),
    use_container_width=True,
)

monthly_totals = sales.sum(axis=0)
best_month_indices = np.argsort(monthly_totals)[-3:][::-1]
best_month_sales = sales[:, best_month_indices]

st.write("Best 3 months, ranked by total sales across all stores")
st.dataframe(
    sales_table(
        best_month_sales,
        store_labels,
        [month_labels[index] for index in best_month_indices],
    ),
    use_container_width=True,
)

# 5. Monthly and store-wise reports with np.linspace targets.
st.subheader("Monthly Report")

# Planned total monthly sales rise evenly from ₹500,000 to ₹800,000.
growth_targets = np.linspace(500000, 800000, 12)

monthly_report = pd.DataFrame(
    {
        "Month": months,
        "Total Sales (₹)": monthly_totals,
        "Average per Store (₹)": sales.mean(axis=0).round(2),
        "Growth Target (₹)": growth_targets.round(2),
        "Target Met": monthly_totals >= growth_targets,
    }
)

st.caption(
    "Growth targets increase evenly from ₹500,000 in Month 1 "
    "to ₹800,000 in Month 12 using np.linspace."
)
st.dataframe(
    monthly_report,
    hide_index=True,
    use_container_width=True,
)

st.subheader("Store-wise Report")

store_report = pd.DataFrame(
    {
        "Store": stores,
        "Annual Sales (₹)": sales.sum(axis=1),
        "Average Monthly Sales (₹)": store_averages.round(2),
        "Q1 Sales (₹)": sales[:, 0:3].sum(axis=1),
        "Q2 Sales (₹)": sales[:, 3:6].sum(axis=1),
        "Q3 Sales (₹)": sales[:, 6:9].sum(axis=1),
        "Q4 Sales (₹)": sales[:, 9:12].sum(axis=1),
        "Below Average": below_average_stores,
    }
)

st.dataframe(
    store_report,
    hide_index=True,
    use_container_width=True,
)