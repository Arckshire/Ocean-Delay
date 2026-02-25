import io
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Ocean Delay Analyzer (VAD vs VAD_PLANNED)", layout="wide")

st.title("Ocean Delay Analyzer — VAD (Actual) vs VAD_PLANNED (Planned)")
st.write(
    "Upload the raw extract (CSV) from `OCEAN_DATA_QUALITY` and download an Excel report with:\n"
    "- Raw Data\n"
    "- Carrier Performance\n"
    "- Lane Performance"
)

# ----------------------------
# Helpers
# ----------------------------
def clean_str(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s if s != "" and s.lower() != "nan" else None


def to_dt(series: pd.Series) -> pd.Series:
    # Parse timestamps consistently; use UTC to avoid mixed timezone issues during math.
    return pd.to_datetime(series, errors="coerce", utc=True)


def build_lane(df: pd.DataFrame) -> pd.Series:
    # Lane logic:
    # Origin node: ORIGIN_CITY if present else POL
    # Destination node: DESTINATION_CITY if present else POD
    origin_node = df["ORIGIN_CITY"].map(clean_str)
    dest_node = df["DESTINATION_CITY"].map(clean_str)

    pol_node = df["POL"].map(clean_str)
    pod_node = df["POD"].map(clean_str)

    origin_final = origin_node.fillna(pol_node).fillna("UNKNOWN_ORIGIN")
    dest_final = dest_node.fillna(pod_node).fillna("UNKNOWN_DEST")

    return origin_final.astype(str) + " -> " + dest_final.astype(str)


def compute_delay_hours(vad_planned: pd.Series, vad_actual: pd.Series) -> pd.Series:
    # Returns float hours (can be negative if early)
    delta = (vad_actual - vad_planned)
    return delta.dt.total_seconds() / 3600.0


def safe_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        df[col] = np.nan
    return df


def first_non_null(series: pd.Series):
    s = series.dropna()
    if len(s) == 0:
        return np.nan
    return s.iloc[0]


def make_excel_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Excel cannot handle timezone-aware datetimes.
    Convert any tz-aware datetime columns to tz-naive before exporting.
    """
    out = df.copy()

    # Convert tz-aware datetime64 columns
    for col in out.columns:
        if pd.api.types.is_datetime64tz_dtype(out[col]):
            out[col] = out[col].dt.tz_convert(None)

    # Handle object columns that may contain tz-aware pd.Timestamp values
    for col in out.columns:
        if out[col].dtype == "object":
            sample = out[col].dropna().head(50)
            if len(sample) and any(isinstance(v, pd.Timestamp) and v.tz is not None for v in sample):
                out[col] = out[col].apply(
                    lambda v: v.tz_convert(None) if isinstance(v, pd.Timestamp) and v.tz is not None else v
                )

    return out


def to_excel_bytes(raw_df: pd.DataFrame, carrier_df: pd.DataFrame, lane_df: pd.DataFrame) -> bytes:
    output = io.BytesIO()

    raw_df = make_excel_safe(raw_df)
    carrier_df = make_excel_safe(carrier_df)
    lane_df = make_excel_safe(lane_df)

    # Force xlsxwriter (no openpyxl dependency needed)
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        raw_df.to_excel(writer, index=False, sheet_name="Raw_Data")
        carrier_df.to_excel(writer, index=False, sheet_name="Carrier_Performance")
        lane_df.to_excel(writer, index=False, sheet_name="Lane_Performance")

    return output.getvalue()


# ----------------------------
# UI
# ----------------------------
uploaded = st.file_uploader("Upload raw extract CSV", type=["csv"])

with st.expander("Settings", expanded=True):
    clamp_negative = st.checkbox(
        "Treat early arrivals as 0 delay (recommended for 'delay' reporting)",
        value=True
    )
    tie_breaker = st.selectbox(
        "Tie-breaker when Avg Delay is equal",
        options=[
            "Lower shipment volume first (least volume ranks higher)",
            "Higher shipment volume first (more volume ranks higher)"
        ],
        index=0
    )
    st.info(
        "Lane = (ORIGIN_CITY -> DESTINATION_CITY). If ORIGIN_CITY or DESTINATION_CITY is blank, "
        "fallback to (POL -> POD)."
    )

if not uploaded:
    st.stop()

# ----------------------------
# Load data
# ----------------------------
try:
    df_raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

# Ensure required columns exist (create missing as NaN)
required_cols = [
    "TENANT_NAME", "SHIPMENT_ID", "MASTER_SHIPMENT_ID", "SUBSCRIPTION_ID",
    "SHIPMENT_CREATED_DATE", "SHIPMENT_MODIFIED_DATE", "SUBSCRIPTION_CREATED_DT",
    "REQUEST_KEY", "REQUEST_KEY_TYPE",
    "CONTAINER_NUMBER", "CONTAINER_TYPE",
    "CARRIER_NAME", "CARRIER_SCAC",
    "FFW_SCAC", "FFW_NAME", "NVOCC_SCAC", "NVOCC_NAME",
    "IS_TRACKED", "IS_NVOCC",
    "ORIGIN_CITY", "ORIGIN_COUNTRY", "DESTINATION_CITY", "DESTINATION_COUNTRY",
    "LIF_CITY", "LIF_COUNTRY",
    "POL_LOCODE", "POL", "POL_COUNTRY",
    "POD_LOCODE", "POD", "POD_COUNTRY",
    "EDI_SOURCE", "CARRIER_CONNECTIVITY",
    "SUBSCRIPTION_STATUS", "LIFECYCLE_STATUS",
    "SHIPMENT_COMPLETED", "RORO_PRODUCT",
    "CONNECTION_TYPE", "RAW_CONNECTION_TYPE",
    "POL_INVALID", "POD_INVALID",
    "ORIGIN_PICKUP_PLANNED_INITIAL", "ORIGIN_PICKUP_ACTUAL",
    "DELIVERY_PLANNED_INITIAL", "DELIVERY_ACTUAL",
    "CEP_PLANNED", "CGI_PLANNED", "CLL_PLANNED", "VDL_PLANNED", "VAD_PLANNED", "CDD_PLANNED", "CGO_PLANNED", "CER_PLANNED",
    "CEP", "CGI", "CLL", "VDL", "VDL_P44", "VAD", "VAD_P44", "CDD", "CGO", "CER",
    "REPORTING_DATE", "REPORTING_DATE_6",
]
for c in required_cols:
    safe_col(df_raw, c)

# Parse key datetime columns
df_raw["SHIPMENT_CREATED_DATE"] = to_dt(df_raw["SHIPMENT_CREATED_DATE"])
df_raw["SHIPMENT_MODIFIED_DATE"] = to_dt(df_raw["SHIPMENT_MODIFIED_DATE"])
df_raw["SUBSCRIPTION_CREATED_DT"] = to_dt(df_raw["SUBSCRIPTION_CREATED_DT"])
df_raw["VAD_PLANNED"] = to_dt(df_raw["VAD_PLANNED"])
df_raw["VAD"] = to_dt(df_raw["VAD"])

# Compute lane and shipment key
df_raw["LANE"] = build_lane(df_raw)

df_raw["SHIPMENT_KEY"] = df_raw["MASTER_SHIPMENT_ID"].map(clean_str)
df_raw.loc[df_raw["SHIPMENT_KEY"].isna(), "SHIPMENT_KEY"] = df_raw["SHIPMENT_ID"].map(clean_str)

# Row-level delay (mostly for debugging/visibility)
df_raw["DELAY_HOURS_RAW"] = compute_delay_hours(df_raw["VAD_PLANNED"], df_raw["VAD"])
df_raw["DELAY_HOURS"] = df_raw["DELAY_HOURS_RAW"].clip(lower=0) if clamp_negative else df_raw["DELAY_HOURS_RAW"]

# ----------------------------
# Roll up to SHIPMENT level
# ----------------------------
shipment_level = (
    df_raw
    .groupby("SHIPMENT_KEY", dropna=False)
    .agg({
        "TENANT_NAME": first_non_null,
        "MASTER_SHIPMENT_ID": first_non_null,
        "SHIPMENT_ID": first_non_null,
        "CARRIER_NAME": first_non_null,
        "CARRIER_SCAC": first_non_null,
        "FFW_NAME": first_non_null,
        "FFW_SCAC": first_non_null,
        "NVOCC_NAME": first_non_null,
        "NVOCC_SCAC": first_non_null,
        "LANE": first_non_null,
        "POL": first_non_null,
        "POD": first_non_null,
        "VAD_PLANNED": "max",
        "VAD": "max",
        "SHIPMENT_CREATED_DATE": "min",
        "SHIPMENT_MODIFIED_DATE": "max",
    })
    .reset_index()
)

shipment_level["DELAY_HOURS_RAW"] = compute_delay_hours(shipment_level["VAD_PLANNED"], shipment_level["VAD"])
shipment_level["DELAY_HOURS"] = shipment_level["DELAY_HOURS_RAW"].clip(lower=0) if clamp_negative else shipment_level["DELAY_HOURS_RAW"]

# Keep shipments usable for delay calcs
shipment_level_valid = shipment_level.dropna(subset=["VAD_PLANNED", "VAD"])

# ----------------------------
# Carrier performance
# ----------------------------
carrier_perf = (
    shipment_level_valid
    .groupby(["TENANT_NAME", "CARRIER_NAME", "CARRIER_SCAC"], dropna=False)
    .agg(
        SHIPMENTS=("SHIPMENT_KEY", "nunique"),
        LANES=("LANE", "nunique"),
        TOTAL_DELAY_HOURS=("DELAY_HOURS", "sum"),
        AVG_DELAY_HOURS=("DELAY_HOURS", "mean"),
        MEDIAN_DELAY_HOURS=("DELAY_HOURS", "median"),
        MAX_DELAY_HOURS=("DELAY_HOURS", "max"),
    )
    .reset_index()
)

ascending_volume = True if tie_breaker.startswith("Lower") else False
carrier_perf = carrier_perf.sort_values(
    by=["AVG_DELAY_HOURS", "SHIPMENTS"],
    ascending=[False, ascending_volume],
    kind="mergesort"
).reset_index(drop=True)

# ----------------------------
# Lane performance
# ----------------------------
lane_perf = (
    shipment_level_valid
    .groupby(["TENANT_NAME", "LANE", "CARRIER_NAME", "CARRIER_SCAC"], dropna=False)
    .agg(
        SHIPMENTS=("SHIPMENT_KEY", "nunique"),
        TOTAL_DELAY_HOURS=("DELAY_HOURS", "sum"),
        AVG_DELAY_HOURS=("DELAY_HOURS", "mean"),
        MEDIAN_DELAY_HOURS=("DELAY_HOURS", "median"),
        MAX_DELAY_HOURS=("DELAY_HOURS", "max"),
    )
    .reset_index()
)

lane_perf = lane_perf.sort_values(
    by=["TENANT_NAME", "LANE", "AVG_DELAY_HOURS", "SHIPMENTS"],
    ascending=[True, True, False, ascending_volume],
    kind="mergesort"
).reset_index(drop=True)

# Create the “lane header row + carriers” layout
lane_rows = []
for (tenant, lane), grp in lane_perf.groupby(["TENANT_NAME", "LANE"], sort=True):
    lane_rows.append({
        "TENANT_NAME": tenant,
        "LANE": lane,
        "CARRIER_NAME": "",
        "CARRIER_SCAC": "",
        "SHIPMENTS": "",
        "AVG_DELAY_HOURS": "",
        "TOTAL_DELAY_HOURS": "",
        "MEDIAN_DELAY_HOURS": "",
        "MAX_DELAY_HOURS": "",
    })
    for _, r in grp.iterrows():
        lane_rows.append({
            "TENANT_NAME": r["TENANT_NAME"],
            "LANE": r["LANE"],
            "CARRIER_NAME": r["CARRIER_NAME"],
            "CARRIER_SCAC": r["CARRIER_SCAC"],
            "SHIPMENTS": int(r["SHIPMENTS"]) if pd.notna(r["SHIPMENTS"]) else "",
            "AVG_DELAY_HOURS": float(r["AVG_DELAY_HOURS"]) if pd.notna(r["AVG_DELAY_HOURS"]) else "",
            "TOTAL_DELAY_HOURS": float(r["TOTAL_DELAY_HOURS"]) if pd.notna(r["TOTAL_DELAY_HOURS"]) else "",
            "MEDIAN_DELAY_HOURS": float(r["MEDIAN_DELAY_HOURS"]) if pd.notna(r["MEDIAN_DELAY_HOURS"]) else "",
            "MAX_DELAY_HOURS": float(r["MAX_DELAY_HOURS"]) if pd.notna(r["MAX_DELAY_HOURS"]) else "",
        })

lane_perf_formatted = pd.DataFrame(lane_rows)

# ----------------------------
# Preview
# ----------------------------
st.subheader("Preview")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Rows (raw)", len(df_raw))
with c2:
    st.metric("Shipments (unique MASTER_SHIPMENT_ID/SHIPMENT_ID)", shipment_level["SHIPMENT_KEY"].nunique())
with c3:
    st.metric("Shipments usable for delay (have VAD & VAD_PLANNED)", len(shipment_level_valid))

st.write("**Carrier Performance (top 20)**")
st.dataframe(carrier_perf.head(20), use_container_width=True)

st.write("**Lane Performance (sample)**")
st.dataframe(lane_perf_formatted.head(40), use_container_width=True)

# ----------------------------
# Export
# ----------------------------
excel_bytes = to_excel_bytes(df_raw, carrier_perf, lane_perf_formatted)

st.download_button(
    label="Download Excel Report",
    data=excel_bytes,
    file_name="ocean_delay_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
