import io
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Ocean Delay Analyzer (VAD/VAD_P44 vs VAD_PLANNED)", layout="wide")

st.title("Ocean Delay Analyzer — Delay vs Plan (VAD / VAD_P44 vs VAD_PLANNED)")
st.write(
    "Upload the raw extract (CSV) and download an Excel report with:\n"
    "- Raw Data\n"
    "- Carrier Performance\n"
    "- Lane Performance\n"
    "- Early Containers (container-level subset)\n"
    "- Delayed Containers (container-level subset)\n"
    "- OnTime Containers (container-level subset)\n\n"
    "Definitions:\n"
    "- Shipment unit = MASTER_SHIPMENT_ID (fallback to SHIPMENT_ID)\n"
    "- Lane = ORIGIN_CITY → DESTINATION_CITY (fallback to POL → POD if missing)\n"
    "- Actual Arrival Used = VAD if present else VAD_P44\n"
    "- Signed Delay = (Actual Arrival Used - VAD_PLANNED) in hours/days\n"
    "- Early: Signed Delay < 0 | Delayed: Signed Delay > 0 | On-time: Signed Delay = 0\n"
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
    return pd.to_datetime(series, errors="coerce", utc=True)


def build_lane(df: pd.DataFrame) -> pd.Series:
    origin_node = df["ORIGIN_CITY"].map(clean_str)
    dest_node = df["DESTINATION_CITY"].map(clean_str)

    pol_node = df["POL"].map(clean_str)
    pod_node = df["POD"].map(clean_str)

    origin_final = origin_node.fillna(pol_node).fillna("UNKNOWN_ORIGIN")
    dest_final = dest_node.fillna(pod_node).fillna("UNKNOWN_DEST")

    return origin_final.astype(str) + " -> " + dest_final.astype(str)


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

    for col in out.columns:
        if pd.api.types.is_datetime64tz_dtype(out[col]):
            out[col] = out[col].dt.tz_convert(None)

    for col in out.columns:
        if out[col].dtype == "object":
            sample = out[col].dropna().head(50)
            if len(sample) and any(isinstance(v, pd.Timestamp) and v.tz is not None for v in sample):
                out[col] = out[col].apply(
                    lambda v: v.tz_convert(None) if isinstance(v, pd.Timestamp) and v.tz is not None else v
                )

    return out


def compute_hours(planned: pd.Series, actual: pd.Series) -> pd.Series:
    delta = actual - planned
    return delta.dt.total_seconds() / 3600.0


def pick_actual_arrival(vad: pd.Series, vad_p44: pd.Series) -> pd.Series:
    return vad.combine_first(vad_p44)


def sum_bool(s: pd.Series) -> int:
    return int(s.sum())


def to_excel_bytes(
    raw_df: pd.DataFrame,
    carrier_df: pd.DataFrame,
    lane_df: pd.DataFrame,
    early_containers_df: pd.DataFrame,
    delayed_containers_df: pd.DataFrame,
    ontime_containers_df: pd.DataFrame
) -> bytes:
    output = io.BytesIO()

    raw_df = make_excel_safe(raw_df)
    carrier_df = make_excel_safe(carrier_df)
    lane_df = make_excel_safe(lane_df)
    early_containers_df = make_excel_safe(early_containers_df)
    delayed_containers_df = make_excel_safe(delayed_containers_df)
    ontime_containers_df = make_excel_safe(ontime_containers_df)

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        raw_df.to_excel(writer, index=False, sheet_name="Raw_Data")
        carrier_df.to_excel(writer, index=False, sheet_name="Carrier_Performance")
        lane_df.to_excel(writer, index=False, sheet_name="Lane_Performance")
        early_containers_df.to_excel(writer, index=False, sheet_name="Early_Containers")
        delayed_containers_df.to_excel(writer, index=False, sheet_name="Delayed_Containers")
        ontime_containers_df.to_excel(writer, index=False, sheet_name="OnTime_Containers")

    return output.getvalue()


# ----------------------------
# UI
# ----------------------------
uploaded = st.file_uploader("Upload raw extract CSV", type=["csv"])

with st.expander("Settings", expanded=True):
    clamp_negative = st.checkbox(
        "Treat early arrivals as 0 delay in the delay metrics (recommended)",
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
        "Lane count (Carrier Performance) = number of unique lanes a carrier operated on "
        "(based on ORIGIN→DESTINATION fallback POL→POD)."
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

# Parse key datetimes
df_raw["SHIPMENT_CREATED_DATE"] = to_dt(df_raw["SHIPMENT_CREATED_DATE"])
df_raw["SHIPMENT_MODIFIED_DATE"] = to_dt(df_raw["SHIPMENT_MODIFIED_DATE"])
df_raw["SUBSCRIPTION_CREATED_DT"] = to_dt(df_raw["SUBSCRIPTION_CREATED_DT"])

df_raw["VAD_PLANNED"] = to_dt(df_raw["VAD_PLANNED"])
df_raw["VAD"] = to_dt(df_raw["VAD"])
df_raw["VAD_P44"] = to_dt(df_raw["VAD_P44"])

# Lane + shipment key
df_raw["LANE"] = build_lane(df_raw)

df_raw["SHIPMENT_KEY"] = df_raw["MASTER_SHIPMENT_ID"].map(clean_str)
df_raw.loc[df_raw["SHIPMENT_KEY"].isna(), "SHIPMENT_KEY"] = df_raw["SHIPMENT_ID"].map(clean_str)

# Row-level actual used + signed delay
df_raw["VAD_ACTUAL_USED"] = pick_actual_arrival(df_raw["VAD"], df_raw["VAD_P44"])
df_raw["ACTUAL_SOURCE_USED"] = np.where(
    df_raw["VAD"].notna(), "VAD",
    np.where(df_raw["VAD_P44"].notna(), "VAD_P44", "")
)

df_raw["SIGNED_DELAY_HOURS"] = compute_hours(df_raw["VAD_PLANNED"], df_raw["VAD_ACTUAL_USED"])
df_raw["SIGNED_DELAY_DAYS"] = df_raw["SIGNED_DELAY_HOURS"] / 24.0
df_raw["DELAY_HOURS"] = df_raw["SIGNED_DELAY_HOURS"].clip(lower=0) if clamp_negative else df_raw["SIGNED_DELAY_HOURS"]
df_raw["DELAY_DAYS"] = df_raw["DELAY_HOURS"] / 24.0

# ----------------------------
# Shipment-level rollup
# ----------------------------
shipment_level = (
    df_raw
    .groupby("SHIPMENT_KEY", dropna=False)
    .agg({
        "TENANT_NAME": first_non_null,
        "MASTER_SHIPMENT_ID": first_non_null,
        "SHIPMENT_ID": first_non_null,
        "SUBSCRIPTION_ID": first_non_null,
        "REQUEST_KEY": first_non_null,
        "REQUEST_KEY_TYPE": first_non_null,

        "CARRIER_NAME": first_non_null,
        "CARRIER_SCAC": first_non_null,

        "FFW_NAME": first_non_null,
        "FFW_SCAC": first_non_null,
        "NVOCC_NAME": first_non_null,
        "NVOCC_SCAC": first_non_null,

        "LANE": first_non_null,
        "POL": first_non_null,
        "POD": first_non_null,
        "POL_LOCODE": first_non_null,
        "POD_LOCODE": first_non_null,
        "ORIGIN_CITY": first_non_null,
        "DESTINATION_CITY": first_non_null,
        "ORIGIN_COUNTRY": first_non_null,
        "DESTINATION_COUNTRY": first_non_null,

        "VAD_PLANNED": "max",
        "VAD": "max",
        "VAD_P44": "max",

        "SHIPMENT_CREATED_DATE": "min",
        "SHIPMENT_MODIFIED_DATE": "max",
        "SUBSCRIPTION_CREATED_DT": "min",
    })
    .reset_index()
)

shipment_level["VAD_ACTUAL_USED"] = pick_actual_arrival(shipment_level["VAD"], shipment_level["VAD_P44"])
shipment_level["ACTUAL_SOURCE_USED"] = np.where(
    shipment_level["VAD"].notna(), "VAD",
    np.where(shipment_level["VAD_P44"].notna(), "VAD_P44", "")
)

shipment_level["SIGNED_DELAY_HOURS"] = compute_hours(shipment_level["VAD_PLANNED"], shipment_level["VAD_ACTUAL_USED"])
shipment_level["SIGNED_DELAY_DAYS"] = shipment_level["SIGNED_DELAY_HOURS"] / 24.0
shipment_level["DELAY_HOURS"] = shipment_level["SIGNED_DELAY_HOURS"].clip(lower=0) if clamp_negative else shipment_level["SIGNED_DELAY_HOURS"]
shipment_level["DELAY_DAYS"] = shipment_level["DELAY_HOURS"] / 24.0

# Usable: planned + actual used
shipment_level_valid = shipment_level.dropna(subset=["VAD_PLANNED", "VAD_ACTUAL_USED"]).copy()

# Classify shipments
shipment_level_valid["IS_EARLY"] = shipment_level_valid["SIGNED_DELAY_HOURS"] < 0
shipment_level_valid["IS_DELAYED"] = shipment_level_valid["SIGNED_DELAY_HOURS"] > 0
shipment_level_valid["IS_ONTIME"] = shipment_level_valid["SIGNED_DELAY_HOURS"] == 0

# ----------------------------
# Carrier performance (includes ON_TIME)
# ----------------------------
carrier_perf = (
    shipment_level_valid
    .groupby(["TENANT_NAME", "CARRIER_NAME", "CARRIER_SCAC"], dropna=False)
    .agg(
        SHIPMENT_VOLUME=("SHIPMENT_KEY", "nunique"),
        DELAYED_SHIPMENTS=("IS_DELAYED", sum_bool),
        EARLY_SHIPMENTS=("IS_EARLY", sum_bool),
        ON_TIME_SHIPMENTS=("IS_ONTIME", sum_bool),
        LANES=("LANE", "nunique"),

        TOTAL_DELAY_HOURS=("DELAY_HOURS", "sum"),
        TOTAL_DELAY_DAYS=("DELAY_DAYS", "sum"),

        AVG_DELAY_HOURS=("DELAY_HOURS", "mean"),
        AVG_DELAY_DAYS=("DELAY_DAYS", "mean"),

        MEDIAN_DELAY_HOURS=("DELAY_HOURS", "median"),
        MEDIAN_DELAY_DAYS=("DELAY_DAYS", "median"),

        MAX_DELAY_HOURS=("DELAY_HOURS", "max"),
        MAX_DELAY_DAYS=("DELAY_DAYS", "max"),
    )
    .reset_index()
)

carrier_perf = carrier_perf[
    [
        "TENANT_NAME", "CARRIER_NAME", "CARRIER_SCAC",
        "SHIPMENT_VOLUME", "DELAYED_SHIPMENTS", "EARLY_SHIPMENTS", "ON_TIME_SHIPMENTS", "LANES",
        "TOTAL_DELAY_HOURS", "TOTAL_DELAY_DAYS",
        "AVG_DELAY_HOURS", "AVG_DELAY_DAYS",
        "MEDIAN_DELAY_HOURS", "MEDIAN_DELAY_DAYS",
        "MAX_DELAY_HOURS", "MAX_DELAY_DAYS",
    ]
]

ascending_volume = True if tie_breaker.startswith("Lower") else False
carrier_perf = carrier_perf.sort_values(
    by=["AVG_DELAY_HOURS", "SHIPMENT_VOLUME"],
    ascending=[False, ascending_volume],
    kind="mergesort"
).reset_index(drop=True)

# ----------------------------
# Lane performance (includes ON_TIME)
# ----------------------------
lane_perf = (
    shipment_level_valid
    .groupby(["TENANT_NAME", "LANE", "CARRIER_NAME", "CARRIER_SCAC"], dropna=False)
    .agg(
        SHIPMENT_VOLUME=("SHIPMENT_KEY", "nunique"),
        DELAYED_SHIPMENTS=("IS_DELAYED", sum_bool),
        EARLY_SHIPMENTS=("IS_EARLY", sum_bool),
        ON_TIME_SHIPMENTS=("IS_ONTIME", sum_bool),

        TOTAL_DELAY_HOURS=("DELAY_HOURS", "sum"),
        TOTAL_DELAY_DAYS=("DELAY_DAYS", "sum"),

        AVG_DELAY_HOURS=("DELAY_HOURS", "mean"),
        AVG_DELAY_DAYS=("DELAY_DAYS", "mean"),

        MEDIAN_DELAY_HOURS=("DELAY_HOURS", "median"),
        MEDIAN_DELAY_DAYS=("DELAY_DAYS", "median"),

        MAX_DELAY_HOURS=("DELAY_HOURS", "max"),
        MAX_DELAY_DAYS=("DELAY_DAYS", "max"),
    )
    .reset_index()
)

lane_perf = lane_perf[
    [
        "TENANT_NAME", "LANE", "CARRIER_NAME", "CARRIER_SCAC",
        "SHIPMENT_VOLUME", "DELAYED_SHIPMENTS", "EARLY_SHIPMENTS", "ON_TIME_SHIPMENTS",
        "TOTAL_DELAY_HOURS", "TOTAL_DELAY_DAYS",
        "AVG_DELAY_HOURS", "AVG_DELAY_DAYS",
        "MEDIAN_DELAY_HOURS", "MEDIAN_DELAY_DAYS",
        "MAX_DELAY_HOURS", "MAX_DELAY_DAYS",
    ]
]

lane_perf = lane_perf.sort_values(
    by=["TENANT_NAME", "LANE", "AVG_DELAY_HOURS", "SHIPMENT_VOLUME"],
    ascending=[True, True, False, ascending_volume],
    kind="mergesort"
).reset_index(drop=True)

# Lane formatted layout (header row + detail rows)
lane_rows = []
for (tenant, lane), grp in lane_perf.groupby(["TENANT_NAME", "LANE"], sort=True):
    lane_rows.append({
        "TENANT_NAME": tenant,
        "LANE": lane,
        "CARRIER_NAME": "",
        "CARRIER_SCAC": "",
        "SHIPMENT_VOLUME": "",
        "DELAYED_SHIPMENTS": "",
        "EARLY_SHIPMENTS": "",
        "ON_TIME_SHIPMENTS": "",
        "TOTAL_DELAY_HOURS": "",
        "TOTAL_DELAY_DAYS": "",
        "AVG_DELAY_HOURS": "",
        "AVG_DELAY_DAYS": "",
        "MEDIAN_DELAY_HOURS": "",
        "MEDIAN_DELAY_DAYS": "",
        "MAX_DELAY_HOURS": "",
        "MAX_DELAY_DAYS": "",
    })

    for _, r in grp.iterrows():
        lane_rows.append({
            "TENANT_NAME": r["TENANT_NAME"],
            "LANE": r["LANE"],
            "CARRIER_NAME": r["CARRIER_NAME"],
            "CARRIER_SCAC": r["CARRIER_SCAC"],

            "SHIPMENT_VOLUME": int(r["SHIPMENT_VOLUME"]) if pd.notna(r["SHIPMENT_VOLUME"]) else "",
            "DELAYED_SHIPMENTS": int(r["DELAYED_SHIPMENTS"]) if pd.notna(r["DELAYED_SHIPMENTS"]) else "",
            "EARLY_SHIPMENTS": int(r["EARLY_SHIPMENTS"]) if pd.notna(r["EARLY_SHIPMENTS"]) else "",
            "ON_TIME_SHIPMENTS": int(r["ON_TIME_SHIPMENTS"]) if pd.notna(r["ON_TIME_SHIPMENTS"]) else "",

            "TOTAL_DELAY_HOURS": float(r["TOTAL_DELAY_HOURS"]) if pd.notna(r["TOTAL_DELAY_HOURS"]) else "",
            "TOTAL_DELAY_DAYS": float(r["TOTAL_DELAY_DAYS"]) if pd.notna(r["TOTAL_DELAY_DAYS"]) else "",

            "AVG_DELAY_HOURS": float(r["AVG_DELAY_HOURS"]) if pd.notna(r["AVG_DELAY_HOURS"]) else "",
            "AVG_DELAY_DAYS": float(r["AVG_DELAY_DAYS"]) if pd.notna(r["AVG_DELAY_DAYS"]) else "",

            "MEDIAN_DELAY_HOURS": float(r["MEDIAN_DELAY_HOURS"]) if pd.notna(r["MEDIAN_DELAY_HOURS"]) else "",
            "MEDIAN_DELAY_DAYS": float(r["MEDIAN_DELAY_DAYS"]) if pd.notna(r["MEDIAN_DELAY_DAYS"]) else "",

            "MAX_DELAY_HOURS": float(r["MAX_DELAY_HOURS"]) if pd.notna(r["MAX_DELAY_HOURS"]) else "",
            "MAX_DELAY_DAYS": float(r["MAX_DELAY_DAYS"]) if pd.notna(r["MAX_DELAY_DAYS"]) else "",
        })

lane_perf_formatted = pd.DataFrame(lane_rows)

# ----------------------------
# Container-level sheets (subset of RAW rows)
# ----------------------------
early_ship_keys = set(shipment_level_valid.loc[shipment_level_valid["IS_EARLY"], "SHIPMENT_KEY"].dropna().astype(str))
delayed_ship_keys = set(shipment_level_valid.loc[shipment_level_valid["IS_DELAYED"], "SHIPMENT_KEY"].dropna().astype(str))
ontime_ship_keys = set(shipment_level_valid.loc[shipment_level_valid["IS_ONTIME"], "SHIPMENT_KEY"].dropna().astype(str))

df_raw["SHIPMENT_KEY_STR"] = df_raw["SHIPMENT_KEY"].astype(str)

early_containers = df_raw[df_raw["SHIPMENT_KEY_STR"].isin(early_ship_keys)].copy()
delayed_containers = df_raw[df_raw["SHIPMENT_KEY_STR"].isin(delayed_ship_keys)].copy()
ontime_containers = df_raw[df_raw["SHIPMENT_KEY_STR"].isin(ontime_ship_keys)].copy()

container_sheet_cols = [
    "TENANT_NAME",
    "CARRIER_NAME", "CARRIER_SCAC",
    "MASTER_SHIPMENT_ID", "SHIPMENT_ID", "SUBSCRIPTION_ID",
    "REQUEST_KEY", "REQUEST_KEY_TYPE",
    "CONTAINER_NUMBER", "CONTAINER_TYPE",
    "SHIPMENT_CREATED_DATE", "SUBSCRIPTION_CREATED_DT",
    "LANE",
    "ORIGIN_CITY", "ORIGIN_COUNTRY", "DESTINATION_CITY", "DESTINATION_COUNTRY",
    "POL_LOCODE", "POL", "POD_LOCODE", "POD",
    "VAD_PLANNED", "VAD", "VAD_P44", "VAD_ACTUAL_USED", "ACTUAL_SOURCE_USED",
    "SIGNED_DELAY_HOURS", "SIGNED_DELAY_DAYS",
]
for c in container_sheet_cols:
    safe_col(early_containers, c)
    safe_col(delayed_containers, c)
    safe_col(ontime_containers, c)

early_containers_export = early_containers[container_sheet_cols].copy()
delayed_containers_export = delayed_containers[container_sheet_cols].copy()
ontime_containers_export = ontime_containers[container_sheet_cols].copy()

# ----------------------------
# Preview
# ----------------------------
st.subheader("Preview")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Rows (raw)", len(df_raw))
with c2:
    st.metric("Shipments (unique)", shipment_level["SHIPMENT_KEY"].nunique())
with c3:
    st.metric("Shipments usable for delay", len(shipment_level_valid))
with c4:
    st.metric("Carriers in performance", carrier_perf[["CARRIER_NAME", "CARRIER_SCAC"]].drop_duplicates().shape[0])

st.write("**Carrier Performance (preview)**")
st.dataframe(carrier_perf, use_container_width=True)

st.write("**Lane Performance (preview sample)**")
st.dataframe(lane_perf_formatted.head(60), use_container_width=True)

st.write("**Container-level sheet row counts**")
st.write({
    "early_container_rows": int(len(early_containers_export)),
    "delayed_container_rows": int(len(delayed_containers_export)),
    "ontime_container_rows": int(len(ontime_containers_export)),
})

# ----------------------------
# Export
# ----------------------------
excel_bytes = to_excel_bytes(
    df_raw,
    carrier_perf,
    lane_perf_formatted,
    early_containers_export,
    delayed_containers_export,
    ontime_containers_export
)

st.download_button(
    label="Download Excel Report",
    data=excel_bytes,
    file_name="ocean_delay_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
