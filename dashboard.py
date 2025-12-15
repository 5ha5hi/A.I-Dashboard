import streamlit as st
import pandas as pd
from openai import OpenAI
import plotly.express as px
import io
import json
import os

def get_openrouter_api_key():
    # 1️⃣ Streamlit Cloud
    if "OPENROUTER_API_KEY" in st.secrets:
        return st.secrets["OPENROUTER_API_KEY"]

    # 2️⃣ Local environment fallback
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        return api_key

    # 3️⃣ Fail safely
    st.error("❌ OPENROUTER_API_KEY not found. Please add it to Streamlit secrets.")
    st.stop()
# ======================================================
# 🔧 Helper: Strip Code Fences
# ======================================================
def strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    if t.startswith("'''"):
        lines = t.splitlines()[1:]
        if lines and lines[-1].strip() == "'''":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    if not t:
        return t
    first = t.splitlines()[0]
    if first.lower().strip() in ("python", "py"):
        t = "\n".join(t.splitlines()[1:]).lstrip()
    return t


# ======================================================
# 🧠 Build Schema for Code Generation
# ======================================================
def build_schema(df: pd.DataFrame) -> dict:
    """
    Simple schema for code-generation LLM:
    column name -> {dtype, sample_values (for categoricals)}.
    """
    schema = {}
    for col in df.columns:
        series = df[col]
        if series.dtype == 'object' or str(series.dtype).startswith("category"):
            uniques = series.dropna().unique().tolist()[:10]
            schema[col] = {
                "dtype": str(series.dtype),
                "sample_values": [str(v) for v in uniques],
            }
        else:
            schema[col] = {"dtype": str(series.dtype)}
    return schema


# ======================================================
# 📊 Dashboard (Charts with “Show More”)
# ======================================================
def generate_dashboard(df: pd.DataFrame):
    if df is None or df.empty:
        return

    st.markdown("### 📊 Data Overview Dashboard")

    if "dashboard_page" not in st.session_state:
        st.session_state.dashboard_page = 0

    numeric_cols = df.select_dtypes(include=["number"]).columns
    numeric_cols = [
        c for c in numeric_cols
        if not any(x in c.lower() for x in ["id", "code", "key", "empid"])
    ]

    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    chart_defs = []

    # Categorical charts
    for col in cat_cols:
        unique_count = df[col].nunique()
        if 1 < unique_count <= 6:
            chart_defs.append(("pie", col))
        elif 6 < unique_count <= 30:
            chart_defs.append(("bar", col))

    # Numeric charts
    for col in numeric_cols:
        if df[col].nunique() > 5:
            chart_defs.append(("hist", col))

    if not chart_defs:
        st.info("No suitable columns found for charting.")
        return

    charts_per_page = 4
    start = st.session_state.dashboard_page * charts_per_page
    end = start + charts_per_page
    page_charts = chart_defs[start:end]

    cols = st.columns(2)

    for i, (chart_type, col) in enumerate(page_charts):
        with cols[i % 2]:
            if chart_type == "pie":
                fig = px.pie(df, names=col, title=f"Distribution by {col}", hole=0.3)
            elif chart_type == "bar":
                counts = df[col].value_counts().reset_index()
                counts.columns = [col, "Count"]
                fig = px.bar(counts, x=col, y="Count", title=f"Count by {col}")
            else:
                fig = px.histogram(df, x=col, title=f"Distribution of {col}", nbins=20)

            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    # Pagination controls
    if end < len(chart_defs):
        if st.button("➕ Show More Charts"):
            st.session_state.dashboard_page += 1
            st.rerun()

    if st.session_state.dashboard_page > 0:
        if st.button("🔄 Reset Dashboard View"):
            st.session_state.dashboard_page = 0
            st.rerun()


# ======================================================
# 🤖 Custom Query → AI Code Generation
# ======================================================
@st.cache_data(show_spinner="AI is thinking...")
def generate_code(schema: dict, user_input: str) -> str:
    """Uses OpenRouter model to generate Python code for a query."""
    model_name = "nvidia/nemotron-nano-9b-v2:free"

    prompt = f"""
You are an expert Python data analyst. Your task is to write Python code to answer a user's question or to modify a pandas DataFrame based on a user's command.

Schema of DataFrame df (JSON):
{json.dumps(schema, indent=2)}

User request: "{user_input}"

Instructions:
1. Analyze the user's request. It can be a question (read operation) or a command (write/update operation).
2. Write a snippet of Python code to perform the requested operation on the `df` DataFrame.
3. For questions that require a visual answer, generate a Plotly figure and assign it to a `result` variable.
4. For questions that require a tabular or single-value answer, assign the final output to a `result` variable.
5. For commands that modify the DataFrame (e.g., adding a column, deleting rows), the code should modify the `df` in place. It does not need to produce a `result` variable.
6. When dealing with dates, always use pd.to_datetime with a specified format if possible, or `dayfirst=True`, and `errors='coerce'`.
7. Return ONLY the Python code snippet, without any markdown formatting or explanations.
8. Be very careful with user input that contains quotes or special characters. Always generate valid Python code that correctly escapes strings.

Rules:
- Generate ONLY Python code.
- Use df, pd, px ONLY (already imported).
- For visual output: assign a Plotly figure to `result`.
- For tabular/single outputs: assign the final object to `result`.
- For modifications: modify df in place (no `result` needed).
- Handle dates with pd.to_datetime(..., dayfirst=True, errors='coerce').
- Escape all quotes properly.
"""

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=get_openrouter_api_key(),
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


# ======================================================
# 🔍 Display Helpers
# ======================================================
def display_person_selector(result_df: pd.DataFrame):
    """Simple 'person' selector if query returns multiple matches."""
    st.info("Multiple people found. Please select one to see details.")

    cols = result_df.columns

    def find_col(keywords):
        for col in cols:
            low = col.lower()
            for kw in keywords:
                if kw in low:
                    return col
        return None

    first_name_col = find_col(["first", "fname"])
    last_name_col = find_col(["last", "lname"])
    dept_col = find_col(["dept", "department"])
    id_col = find_col(["id", "empid"])

    def make_label(row):
        fname = row[first_name_col] if first_name_col in result_df.columns else ""
        lname = row[last_name_col] if last_name_col in result_df.columns else ""
        dept = row[dept_col] if dept_col in result_df.columns else "N/A"
        empid = row[id_col] if id_col in result_df.columns else "N/A"
        return f"{fname} {lname} (Dept: {dept}) — ID: {empid}"

    df2 = result_df.copy()
    df2["display_name"] = df2.apply(make_label, axis=1)
    options = ["- Select -"] + df2["display_name"].tolist()
    choice = st.selectbox("Select a person:", options)

    if choice != "- Select -":
        row = df2[df2["display_name"] == choice]
        if not row.empty:
            st.dataframe(result_df.loc[row.index])


def display_results(result, user_input: str):
    """Displays the result based on its type + simple 'person search' heuristic."""
    if result is None:
        st.success("Command executed successfully.")
        st.dataframe(st.session_state.df.head())
        return

    person_keywords = ["who is", "find", "search", "employee", "person", "details for"]
    if any(k in user_input.lower() for k in person_keywords):
        if isinstance(result, pd.DataFrame) and len(result) > 1:
            display_person_selector(result)
            return

    if isinstance(result, pd.DataFrame):
        st.dataframe(result)
    elif hasattr(result, "to_plotly_json"):
        st.plotly_chart(result, use_container_width=True)
    else:
        st.write(result)


# ======================================================
# 🧠 AI KPI Analyzer using YOUR big Business Analyst prompt
# ======================================================
BUSINESS_ANALYST_PROMPT = """
You are a highly experienced Business Analyst with 20+ years of expertise in data analysis, KPI design, and visualization. Your job: analyze the provided dataset and produce actionable, business-focused KPIs, recommended visualizations, key insights and prioritized recommendations.

Strict rules:
1. Output MUST be valid JSON exactly matching the schema below. No extra text, markdown, or commentary outside the JSON block.
2. Never invent columns or values that do not exist in the dataset.
3. Do NOT produce meaningless metrics (e.g., averages/sums of ID fields, counts of unique identifiers, or trivial totals).
4. Focus on business relevance: KPIs must be actionable and explain how they influence decisions.
5. Visualizations must be appropriate and meaningful (avoid useless pie charts of unique ids).
6. When dealing with dates/times, always parse with `pd.to_datetime(..., dayfirst=True, errors='coerce')` and state the interpreted granularity (daily/weekly/monthly).
7. If any required column is missing for a KPI / visualization, return a clear placeholder in the `current_value` field like: `"missing_column: <col_name>"` and do not fabricate a value.
8. Keep each textual field concise (max ~40 words). Numeric `current_value` fields should be formatted (commas and two decimal places where applicable).
9. For any ratio/percent KPI include numerator, denominator, and the calculation formula. For trend, compute a simple direction (positive/negative/neutral) based on the last 3 comparable periods where possible; if not possible, use "neutral" and explain in one sentence inside the `recommendations` array (still inside JSON).
10. If the dataset appears to be of a sensitive domain (health, finance, personal data), flag it in `business_context` and avoid public-facing recommendations.
11. You may ONLY reference column names that exist exactly in the provided Dataset Schema. Do NOT invent derived column names (e.g., age, tenure, distribution, summary). If a KPI uses a derived metric, treat the source column as the required column.

Before finalizing the output, verify that every column name used in KPIs and visualizations exists in the Dataset Schema. If not, remove or correct the KPI.
Final check (mandatory): List internally all column names you referenced and ensure each one exists in the Dataset Schema. If any do not exist, revise the output before responding.
Data and context inputs (the runner will substitute these):
- Dataset Schema: {schema}
- Sample Data (first 5 rows): {sample_data}
- Data Context / User Notes: {context}

Output JSON schema (exact keys required):
{
  "business_context": "<brief description of business domain & data scope, mention any inferred granularity (e.g., daily sales) and data quality caveats>",
  "meaningful_kpis": [
    {
      "name": "<KPI Name>",
      "description": "<what it measures, business interpretation>",
      "calculation": "<clear formula e.g., SUM(revenue)/COUNT(orders)>",
      "business_value": "<why it matters (one short sentence)>",
      "current_value": "<numeric or explicit placeholder if missing: e.g., 1,234.56 or 'missing_column: revenue'>",
      "trend": "<positive | negative | neutral>"
    }
    // 8-12 such KPI objects
  ],
  "recommended_visualizations": [
    {
      "chart_type": "bar | line | scatter | histogram | box | stacked_bar | heatmap",
      "title": "<short title>",
      "x_axis": "<column name>",
      "y_axis": "<column name or [col1,col2]>",
      "purpose": "<one-sentence explanation of insight this chart provides>",
      "notes": "<optional - e.g., 'aggregate monthly', 'top 10 categories only', 'log scale'>"
    }
  ],
  "key_insights": [
    "<concise insight 1>",
    "<concise insight 2>"
  ],// array of 3-5 insights
  "recommendations": [
    "<actionable recommendation 1 (prioritized)>",
    "<actionable recommendation 2>"
  ]// array of 3-5 recommendations
}

Additional guidance for generating content:
- Produce 8-12 KPIs. Prioritize those that tie metrics (revenue, conversion, retention, defect rates, cycle time, cost per unit) to business levers.
- When suggesting visualizations, prefer clear, modern charts (time series = line, distribution = histogram/box, categorical comparison = bar/stacked_bar, relationships = scatter with trendline).
- For multi-series charts (e.g., revenue by product over time) propose aggregation (monthly/weekly) and a max number of series to display (e.g., top 8 categories).
- Use clear naming and avoid abbreviations unless obvious. If you detect currency, include the currency symbol in `current_value` (if present in data).
- Provide `notes` for charts if pre-processing is needed (e.g., "fill missing dates with 0", "convert to monthly sums").
- For `trend` calculation: if timestamps exist, compute percentage change between the most recent period and the prior period (or average of last 3 periods) to determine direction. If insufficient historical data, set `trend` to "neutral".
- If numeric columns contain extreme outliers, add a short note in `recommendations` to validate or trim outliers before operationalizing KPI thresholds.

Error handling:
- If dataset is empty, respond with JSON where `business_context` = "empty_dataset" and `meaningful_kpis`, `recommended_visualizations`, `key_insights`, `recommendations` are empty arrays.
- If schema contains fewer than 2 columns, return JSON with `business_context` = "insufficient_columns" and list columns present.

Visual styling guidance (for downstream graph rendering teams):
- Prefer clean fonts, minimal gridlines, clear axis labels, and tooltips for explanations.
- Default chart height: 400px. For dashboards, recommend compact KPI cards (no heavy tables).
- Color guidance: use diverging palettes for positive/negative metrics and sequential palettes for ordered metrics (trend lines).

Example KPI entry (for guidance only; DO NOT output examples in final run):
{
  "name": "Monthly Recurring Revenue (MRR)",
  "description": "Total recurring revenue aggregated monthly from subscription plans.",
  "calculation": "SUM(monthly_subscription_fee) grouped by month",
  "business_value": "Key indicator of subscription growth and revenue stability.",
  "current_value": "₹123,456.00",
  "trend": "positive"
}
"""


class AIKPIAnalyzer:
    """Uses the big Business Analyst prompt to generate KPI JSON."""

    def __init__(self, api_key: str):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=get_openrouter_api_key(),
        )
        self.model_name = "openai/gpt-4.1-mini"

    def _build_schema_info(self, df: pd.DataFrame) -> dict:
        cols_info = []
        for col in df.columns:
            series = df[col]
            info = {
                "name": col,
                "dtype": str(series.dtype),
                "null_count": int(series.isna().sum()),
                "non_null_count": int(series.notna().sum()),
                "unique_values": int(series.nunique(dropna=True)),
                "sample_values": [str(v) for v in series.dropna().unique()[:5]],
            }
            cols_info.append(info)

        return {
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "columns": cols_info,
        }

    def _build_sample_data(self, df: pd.DataFrame):
        if df.empty:
            return []
        return df.head(5).to_dict(orient="records")

    def analyze(self, df: pd.DataFrame, context: str = "") -> dict:
        # Handle empty / tiny datasets ourselves (as per instructions)
        if df.empty:
            return {
                "business_context": "empty_dataset",
                "meaningful_kpis": [],
                "recommended_visualizations": [],
                "key_insights": [],
                "recommendations": [],
            }

        if len(df.columns) < 2:
            return {
                "business_context": "insufficient_columns",
                "meaningful_kpis": [],
                "recommended_visualizations": [],
                "key_insights": [],
                "recommendations": [],
            }

        schema_obj = self._build_schema_info(df)
        sample_data = self._build_sample_data(df)

        schema_json = json.dumps(schema_obj, indent=2)
        sample_json = json.dumps(sample_data, indent=2)
        ctx = context or ""

        # Use .replace() to substitute placeholders without messing with braces
        prompt = (
            BUSINESS_ANALYST_PROMPT
            .replace("{schema}", schema_json)
            .replace("{sample_data}", sample_json)
            .replace("{context}", ctx)
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2200,
            )
            content = response.choices[0].message.content.strip()

            # Try direct JSON parse
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Try extracting JSON substring
                start = content.find("{")
                end = content.rfind("}") + 1
                if start != -1 and end != -1:
                    json_str = content[start:end]
                    return json.loads(json_str)

        except Exception as e:
            st.warning(f"AI KPI analysis failed: {e}")

        # Fallback minimal structure
        return {
            "business_context": "fallback_analysis",
            "meaningful_kpis": [],
            "recommended_visualizations": [],
            "key_insights": [
                f"Dataset has {len(df)} rows and {len(df.columns)} columns."
            ],
            "recommendations": [
                "Review dataset schema and rerun KPI analysis when connectivity is stable."
            ],
        }


@st.cache_data(show_spinner="💡 Generating AI KPI insights...")
def run_ai_kpi_analysis(df: pd.DataFrame, context: str = "") -> dict:
    api_key = get_openrouter_api_key()
    analyzer = AIKPIAnalyzer(api_key)
    return analyzer.analyze(df, context=context)


# ======================================================
# ⚙️ Streamlit Setup & State
# ======================================================
st.set_page_config(
    page_title="AI Conversational Data Tool",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "df" not in st.session_state:
    st.session_state.df = None
if "original_df" not in st.session_state:
    st.session_state.original_df = None
if "last_run_result" not in st.session_state:
    st.session_state.last_run_result = None
if "last_generated_code" not in st.session_state:
    st.session_state.last_generated_code = None
if "last_user_input" not in st.session_state:
    st.session_state.last_user_input = ""
if "error_occurred" not in st.session_state:
    st.session_state.error_occurred = False
if "kpi_output" not in st.session_state:
    st.session_state.kpi_output = None
if "dashboard_page" not in st.session_state:
    st.session_state.dashboard_page = 0
if "kpi_page" not in st.session_state:
    st.session_state.kpi_page = 0


# ======================================================
# 🧱 Sidebar
# ======================================================
with st.sidebar:
    st.title("🚀 AI-Powered Data Tool")
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if st.session_state.df is not None:
        st.download_button(
            "📥 Download Modified CSV",
            st.session_state.df.to_csv(index=False).encode("utf-8"),
            "modified_data.csv",
            "text/csv",
        )
        if st.button("🔄 Reset Data"):
            st.session_state.df = st.session_state.original_df.copy()
            st.session_state.last_run_result = None
            st.session_state.last_generated_code = None
            st.session_state.last_user_input = ""
            st.session_state.error_occurred = False
            st.session_state.kpi_output = None
            st.session_state.dashboard_page = 0
            st.session_state.kpi_page = 0
            st.success("Data has been reset.")
            st.rerun()

    with st.expander("💡 Example Prompts", expanded=True):
        st.markdown(
            """
        **Basic Exploration:**
        - `Describe the dataset`
        - `Show the first 10 rows`
        - `List all who work in the Production department`
        
        **Analysis & Aggregation:**
        - `What is the average salary by department?`
        - `Show the top 10 highest paid employees`

        **Visualization:**
        - `Plot a bar chart of the top 5 departments by employee count`
        - `Plot a scatter plot of Age vs Salary`
        
        **Data Modification:**
        - `Create a new column called 'Full Name'`
        - `Delete rows where 'Salary' is less than 50000`
        """
        )


# ======================================================
# 📥 Load Data
# ======================================================
if uploaded_file:
    if st.session_state.df is None or uploaded_file.name != st.session_state.get("file_name"):
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

            st.session_state.df = df
            st.session_state.original_df = df.copy()
            st.session_state.file_name = uploaded_file.name

            st.session_state.last_run_result = None
            st.session_state.last_generated_code = None
            st.session_state.last_user_input = ""
            st.session_state.error_occurred = False
            st.session_state.kpi_output = None
            st.session_state.dashboard_page = 0
            st.session_state.kpi_page = 0
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            st.stop()
else:
    st.info("👋 Upload a CSV file in the sidebar to begin.")
    st.stop()


# ======================================================
# 1️⃣ Data Preview
# ======================================================
st.subheader("Current Data Preview")
st.dataframe(st.session_state.df.head())
st.divider()


# ======================================================
# 2️⃣ AI KPI Section (with pagination, full fields)
# ======================================================
st.subheader("1. 🤖 AI Analysis & KPIs")

col_btn, col_ctx = st.columns([1, 3])
with col_btn:
    if st.button("Generate AI KPI Insights"):
        try:
            st.session_state.kpi_output = run_ai_kpi_analysis(st.session_state.df)
            st.session_state.kpi_page = 0
        except Exception as e:
            st.error(f"Failed to run AI KPI analysis: {e}")

analysis = st.session_state.kpi_output

if analysis:
    # Business context
    st.markdown("### 🏢 Business Context")
    st.write(analysis.get("business_context", "No context available."))

    # Paginated KPIs
    kpis = analysis.get("meaningful_kpis", [])
    if kpis:
        st.markdown("### 📌 Key KPIs")

        KPIS_PER_PAGE = 4
        start = st.session_state.kpi_page * KPIS_PER_PAGE
        end = start + KPIS_PER_PAGE
        page_kpis = kpis[start:end]

        for k in page_kpis:
            st.markdown(f"#### {k.get('name', '')}")
            st.write(k.get("description", ""))
            st.write(f"**Calculation:** {k.get('calculation', '-')}")
            st.write(f"**Current Value:** {k.get('current_value', '-')}")
            st.write(f"**Business Value:** {k.get('business_value', '-')}")
            st.write(f"**Trend:** {k.get('trend', 'neutral')}")
            st.markdown("---")

        # Show More / Reset buttons
        btn_cols = st.columns(2)
        with btn_cols[0]:
            if end < len(kpis):
                if st.button("➕ Show More KPIs"):
                    st.session_state.kpi_page += 1
                    st.rerun()
        with btn_cols[1]:
            if st.session_state.kpi_page > 0:
                if st.button("🔄 Reset KPI View"):
                    st.session_state.kpi_page = 0
                    st.rerun()
    else:
        st.info("No KPIs returned by the AI.")

    # Key insights
    insights = analysis.get("key_insights", [])
    if insights:
        st.markdown("### 🔍 Key Insights")
        for ins in insights:
            st.markdown(f"- {ins}")

    # Recommendations
    recs = analysis.get("recommendations", [])
    if recs:
        st.markdown("### 📝 Recommendations")
        for r in recs:
            st.markdown(f"- {r}")

st.divider()


# ======================================================
# 3️⃣ Dashboard
# ======================================================
st.subheader("2. Dashboard")
generate_dashboard(st.session_state.df)
st.divider()


# ======================================================
# 4️⃣ Custom Query
# ======================================================
st.subheader("3. Custom Query")
st.markdown("Ask a question or give a command about the dataset.")

user_input = st.text_area("Ask a question:", key="user_input_area")

if st.button("Run", type="primary"):
    if user_input.strip():
        st.session_state.error_occurred = False
        st.session_state.last_run_result = None

        try:
            schema = build_schema(st.session_state.df)
            raw_code = generate_code(schema, user_input)
            code_to_run = strip_fences(raw_code)

            st.session_state.last_generated_code = code_to_run
            st.session_state.last_user_input = user_input

            local_vars = {"df": st.session_state.df.copy(), "pd": pd, "px": px}
            exec(code_to_run, {}, local_vars)

            if not st.session_state.df.equals(local_vars["df"]):
                st.session_state.df = local_vars["df"]

            st.session_state.last_run_result = local_vars.get("result", None)
        except Exception as e:
            st.session_state.error_occurred = True
            st.error(f"An error occurred while running the code: {e}")
    else:
        st.warning("Please enter a question or command.")

# Persistent result display
if st.session_state.last_user_input:
    st.divider()
    st.subheader("Result from your Custom Query")

    if st.session_state.last_generated_code:
        with st.expander("🤖 View AI Generated Code", expanded=False):
            st.code(st.session_state.last_generated_code, language="python")

    if not st.session_state.error_occurred:
        if st.session_state.last_run_result is not None:
            st.subheader("✅ Result")
            display_results(st.session_state.last_run_result, st.session_state.last_user_input)
        else:
            st.success("Command executed successfully. The data has been updated.")
