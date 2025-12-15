import streamlit as st
import pandas as pd
from openai import OpenAI
import plotly.express as px
import io

# --- Helper Functions ---

def strip_fences(text: str) -> str:
    """Removes markdown code fences and language identifiers from a string."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()[1:]
        if lines and lines[-1].strip() == "```": lines = lines[:-1]
        t = "\n".join(lines).strip()
    if t.startswith("'''"):
        lines = t.splitlines()[1:]
        if lines and lines[-1].strip() == "'''": lines = lines[:-1]
        t = "\n".join(lines).strip()
    first_line = t.splitlines()[0] if t.splitlines() else ""
    if first_line.lower().strip() in ("python", "py"):
        t = "\n".join(t.splitlines()[1:]).lstrip()
    return t

def build_schema(df):
    """
    Build schema with dtype + sample unique values (up to 10).
    This dramatically improves LLM accuracy for filtering commands.
    """
    schema = {}
    for col in df.columns:
        series = df[col]

        if series.dtype == 'object' or str(series.dtype).startswith('category'):
            uniques = series.dropna().unique().tolist()
            uniques = uniques[:10]  # limit to avoid large prompts
            schema[col] = {
                "dtype": str(series.dtype),
                "sample_values": [str(v) for v in uniques]
            }
        else:
            schema[col] = {
                "dtype": str(series.dtype)
            }
    return schema

def generate_dashboard(df):
    """
    Automatically generates charts for all columns in batches of 4.
    User can click "Show More" to load more charts.
    """

    if df is None or df.empty:
        return

    st.markdown("### 📊 Data Overview Dashboard")

    # Initialize session state to track pagination
    if "dashboard_page" not in st.session_state:
        st.session_state.dashboard_page = 0

    numerical_cols = df.select_dtypes(include=['number']).columns
    numerical_cols = [
        c for c in numerical_cols 
        if not any(x in c.lower() for x in ['id', 'code', 'key', 'empid'])
    ]

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    chart_list = []

    # Generate chart definitions (not plotting yet)
    for col in categorical_cols:
        unique_count = df[col].nunique()
        if 1 < unique_count <= 6:
            chart_list.append(("pie", col))
        elif 6 < unique_count <= 30:
            chart_list.append(("bar", col))

    for col in numerical_cols:
        if df[col].nunique() > 5:
            chart_list.append(("hist", col))

    if not chart_list:
        st.info("No suitable columns found for charting.")
        return

    # Pagination: 4 charts per page
    charts_per_page = 4
    start = st.session_state.dashboard_page * charts_per_page
    end = start + charts_per_page

    page_charts = chart_list[start:end]

    # Show the charts in a 2-column layout
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

    # Show More button (if more charts remain)
    if end < len(chart_list):
        if st.button("➕ Show More Charts"):
            st.session_state.dashboard_page += 1
            st.rerun()

    if st.session_state.dashboard_page > 0:
        if st.button("🔄 Reset Dashboard View"):
            st.session_state.dashboard_page = 0
            st.rerun()



@st.cache_data(show_spinner="AI is thinking...")
def generate_code(schema: str, user_input: str):
    """Calls the AI model to generate Python code."""
    model_to_use = "nvidia/nemotron-nano-9b-v2:free"
    
    prompt = f'''You are an expert Python data analyst.
Here is the schema of the Pandas DataFrame (column names, dtype, sample unique values):
{schema}

User's request: "{user_input}"

Instructions:
1. Analyze the user's request. It can be a question (read operation) or a command (write/update operation).
2. Write a snippet of Python code to perform the requested operation on the `df` DataFrame.
3. For questions that require a visual answer, generate a Plotly figure and assign it to a `result` variable.
4. For questions that require a tabular or single-value answer, assign the final output to a `result` variable.
5. For commands that modify the DataFrame (e.g., adding a column, deleting rows), the code should modify the `df` in place. It does not need to produce a `result` variable.
6. When dealing with dates, always use pd.to_datetime with a specified format if possible, or `dayfirst=True`, and `errors='coerce'`.
7. Return ONLY the Python code snippet, without any markdown formatting or explanations.
'''
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=st.secrets["OPENROUTER_API_KEY"])
    response = client.chat.completions.create(model=model_to_use, messages=[{"role": "user", "content": prompt}], temperature=0.0)
    return response.choices[0].message.content or ""

def display_results(result, user_input=""):
    """Dynamically displays the result based on its type."""
    if result is None:
        st.success("Your command was executed successfully.")
        st.dataframe(st.session_state.df.head())
        return

    is_person_search = any(keyword in user_input.lower() for keyword in ["who is", "find", "search for", "employee", "person", "details for"])

    if is_person_search and isinstance(result, pd.DataFrame) and len(result) > 1:
        display_person_selector(result)
    elif isinstance(result, pd.DataFrame):
        st.dataframe(result)
    elif isinstance(result, (int, float, str, pd.Series)):
        st.write(result)
    elif "plotly" in str(type(result)):
        st.plotly_chart(result, use_container_width=True)
    else:
        st.write(result)

def display_person_selector(result_df):
    """Intelligently finds name/dept/id columns and displays a selector."""
    st.info("Multiple people found. Please select one to see details.")
    
    cols = result_df.columns
    
    def find_col(keywords):
        for col in cols:
            for keyword in keywords:
                if keyword in col.lower():
                    return col
        return None

    first_name_col = find_col(['first', 'fname'])
    last_name_col = find_col(['last', 'lname'])
    dept_col = find_col(['department', 'dept'])
    id_col = find_col(['id', 'empid'])
    
    def create_display_name(row):
        fname = row[first_name_col] if first_name_col and first_name_col in row else ""
        lname = row[last_name_col] if last_name_col and last_name_col in row else ""
        dept = row[dept_col] if dept_col and dept_col in row else "N/A"
        empid = row[id_col] if id_col and id_col in row else "N/A"
        return f"{fname} {lname} (Dept: {dept}) — ID: {empid}".strip()
        
    result_copy = result_df.copy()
    result_copy["display_name"] = result_copy.apply(create_display_name, axis=1)
    options = ["- Select -"] + result_copy["display_name"].tolist()
    selection = st.selectbox("Select a person:", options=options)
    
    if selection != "- Select -":
        sel_row_df = result_copy[result_copy["display_name"] == selection]
        if not sel_row_df.empty:
            original_idx = sel_row_df.index[0]
            st.write("Details:")
            st.dataframe(result_df.loc[[original_idx]])

# --- Page Configuration ---
st.set_page_config(page_title="AI Conversational Data Tool", layout="wide", initial_sidebar_state="expanded")

# --- Session State Initialization ---
if "df" not in st.session_state: st.session_state.df = None
if "original_df" not in st.session_state: st.session_state.original_df = None
if "last_run_result" not in st.session_state: st.session_state.last_run_result = None
if "last_generated_code" not in st.session_state: st.session_state.last_generated_code = None
if "last_user_input" not in st.session_state: st.session_state.last_user_input = ""
if "error_occurred" not in st.session_state: st.session_state.error_occurred = False

# --- Sidebar ---
with st.sidebar:
    st.title("🚀 AI-Powered Data Tool")
    st.write("Upload a CSV file and use natural language to interact with your data.")
    uploaded_file = st.file_uploader("1. Upload your CSV file", type="csv")
    
    if st.session_state.df is not None:
        st.download_button("📥 Download Modified CSV", st.session_state.df.to_csv(index=False).encode('utf-8'), 'modified_data.csv', 'text/csv')
        if st.button("🔄 Reset Data"):
            st.session_state.df = st.session_state.original_df.copy()
            for key in ["last_run_result", "last_generated_code", "last_user_input"]:
                if key in st.session_state:
                    st.session_state[key] = None
            st.success("Data has been reset to its original state.")
            st.rerun()
            
    with st.expander("💡 Example Prompts", expanded=True):
        st.markdown("""
        **Basic Exploration:**
        - `Describe the dataset`
        - `Show the first 10 rows`
        - `Find the person with the last name 'Smith'`
        
        **Analysis & Aggregation:**
        - `What is the average salary by department?`
        - `Show the top 10 highest paid employees`

        **Visualization:**
        - `Plot a bar chart of the top 5 departments by employee count`
        - `Plot a scatter plot of Age vs Salary`
        
        **Data Modification:**
        - `Create a new column called 'Full Name'`
        - `Delete rows where 'Salary' is less than 50000`
        """)

# --- Main Page Logic ---
if uploaded_file:
    if st.session_state.df is None or uploaded_file.name != st.session_state.get("file_name"):
        try:
            df = pd.read_csv(uploaded_file)
            # --- Standardize column names robustly ---
            original_cols = df.columns
            df.columns = [col.strip().replace(' ', '_').lower() for col in original_cols]
            
            st.session_state.df = df
            st.session_state.original_df = df.copy()
            st.session_state.file_name = uploaded_file.name
            
            # Reset states
            for key in ["last_run_result", "last_generated_code", "last_user_input"]:
                if key in st.session_state:
                    st.session_state[key] = None
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            st.stop()
else:
    st.info("👋 Welcome! Upload a CSV file in the sidebar to begin.")
    st.stop()

st.subheader("Current Data Preview:")
st.dataframe(st.session_state.df.head())

# --- NEW: Dashboard Section (Replaces Dropdown) ---
st.divider()
generate_dashboard(st.session_state.df)
st.divider()

# --- Custom Query Section ---
st.subheader("2. Custom Query")
st.markdown("Ask a more detailed question or give a command below.")
user_input = st.text_area("Ask a question:", key="user_input_area")

if st.button("Run", type="primary"):
    if user_input:
        st.session_state.error_occurred = False
        st.session_state.last_run_result = None
        try:
            schema = build_schema(st.session_state.df)

            generated_code = generate_code(schema, user_input)
            code_to_execute = strip_fences(generated_code)
            st.session_state.last_generated_code = code_to_execute
            st.session_state.last_user_input = user_input
            
            local_vars = {"df": st.session_state.df.copy(), "pd": pd, "px": px}
            exec(code_to_execute, {}, local_vars)
            
            # Check if df was modified
            if not st.session_state.df.equals(local_vars['df']):
                st.session_state.df = local_vars['df']
            
            st.session_state.last_run_result = local_vars.get("result", None)
            
        except Exception as e:
            st.session_state.error_occurred = True
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question or command.")

# --- Persistent Display Block for Custom Queries ---
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