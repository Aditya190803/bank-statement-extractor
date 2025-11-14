import io
import re
import string
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st
from rapidfuzz import fuzz, process

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - surfaced in UI
    fitz = None

st.set_page_config(
    page_title="Bank Statement – Customer Name Matcher",
    page_icon="📄",
    layout="wide",
)

APP_ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = APP_ROOT / "sample_data"
SAMPLE_FILES = {
    "pdf": SAMPLE_DIR / "bank_statement.pdf",
    # Single CSV that contains both names and additional customer details
    "customers": SAMPLE_DIR / "customer_details.csv",
}

_NAME_PUNCT_TRANSLATOR = str.maketrans("", "", string.punctuation)


@st.cache_data(show_spinner=False)
def load_tabular_data(data: bytes, filename: Optional[str] = None) -> pd.DataFrame:
    """Load a CSV or Excel file from raw bytes."""
    buffer = io.BytesIO(data)
    suffix = ""
    if filename:
        suffix = Path(filename).suffix.lower()

    if suffix in {".xlsx", ".xls", ".xlsm"}:
        buffer.seek(0)
        return pd.read_excel(buffer)

    if suffix == ".csv":
        buffer.seek(0)
        return pd.read_csv(buffer)

    # Fallback: try CSV first, then Excel if CSV parsing fails.
    buffer.seek(0)
    try:
        return pd.read_csv(buffer)
    except Exception:
        buffer.seek(0)
        return pd.read_excel(buffer)


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from a PDF using PyMuPDF."""
    if fitz is None:
        raise ImportError(
            "PyMuPDF (package `pymupdf`) is required to extract text from PDFs."
        )

    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        text_chunks = []
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            text_chunks.append(page.get_text("text"))
    return "\n".join(text_chunks)


def sanitize_text(text: str) -> str:
    """Sanitize arbitrary PDF-extracted text before regex matches.

    - Remove non-printable characters
    - Normalize whitespace
    """
    if not text:
        return ""
    printable = "".join(ch for ch in text if ch.isprintable())
    # Insert spacing between characters when PDFs omit column gaps (e.g. letter-digit joins).
    printable = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", printable)
    printable = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", printable)
    printable = re.sub(r"\s+", " ", printable)
    return printable.strip()


def extract_candidate_names(text: str) -> List[str]:
    """Pull out likely customer names from raw PDF text."""
    if not text:
        return []

    # Support name components with hyphens or apostrophes (e.g. O'Neil, Rivera-Santos)
    name_word = r"[A-Z][a-z]+(?:['-][A-Z][a-z]+)?"

    # Detect standalone title-case and all-caps name phrases
    title_case_pattern = re.compile(
        rf"({name_word}(?:\s+{name_word}){{1,3}})(?=[^A-Za-z]|$)"
    )
    all_caps_pattern = re.compile(r"\b([A-Z]{2,}(?:\s+[A-Z]{2,}){1,3})(?=[^A-Z]|$)")

    # Focus on transactional phrases that explicitly reference customers
    transaction_pattern = re.compile(
        rf"(?:ACH Payment to|Transfer to|Transfer from|Wire transfer from)\s+({name_word}(?:\s+{name_word}){{1,3}})"
    )

    candidates = set()

    for match in transaction_pattern.findall(text):
        cleaned = match.strip()
        if _is_plausible_name(cleaned):
            candidates.add(cleaned)

    for match in title_case_pattern.findall(text):
        cleaned = match.strip()
        if _is_plausible_name(cleaned):
            candidates.add(cleaned)

    for match in all_caps_pattern.findall(text):
        cleaned = match.title().strip()
        if _is_plausible_name(cleaned):
            candidates.add(cleaned)

    # Return sorted unique list, in their original form; normalization happens later
    return sorted(candidates)


def _is_plausible_name(name: str) -> bool:
    """Basic heuristics to reduce false positives from PDF text."""
    parts = name.split()
    if len(parts) < 2:
        return False
    if any(len(part) == 1 for part in parts):
        return False
    if any(part.isdigit() for part in parts):
        return False
    return True


def normalize_name(name: str) -> str:
    """Normalize a name for fuzzy comparison."""
    lowered = name.lower().strip()
    normalized = lowered.translate(_NAME_PUNCT_TRANSLATOR)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _ensure_session_state_option(
    key: str,
    choices: List[str],
    preferred: Optional[List[str]] = None,
) -> None:
    """Ensure Streamlit session state keeps a valid selection for dynamic selectboxes."""
    if not choices:
        st.session_state.pop(key, None)
        return

    current = st.session_state.get(key)
    if current in choices:
        return

    preferred = preferred or []
    for candidate in preferred:
        if candidate in choices:
            st.session_state[key] = candidate
            return

    st.session_state[key] = choices[0]


def _stringify_series(series: pd.Series) -> pd.Series:
    """Return a trimmed string series for consistent comparisons and filtering."""
    cleaned = series.fillna("").astype(str).str.strip()
    return cleaned.replace({"nan": ""})


def match_names(
    pdf_names: List[str],
    customers: pd.DataFrame,
    threshold: int,
    customer_name_column: str,
) -> pd.DataFrame:
    """Run fuzzy matching between PDF-derived names and customer records."""
    if customers.empty or not pdf_names:
        return pd.DataFrame(columns=["PDF Name", "Matched Customer", "Match Score"])

    working = customers.copy()
    working["__normalized"] = working[customer_name_column].astype(str).apply(normalize_name)

    choices = working["__normalized"].tolist()
    matches = []

    for pdf_name in pdf_names:
        normalized_pdf = normalize_name(pdf_name)
        if not normalized_pdf:
            continue

        result = process.extractOne(
            query=normalized_pdf,
            choices=choices,
            scorer=fuzz.WRatio,
        )

        if result is None:
            continue

        matched_value, score, index = result
        if score < threshold:
            continue

        row = working.iloc[index]
        matches.append(
            {
                "PDF Name": pdf_name,
                "Matched Customer": row[customer_name_column],
                "Match Score": round(float(score), 2),
            }
        )

    if not matches:
        return pd.DataFrame(columns=["PDF Name", "Matched Customer", "Match Score"])

    matched_df = pd.DataFrame(matches)
    matched_df.sort_values(by=["Match Score", "Matched Customer"], ascending=[False, True], inplace=True)
    matched_df.drop_duplicates(subset="Matched Customer", keep="first", inplace=True)
    matched_df.reset_index(drop=True, inplace=True)
    return matched_df


def merge_with_details(
    matches: pd.DataFrame,
    customers: pd.DataFrame,
    customers_name_column: str,
) -> pd.DataFrame:
    """Merge matched names with their additional details."""
    if matches.empty:
        return matches
    merged = matches.merge(
        customers,
        left_on="Matched Customer",
        right_on=customers_name_column,
        how="left",
    )
    if customers_name_column in merged.columns:
        merged.drop(columns=[customers_name_column], inplace=True)
    return merged


def main() -> None:
    st.title("Bank Statement – Customer Name Matcher")
    st.markdown(
        """
        Upload a bank statement PDF and customer datasets to automatically identify overlapping
        names. Matched entries include confidence scores and customer details that you can review
        or export.
        """
    )

    st.sidebar.header("Matching Settings")
    use_samples = st.sidebar.toggle(
        "Use sample files",
        value=False,
        help="Automatically load the demo PDF and CSVs shipped with the app for a quick test run.",
    )
    threshold = st.sidebar.slider(
        "Minimum match score",
        min_value=60,
        max_value=100,
        value=85,
        step=1,
        help="Lower the threshold to surface more matches, raise it to focus on high-confidence pairs.",
    )
    
    show_file_details = st.sidebar.toggle(
        "Show file details",
        value=False,
        help="Display a preview of the loaded files (sample or uploaded).",
    )
    
    st.sidebar.info(
        "Processing happens locally inside this Streamlit session. No files leave your machine."
    )

    st.subheader("1. Upload files")
    pdf_file = st.file_uploader(
        "Bank Statement (PDF)",
        type=["pdf"],
        accept_multiple_files=False,
        disabled=use_samples,
    )

    customers_file = st.file_uploader(
        "Customers (CSV or Excel) — names and details in one file",
        type=["csv", "xlsx", "xls", "xlsm"],
        accept_multiple_files=False,
        key="customers-csv",
        help="Upload a single CSV containing CustomerName and any detail columns (including optional employee/filter column).",
        disabled=use_samples,
    )

    pdf_bytes: Optional[bytes] = None
    customers_df: Optional[pd.DataFrame] = None
    customers_error: Optional[str] = None

    if use_samples:
        missing = [label for label, path in SAMPLE_FILES.items() if not path.exists()]
        if missing:
            missing_labels = ", ".join(missing)
            st.error(
                f"Sample files missing: {missing_labels}. Upload your own files or restore the sample_data folder."
            )
        else:
            pdf_bytes = SAMPLE_FILES["pdf"].read_bytes()
            try:
                customers_df = load_tabular_data(
                    SAMPLE_FILES["customers"].read_bytes(),
                    SAMPLE_FILES["customers"].name,
                )
            except Exception as exc:
                customers_error = str(exc)
            st.info(
                "Using bundled demo files from the `sample_data` directory. Toggle off to upload your own."
            )
    else:
        if pdf_file is not None:
            pdf_bytes = pdf_file.getvalue()
        if customers_file is not None:
            try:
                customers_df = load_tabular_data(
                    customers_file.getvalue(),
                    customers_file.name,
                )
            except Exception as exc:
                customers_error = str(exc)

    if show_file_details:
        st.subheader("📋 File Details Preview")
        if customers_df is not None:
            st.write("**Customers (combined CSV):**")
            st.dataframe(customers_df, width='stretch', hide_index=True)
        elif customers_error:
            st.error(f"Unable to preview the customers CSV: {customers_error}")
        else:
            st.info("Upload a customers CSV to preview its contents.")

    st.subheader("2. Map columns")
    customer_name_column: Optional[str] = None
    # We use 'sub_broker' filtering via a dedicated multi-select below

    if customers_error and not show_file_details:
        st.error(f"Unable to read the customers CSV: {customers_error}")

    if customers_df is None:
        st.info("Upload the customers CSV to configure column mappings.")
    else:
        customer_columns = list(customers_df.columns)

        _ensure_session_state_option(
                "customer_name_column",
                customer_columns,
                preferred=["long_name", "CustomerName", "Customer Name", "Name"],
            )
        customer_name_column = st.selectbox(
            "Customer names column (customers CSV)",
            customer_columns,
            key="customer_name_column",
            help="Select the column that contains customer names in the uploaded customers CSV.",
        )

        # For a single customers CSV we only need to select the name column once

        # We no longer ask users to pick a generic filter column - use sub_broker multi-select instead

        # If the dataset contains a 'sub_broker' column we allow multi-selection
        sub_broker_col_present = "sub_broker" in customer_columns
        selected_sub_brokers = []
        if sub_broker_col_present:
            sub_broker_options = sorted(customers_df["sub_broker"].dropna().astype(str).unique())
            selected_sub_brokers = st.multiselect(
                "Filter by sub_broker (multi-select)",
                options=sub_broker_options,
                default=sub_broker_options,
                help="Limit matching to customers for the selected sub_broker(s).",
            )
            filter_counts_df = (
                customers_df[["sub_broker", customer_name_column]]
                .dropna(subset=[customer_name_column])
                .copy()
            )
            filter_counts_df["sub_broker"] = _stringify_series(
                filter_counts_df["sub_broker"]
            )
            filter_counts_df[customer_name_column] = _stringify_series(
                filter_counts_df[customer_name_column]
            )
            filter_counts_df = filter_counts_df[filter_counts_df["sub_broker"] != ""]

            filter_counts = (
                filter_counts_df.groupby("sub_broker")[customer_name_column]
                .nunique()
                .sort_index()
            )
            total_unique_customers = int(filter_counts_df[customer_name_column].nunique())

            if filter_counts.empty:
                st.info("No values detected in the 'sub_broker' column for filtering.")
        else:
            # no selection or values in 'sub_broker', clear any previous selection state
            st.session_state.pop("filter_column", None)

    st.markdown("---")
    if st.button("Extract & Match", type="primary", width="stretch"):
        if pdf_bytes is None:
            st.warning("Please upload a bank statement PDF before processing.")
            return
        if customers_df is None:
            if customers_error:
                st.error(f"Unable to read the customers CSV: {customers_error}")
            else:
                st.warning("Please upload a customers CSV before processing.")
            return
        if not customer_name_column:
            st.warning("Select the customer name column before processing.")
            return

        candidate_names: List[str] = []
        matches_df = pd.DataFrame()
        merged_df = pd.DataFrame()
        statement_text = ""

        with st.spinner("Extracting text and running matches..."):
            try:
                statement_text = extract_pdf_text(pdf_bytes)
            except Exception as exc:  # surfaced to user for debugging
                st.error(f"Unable to extract text from the PDF: {exc}")
                return

            statement_text = sanitize_text(statement_text)
            candidate_names = extract_candidate_names(statement_text)

            filtered_customers_df = customers_df.copy()
            applied_filter: Optional[list] = None
            # Apply sub_broker multi-select filter if available and a subset is selected
            if sub_broker_col_present and selected_sub_brokers:
                if set(selected_sub_brokers) != set(sub_broker_options):
                    normalized_sub_broker = _stringify_series(filtered_customers_df["sub_broker"])
                    filter_mask = normalized_sub_broker.isin(selected_sub_brokers)
                    filtered_customers_df = filtered_customers_df.loc[filter_mask].copy()
                    applied_filter = list(selected_sub_brokers)
                    if filtered_customers_df.empty:
                        st.warning(
                            f"No customer records found for the selected sub_broker(s): {', '.join(selected_sub_brokers)}."
                        )
                        return

            filtered_customers_df = filtered_customers_df.drop_duplicates(
                subset=[customer_name_column]
            )
            matches_df = match_names(
                candidate_names,
                filtered_customers_df,
                threshold,
                customer_name_column,
            )
            merged_df = merge_with_details(
                matches_df,
                filtered_customers_df,
                customer_name_column,
            )

        st.subheader("2. Extracted names")
        st.caption("Names detected in the PDF after basic cleaning.")
        if candidate_names:
            extracted_preview = pd.DataFrame({"Extracted Names": candidate_names})
            st.dataframe(extracted_preview, width="stretch", hide_index=True)
        else:
            st.info("No plausible customer names were found in the uploaded PDF.")

        st.subheader("3. Matched customers")
        if sub_broker_col_present:
            if applied_filter:
                st.caption(f"Sub-broker filter applied: {', '.join(applied_filter)}")
            else:
                st.caption("Sub-broker filter: all sub_brokers")

        total_extracted = len(candidate_names)
        total_matches = len(matches_df)
        match_rate = (total_matches / total_extracted * 100) if total_extracted else 0.0
        st.caption(
            f"Matching stats: {total_extracted} names extracted · {total_matches} matched · Match rate {match_rate:.1f}%"
        )

        if matches_df.empty:
            st.warning(
                "No matches cleared the selected score threshold. Try adjusting the score, sub_broker filter, or reviewing the input data."
            )
            return

        st.success(f"Matched {len(matches_df)} customer(s) with a minimum score of {threshold}%.")
        st.dataframe(matches_df, width="stretch", hide_index=True)

        matches_csv = matches_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download matched names CSV",
            data=matches_csv,
            file_name="matched_names.csv",
            mime="text/csv",
        )

        st.subheader("4. Matched customer details")
        if merged_df.empty:
            st.info("No additional details were found for the matched customers.")
        else:
            st.dataframe(merged_df, width="stretch", hide_index=True)
            details_csv = merged_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download matched details CSV",
                data=details_csv,
                file_name="matched_customer_details.csv",
                mime="text/csv",
            )

        with st.expander("PDF text preview"):
            snippet = statement_text[:5000]
            suffix = "..." if len(statement_text) > 5000 else ""
            st.text(snippet + suffix)


if __name__ == "__main__":
    main()
