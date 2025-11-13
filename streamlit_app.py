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
    "customers": SAMPLE_DIR / "customer_names.csv",
    "details": SAMPLE_DIR / "customer_details.csv",
}

_NAME_PUNCT_TRANSLATOR = str.maketrans("", "", string.punctuation)


@st.cache_data(show_spinner=False)
def load_csv(data: bytes) -> pd.DataFrame:
    """Load a CSV file from raw bytes."""
    return pd.read_csv(io.BytesIO(data))


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from a PDF using PyMuPDF."""
    if fitz is None:
        raise ImportError(
            "PyMuPDF (package `pymupdf`) is required to extract text from PDFs."
        )

    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        text_chunks = [page.get_text("text") for page in document]
    return "\n".join(text_chunks)


def extract_candidate_names(text: str) -> List[str]:
    """Pull out likely customer names from raw PDF text."""
    if not text:
        return []

    title_case_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b")
    all_caps_pattern = re.compile(r"\b([A-Z]{2,}(?:\s+[A-Z]{2,}){1,3})\b")

    candidates = set()

    for match in title_case_pattern.findall(text):
        cleaned = match.strip()
        if _is_plausible_name(cleaned):
            candidates.add(cleaned)

    for match in all_caps_pattern.findall(text):
        cleaned = match.title().strip()
        if _is_plausible_name(cleaned):
            candidates.add(cleaned)

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
    details: pd.DataFrame,
    details_name_column: str,
) -> pd.DataFrame:
    """Merge matched names with their additional details."""
    if matches.empty:
        return matches
    merged = matches.merge(
        details,
        left_on="Matched Customer",
        right_on=details_name_column,
        how="left",
    )
    if details_name_column in merged.columns:
        merged.drop(columns=[details_name_column], inplace=True)
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

    col1, col2 = st.columns(2)
    with col1:
        customer_file = st.file_uploader(
            "Customer Names (CSV)",
            type=["csv"],
            accept_multiple_files=False,
            key="customer-csv",
            help="Upload the customer listing. You'll map the name column after upload.",
            disabled=use_samples,
        )
    with col2:
        details_file = st.file_uploader(
            "Customer Details (CSV)",
            type=["csv"],
            accept_multiple_files=False,
            key="details-csv",
            help="Upload supporting customer details. Column mapping is configured below.",
            disabled=use_samples,
        )

    pdf_bytes: Optional[bytes] = None
    customers_df: Optional[pd.DataFrame] = None
    details_df: Optional[pd.DataFrame] = None
    customer_error: Optional[str] = None
    details_error: Optional[str] = None

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
                customers_df = load_csv(SAMPLE_FILES["customers"].read_bytes())
            except Exception as exc:
                customer_error = str(exc)
            try:
                details_df = load_csv(SAMPLE_FILES["details"].read_bytes())
            except Exception as exc:
                details_error = str(exc)
            st.info(
                "Using bundled demo files from the `sample_data` directory. Toggle off to upload your own."
            )
    else:
        if pdf_file is not None:
            pdf_bytes = pdf_file.getvalue()
        if customer_file is not None:
            try:
                customers_df = load_csv(customer_file.getvalue())
            except Exception as exc:
                customer_error = str(exc)
        if details_file is not None:
            try:
                details_df = load_csv(details_file.getvalue())
            except Exception as exc:
                details_error = str(exc)

    if show_file_details:
        st.subheader("📋 File Details Preview")
        if customers_df is not None:
            st.write("**Customer Names:**")
            st.dataframe(customers_df, use_container_width=True, hide_index=True)
        elif customer_error:
            st.error(f"Unable to preview customer names CSV: {customer_error}")
        else:
            st.info("Upload a customer names CSV to preview its contents.")

        if details_df is not None:
            st.write("**Customer Details:**")
            st.dataframe(details_df, use_container_width=True, hide_index=True)
        elif details_error:
            st.error(f"Unable to preview customer details CSV: {details_error}")
        else:
            st.info("Upload a customer details CSV to preview its contents.")

    st.subheader("2. Map columns")
    customer_name_column: Optional[str] = None
    details_name_column: Optional[str] = None
    employee_column: Optional[str] = None
    employee_filter_value: str = "__all__"

    if customer_error and not show_file_details:
        st.error(f"Unable to read the customer names CSV: {customer_error}")
    if details_error and not show_file_details:
        st.error(f"Unable to read the details CSV: {details_error}")

    if customers_df is None or details_df is None:
        st.info("Upload both customer CSV files to configure column mappings.")
    else:
        customer_columns = list(customers_df.columns)
        details_columns = list(details_df.columns)

        _ensure_session_state_option(
            "customer_name_column",
            customer_columns,
            preferred=["CustomerName", "Customer Name", "Name"],
        )
        customer_name_column = st.selectbox(
            "Customer names column (names CSV)",
            customer_columns,
            key="customer_name_column",
            help="Select the column that contains customer names in the uploaded customer names CSV.",
        )

        _ensure_session_state_option(
            "details_name_column",
            details_columns,
            preferred=["CustomerName", "Customer Name", "Name"],
        )
        details_name_column = st.selectbox(
            "Customer names column (details CSV)",
            details_columns,
            key="details_name_column",
            help="Select the column that contains customer names in the customer details CSV.",
        )

        employee_column_options = [None] + details_columns
        preferred_employee = next(
            (column for column in details_columns if "employee" in column.lower()),
            None,
        )
        if st.session_state.get("employee_column") not in employee_column_options:
            st.session_state["employee_column"] = preferred_employee

        employee_column = st.selectbox(
            "Employee column (optional)",
            employee_column_options,
            format_func=lambda value: "— No employee filtering —" if value is None else value,
            key="employee_column",
            help="If selected, the app filters both CSVs to the chosen employee before matching.",
        )

        if employee_column:
            employee_counts_df = (
                details_df[[employee_column, details_name_column]]
                .dropna(subset=[details_name_column])
                .copy()
            )
            employee_counts_df[employee_column] = _stringify_series(
                employee_counts_df[employee_column]
            )
            employee_counts_df[details_name_column] = _stringify_series(
                employee_counts_df[details_name_column]
            )
            employee_counts_df = employee_counts_df[employee_counts_df[employee_column] != ""]

            employee_counts = (
                employee_counts_df.groupby(employee_column)[details_name_column]
                .nunique()
                .sort_index()
            )
            total_unique_customers = int(employee_counts_df[details_name_column].nunique())

            if not employee_counts.empty:
                employee_filter_options = ["__all__"] + employee_counts.index.tolist()
                if st.session_state.get("employee_filter_value") not in employee_filter_options:
                    st.session_state["employee_filter_value"] = "__all__"

                def _format_employee_choice(value: str) -> str:
                    if value == "__all__":
                        suffix = "customer" if total_unique_customers == 1 else "customers"
                        return f"All employees ({total_unique_customers} {suffix})"
                    count = int(employee_counts.loc[value])
                    suffix = "customer" if count == 1 else "customers"
                    return f"{value} ({count} {suffix})"

                employee_filter_value = st.selectbox(
                    "Filter customers by employee",
                    employee_filter_options,
                    key="employee_filter_value",
                    format_func=_format_employee_choice,
                    help="Choose an employee to limit matching to their customers only.",
                )
            else:
                st.info("No employee values detected in the selected column.")
                st.session_state["employee_filter_value"] = "__all__"
        else:
            st.session_state.pop("employee_filter_value", None)

    st.markdown("---")
    if st.button("Extract & Match", type="primary", use_container_width=True):
        if pdf_bytes is None:
            st.warning("Please upload a bank statement PDF before processing.")
            return
        if customers_df is None:
            if customer_error:
                st.error(f"Unable to read the customer names CSV: {customer_error}")
            else:
                st.warning("Please upload a customer names CSV before processing.")
            return
        if details_df is None:
            if details_error:
                st.error(f"Unable to read the details CSV: {details_error}")
            else:
                st.warning("Please upload a customer details CSV before processing.")
            return
        if not customer_name_column or not details_name_column:
            st.warning("Select the customer name columns before processing.")
            return

        candidate_names: List[str] = []
        matches_df = pd.DataFrame()
        merged_df = pd.DataFrame()
        statement_text = ""
        applied_employee_filter: Optional[str] = None

        with st.spinner("Extracting text and running matches..."):
            try:
                statement_text = extract_pdf_text(pdf_bytes)
            except Exception as exc:  # surfaced to user for debugging
                st.error(f"Unable to extract text from the PDF: {exc}")
                return

            candidate_names = extract_candidate_names(statement_text)

            filtered_customers_df = customers_df.copy()
            filtered_details_df = details_df.copy()
            current_employee_filter = employee_filter_value

            if employee_column and current_employee_filter != "__all__":
                normalized_details_employee = _stringify_series(
                    filtered_details_df[employee_column]
                )
                filter_mask = normalized_details_employee == current_employee_filter
                filtered_details_df = filtered_details_df.loc[filter_mask].copy()
                applied_employee_filter = current_employee_filter

                if filtered_details_df.empty:
                    st.warning(
                        f"No customer records found for employee '{current_employee_filter}'."
                    )
                    return

                allowed_names_series = _stringify_series(
                    filtered_details_df[details_name_column]
                )
                allowed_names = set(allowed_names_series[allowed_names_series != ""])
                if not allowed_names:
                    st.warning(
                        "No valid customer names available for the selected employee."
                    )
                    return

                customer_names_series = _stringify_series(
                    filtered_customers_df[customer_name_column]
                )
                filtered_customers_df = filtered_customers_df.loc[
                    customer_names_series.isin(allowed_names)
                ].copy()

                if filtered_customers_df.empty:
                    st.warning(
                        "No customer names found in the names CSV for the selected employee."
                    )
                    return
            else:
                applied_employee_filter = None

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
                filtered_details_df,
                details_name_column,
            )

        st.subheader("2. Extracted names")
        st.caption("Names detected in the PDF after basic cleaning.")
        if candidate_names:
            extracted_preview = pd.DataFrame({"Extracted Names": candidate_names})
            st.dataframe(extracted_preview, use_container_width=True, hide_index=True)
        else:
            st.info("No plausible customer names were found in the uploaded PDF.")

        st.subheader("3. Matched customers")
        if employee_column:
            if applied_employee_filter:
                st.caption(f"Employee filter applied: {applied_employee_filter}")
            else:
                st.caption("Employee filter: all employees")

        total_extracted = len(candidate_names)
        total_matches = len(matches_df)
        match_rate = (total_matches / total_extracted * 100) if total_extracted else 0.0
        st.caption(
            f"Matching stats: {total_extracted} names extracted · {total_matches} matched · Match rate {match_rate:.1f}%"
        )

        if matches_df.empty:
            st.warning(
                "No matches cleared the selected score threshold. Try adjusting the score, employee filter, or reviewing the input data."
            )
            return

        st.success(f"Matched {len(matches_df)} customer(s) with a minimum score of {threshold}%.")
        st.dataframe(matches_df, use_container_width=True, hide_index=True)

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
            st.dataframe(merged_df, use_container_width=True, hide_index=True)
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
