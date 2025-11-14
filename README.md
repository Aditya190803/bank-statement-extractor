# Bank Statement – Customer Name Matcher

Streamlit application that extracts potential customer names from a bank statement PDF, flexibly maps uploaded CSV columns, supports employee-based filtering, and surfaces the matched customer details.

## Prerequisites

- Python 3.9+
- The following Python packages (install via `pip install -r requirements.txt`):
  - streamlit
  - pandas
  - pymupdf
  - rapidfuzz

## Running the app

```bash
cd "Bank-Statement Extractor"
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Usage workflow

1. Upload the bank statement PDF and a single Customers CSV. The CSV must contain a `long_name` column that holds full customer names and may include an optional `sub_broker` column for filtering.
2. Choose which column in the customers CSV represents the customer name (default preference is `long_name`). If your file includes `sub_broker`, you can filter by one or more sub_broker values.
3. Adjust the fuzzy-match score threshold or employee filter from the sidebar if needed.
4. Click **Extract & Match** to generate matches. The app reports extraction counts, match totals, and the match rate.
5. Review the extracted names, matched customers with confidence scores, and merged customer details. Download either dataset as CSV when ready.

All processing happens locally within the Streamlit session; no data leaves your machine.
