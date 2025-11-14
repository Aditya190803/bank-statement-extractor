# Bank Statement – Customer Name Matcher

Streamlit application that extracts potential customer names from a bank statement PDF, flexibly maps uploaded CSV columns, supports employee-based filtering, and surfaces the matched customer details.

## Prerequisites

- Python 3.9+
- The following Python packages (install via `pip install -r requirements.txt`):
  - streamlit
  - pandas
  - pymupdf
  - rapidfuzz
  - openpyxl

## Running the app

```bash
cd "Bank-Statement Extractor"
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Usage workflow

1. Upload the bank statement PDF and a single customers CSV or Excel workbook. The file must contain a `long_name` column that holds full customer names and may include an optional `sub_broker` column for filtering.
2. Choose which column in the customers CSV represents the customer name (default preference is `long_name`). If your file includes `sub_broker`, you can filter by one or more sub_broker values.
3. Adjust the fuzzy-match score threshold or sub_broker filter from the sidebar if needed.
CSV example (first line header):
```
sub_broker,cl_code,long_name,mobile_pager,pan_gir_no,BOID,DP_Status,address_line,city,state,postal_code,email,account_type,risk_rating
BrokerA,CL001,Alice Johnson,999-100-0001,A1B2C3D4E5,BOID0001,Active,12 Garden Street,Mumbai,MH,400001,alice.johnson@example.com,Savings,Low
```

4. Click **Extract & Match** to generate matches. The app reports extraction counts, match totals, and the match rate.
5. Review the extracted names, matched customers with confidence scores, and merged customer details. Download either dataset as CSV when ready.

The bundled `sample_data/customer_details.csv` now includes 20 customer records across multiple sub_brokers with extended contact and account metadata, and a matching Excel version (`sample_data/customer_details.xlsx`) is provided for testing the Excel upload path.

All processing happens locally within the Streamlit session; no data leaves your machine.
