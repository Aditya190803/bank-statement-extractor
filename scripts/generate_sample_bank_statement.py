"""Generate a multi-page sample bank statement PDF with structured tables."""
from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path
from typing import List

import fitz  # PyMuPDF

PAGE_SIZE = (595, 842)  # A4 in points
MARGIN_X = 40
MARGIN_TOP = 60
MARGIN_BOTTOM = 40
ROW_HEIGHT = 20
ROWS_PER_PAGE = 26  # allows room for header/footer
TOTAL_PAGES = 33
TOTAL_ROWS = ROWS_PER_PAGE * TOTAL_PAGES

STATEMENT_TITLE = "Fidelity Federal Banking Group"
ACCOUNT_HOLDER = "Primary Account Holder: Alice Johnson"
ACCOUNT_NUMBER = "Account Number: 1112-2233-4455-66"
REPORT_PERIOD = "Statement Period: January 1 – March 31, 2024"

COLUMNS = (
    ("Date", MARGIN_X),
    ("Reference", MARGIN_X + 80),
    ("Description", MARGIN_X + 160),
    ("Debit", MARGIN_X + 360),
    ("Credit", MARGIN_X + 430),
    ("Balance", MARGIN_X + 500),
)

COUNTERPARTIES = [
    "Lakeside Market",
    "Green Meadow Pharmacy",
    "Summit Ridge Utilities",
    "Riverside Insurance Group",
    "Northern Lights Travel Co.",
    "Downtown Fitness Club",
    "Bright Future Education",
    "Harmony Health Clinic",
    "Silverline Digital",
    "Urban Garden Center",
]

BASE_DATE = date(2024, 1, 1)


def _load_customer_names() -> List[str]:
    """Load all customer names from the sample CSV for PDF generation."""
    csv_path = Path(__file__).resolve().parent.parent / "sample_data" / "customer_details.csv"
    if not csv_path.exists():
        return []

    names: List[str] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = (row.get("long_name") or "").strip()
            if name:
                names.append(name)
    return names


def build_transactions(customer_names: List[str]) -> list[dict[str, str]]:
    """Create a list of synthetic bank transactions."""
    transactions: list[dict[str, str]] = []
    running_balance = 12500.00

    if not customer_names:
        # Fallback to a small static set when the CSV is missing or empty.
        customer_names = [
            "Alice Johnson",
            "Bob Smith",
            "Carlos Diaz",
            "Danielle Young",
            "Elaine O'Neil",
        ]

    for index in range(TOTAL_ROWS):
        tx_date = BASE_DATE + timedelta(days=index % 90)
        customer = customer_names[index % len(customer_names)]
        counterparty = random.choice(COUNTERPARTIES)
        reference = f"TX-{tx_date.strftime('%y%m%d')}-{index:05d}"

        if index % 3 == 0:
            debit = round(random.uniform(45, 625), 2)
            credit = 0.0
            descriptor = f"ACH Payment to {customer}"
        elif index % 5 == 0:
            debit = 0.0
            credit = round(random.uniform(250, 1500), 2)
            descriptor = f"Wire transfer from {counterparty}"
        else:
            debit = round(random.uniform(12, 240), 2)
            credit = 0.0
            descriptor = f"Card purchase at {counterparty}"

        running_balance += credit - debit

        transactions.append(
            {
                "Date": tx_date.strftime("%d %b %Y"),
                "Reference": reference,
                "Description": descriptor,
                "Debit": f"{debit:,.2f}" if debit else "-",
                "Credit": f"{credit:,.2f}" if credit else "-",
                "Balance": f"{running_balance:,.2f}",
            }
        )

    return transactions


def add_page_header(page: fitz.Page, page_number: int) -> float:
    """Draw the page heading and return the starting Y coordinate for table rows."""
    page.insert_text((MARGIN_X, MARGIN_TOP - 30), STATEMENT_TITLE, fontsize=18, fontname="helv", fill=(0, 0, 0))
    page.insert_text((MARGIN_X, MARGIN_TOP - 10), ACCOUNT_HOLDER, fontsize=11, fontname="helv")
    page.insert_text((MARGIN_X, MARGIN_TOP + 6), ACCOUNT_NUMBER, fontsize=11, fontname="helv")
    page.insert_text((MARGIN_X, MARGIN_TOP + 22), REPORT_PERIOD, fontsize=11, fontname="helv")

    page.insert_text((PAGE_SIZE[0] - MARGIN_X - 90, MARGIN_TOP - 30), f"Page {page_number}/{TOTAL_PAGES}", fontsize=10, fontname="helv")

    table_top = MARGIN_TOP + 50
    # Column headers
    for title, x_pos in COLUMNS:
        page.insert_text((x_pos, table_top), title.upper(), fontsize=11, fontname="helv")

    # Horizontal line below headers
    page.draw_line((MARGIN_X, table_top + 14), (PAGE_SIZE[0] - MARGIN_X, table_top + 14))
    return table_top + 24


def add_page_footer(page: fitz.Page) -> None:
    """Draw the footer annotation."""
    footer_y = PAGE_SIZE[1] - MARGIN_BOTTOM + 10
    page.draw_line((MARGIN_X, footer_y - 12), (PAGE_SIZE[0] - MARGIN_X, footer_y - 12))
    page.insert_text(
        (MARGIN_X, footer_y),
        "This statement is for illustrative purposes only and contains fabricated data.",
        fontsize=9,
        fontname="helv",
    )


def write_transactions(path: Path) -> None:
    customer_names = _load_customer_names()
    transactions = build_transactions(customer_names)
    doc = fitz.open()

    row_index = 0
    for page_number in range(1, TOTAL_PAGES + 1):
        page = doc.new_page(width=PAGE_SIZE[0], height=PAGE_SIZE[1])
        y_cursor = add_page_header(page, page_number)

        for _ in range(ROWS_PER_PAGE):
            if row_index >= len(transactions):
                break

            tx = transactions[row_index]
            for (column, x_pos) in COLUMNS:
                page.insert_text((x_pos, y_cursor), tx[column], fontsize=10, fontname="helv")
            y_cursor += ROW_HEIGHT
            row_index += 1

        add_page_footer(page)

    doc.save(path)
    doc.close()


if __name__ == "__main__":
    output_path = Path(__file__).resolve().parent.parent / "sample_data" / "bank_statement.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_transactions(output_path)
    print(f"Generated sample statement at: {output_path}")
