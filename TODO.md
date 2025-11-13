# Bank Statement Extractor - Internal Tool Todo (Prototype)

## Core Features (Required)

### Employee-Based Customer Filtering
- [x] **Add employee filter option**
  - [x] Display list of unique employees from customer CSV
  - [x] Add dropdown/select widget to filter by specific employee
  - [x] Filter both customer names and customer details by selected employee
  - [x] Show number of customers per employee in dropdown
  - [x] Option to select "All employees" for no filtering

- [x] **Implement flexible column selection UI**
  - [x] Auto-detect all columns in uploaded CSVs
  - [x] Display columns in dropdown for user selection
  - [x] Allow user to select which column contains:
    - [x] Customer Name (for customer names CSV)
    - [x] Customer Name (for customer details CSV)
    - [x] Employee ID or Name (for filtering option)
  - [x] Validate column existence before processing
  - [x] Remember user selections during session

- [x] **Update matching logic to use selected columns**
  - [x] Modify `match_names()` to use dynamically selected column
  - [x] Modify `merge_with_details()` to use dynamically selected column
  - [x] Add column selection to employee filtering logic

### Improved Name Extraction
- [ ] **Enhance PDF text extraction**
  - Better handling of special characters and formatting
  - Improve regex patterns for different name formats
  - Filter out common non-name text (dates, amounts, etc.)

### Better UI Organization
- [x] **Reorganize file upload section**
  - [x] Show column selection immediately after file upload
  - [x] Add preview of selected columns before processing
  - [x] Clear validation messages for column mapping

- [ ] **Improve results display**
  - [x] Show which employee's customers were used
  - [x] Display matching statistics (names extracted, matches found, match rate)
  - [ ] Better organization of matched vs unmatched results

## Data Quality & Validation

### CSV Validation
- [ ] **Pre-process CSV uploads**
  - Validate CSV can be read without errors
  - Show column list preview after upload
  - Detect and handle missing values in key columns
  - Show row count and data sample

### Name Extraction Improvements
- [ ] **Refine candidate name detection**
  - Filter common false positives (BANK, ACCOUNT, STATEMENT, etc.)
  - Better handling of abbreviated names
  - Improve detection of multi-word names
  - Handle names with prefixes (Mr., Ms., Dr., etc.)

## Reporting & Export

### Enhanced Export
- [ ] **Improve export functionality**
  - Include extraction parameters in export (threshold, employee filter used)
  - Add column headers clearly in exports
  - Include unmatched extracted names in optional export
  - Generate summary statistics in export

### Result Presentation
- [ ] **Better display of results**
  - Show extracted names sorted by frequency
  - Display match confidence with visual indicators
  - Highlight low-confidence matches for review
  - Show which customer details columns were merged in results

## Code Quality

### Refactoring & Organization
- [ ] **Improve code structure**
  - Add clear docstrings to all functions
  - Organize code with helper functions for CSV operations
  - Add comments explaining fuzzy matching threshold rationale
  - Separate column mapping logic into utility functions

### Error Handling
- [ ] **Better error messages**
  - Clear messages when columns are missing
  - Guide user to select correct columns on error
  - Specific error for empty extraction results
  - Helpful messages for CSV format issues

---

## Implementation Priority

### Phase 1 (Must Have)
1. Dynamic CSV column mapping with UI selection
2. Employee-based customer filtering
3. Update matching logic to use selected columns

### Phase 2 (Should Have)
4. CSV validation and preview
5. Better name extraction filtering
6. Enhanced export with metadata
7. Improved error messages

### Phase 3 (Nice to Have)
8. Result presentation improvements
9. Unmatched names export option
10. Code refactoring and documentation

---

## Notes
- **Logging, authentication, audit trails, and checksums will be handled in a separate main project**
- This is an internal prototype tool - focus on core functionality and usability
- CSV structure flexibility is critical since internal data sources vary
- Employee filtering is key for multi-employee organization audits
