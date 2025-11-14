
# Bank Statement Extractor TODO

## Core Tasks
- [x] Sanitize all text before matching
- [x] Use single CSV source for data
- [x] Implement filtering from column details
- [x] Support Excel files as input
- [x] Extract text from all PDF pages (not just first page)

## CSV/Excel Structure
```
sub_broker | cl_code | long_name | mobile_pager | pan_gir_no | BOID | DP_Status
```

## Filtering & Matching
- [x] Filter by common `sub_broker` values (multi-select)
- [x] Match `long_name` with details extracted from PDF
- [x] List matched names as output
- [x] Add sample data with 20 records to verify filtering logic

