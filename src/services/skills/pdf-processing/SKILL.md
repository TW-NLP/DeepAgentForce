---
name: pdf-processing
description: Extract text and tables from PDF files, merge/split documents, create PDFs, fill forms. Use when working with PDF documents.
version: 1.0.0
author: PDF Skills Team
tags:
  - pdf
  - document-processing
  - data-extraction
---

# PDF Processing Skill

## When to use this skill

Use this skill when the user mentions:
- Reading or extracting content from PDFs
- Working with tables in PDF documents
- Merging, splitting, or reorganizing PDFs
- Creating new PDF documents
- Filling out PDF forms
- Processing scanned PDFs (OCR)

## Progressive Disclosure Strategy

### Level 1: Quick Start
When user asks about PDF:
1. Ask ONLY for the file path
2. Choose the appropriate script based on their need
3. Execute the script with minimal parameters
4. Show results

### Level 2: Core Operations
Only reveal these options after initial execution or if user asks.

### Level 3: Advanced Features
Only introduce when user explicitly needs them.

## Available Scripts

All scripts are in the `src/services/skills/pdf-processing/scripts` directory. Execute them using bash or python.

### 1. Extract Text: `extract_text.py`

**When to use**: User wants to read PDF content, extract text.

**Usage**:
```bash
python src/services/skills/pdf-processing/scripts/extract_text.py <pdf_path> [--pages PAGES] [--output OUTPUT]
```

**Parameters**:
- `pdf_path`: (required) Path to PDF file
- `--pages`: (optional) Page range, e.g., "1-5" or "1,3,5". Default: all pages
- `--output`: (optional) Output file path. Default: stdout

**Example**:
```bash
# Extract all text
python src/services/skills/pdf-processing/scripts/extract_text.py report.pdf

# Extract specific pages
python src/services/skills/pdf-processing/scripts/extract_text.py report.pdf --pages "1-5"

# Save to file
python src/services/skills/pdf-processing/scripts/extract_text.py report.pdf --output extracted.txt
```

**Progressive disclosure**: Start with just the pdf_path, add options only if user needs them.

### 2. Extract Tables: `extract_tables.py`

**When to use**: User wants to extract tables, data, or spreadsheet content from PDF.

**Usage**:
```bash
python src/services/skills/pdf-processing/scripts/extract_tables.py <pdf_path> [--format FORMAT] [--output OUTPUT]
```

**Parameters**:
- `pdf_path`: (required) Path to PDF file
- `--format`: (optional) Output format: excel, csv, json. Default: excel
- `--output`: (optional) Output file path. Default: auto-generated

**Example**:
```bash
# Extract to Excel (default)
python src/services/skills/pdf-processing/scripts/extract_tables.py financial_report.pdf

# Extract to CSV
python src/services/skills/pdf-processing/scripts/extract_tables.py data.pdf --format csv

# Custom output path
python src/services/skills/pdf-processing/scripts/extract_tables.py data.pdf --output results/tables.xlsx
```

**Progressive disclosure**: Default to Excel format unless user specifies otherwise.

### 3. Merge PDFs: `merge_pdfs.py`

**When to use**: User wants to combine multiple PDFs into one.

**Usage**:
```bash
python src/services/skills/pdf-processing/scripts/merge_pdfs.py <output_path> <pdf1> <pdf2> [pdf3 ...]
```

**Parameters**:
- `output_path`: (required) Path for merged PDF
- `pdf1, pdf2, ...`: (required) PDFs to merge (in order)

**Example**:
```bash
# Merge PDFs
python src/services/skills/pdf-processing/scripts/merge_pdfs.py merged.pdf doc1.pdf doc2.pdf doc3.pdf

# Merge all PDFs in directory
python src/services/skills/pdf-processing/scripts/merge_pdfs.py combined.pdf *.pdf
```

### 4. Split PDF: `split_pdf.py`

**When to use**: User wants to split PDF into separate files.

**Usage**:
```bash
python src/services/skills/pdf-processing/scripts/split_pdf.py <pdf_path> [--mode MODE] [--output-dir OUTPUT_DIR]
```

**Parameters**:
- `pdf_path`: (required) Path to PDF file
- `--mode`: (optional) Split mode: pages (one per file), range. Default: pages
- `--output-dir`: (optional) Output directory. Default: ./split_output

**Example**:
```bash
# Split into individual pages
python src/services/skills/pdf-processing/scripts/split_pdf.py document.pdf

# Split into custom directory
python src/services/skills/pdf-processing/scripts/split_pdf.py document.pdf --output-dir pages/
```

### 5. Create PDF: `create_pdf.py`

**When to use**: User wants to create a PDF from text or content.

**Usage**:
```bash
python src/services/skills/pdf-processing/scripts/create_pdf.py <output_path> [--title TITLE] [--content CONTENT] [--input INPUT_FILE]
```

**Parameters**:
- `output_path`: (required) Path for new PDF
- `--title`: (optional) Document title
- `--content`: (optional) Text content (for short content)
- `--input`: (optional) Text file to read content from

**Example**:
```bash
# From text content
python src/services/skills/pdf-processing/scripts/create_pdf.py report.pdf --title "Monthly Report" --content "Report content here..."

# From text file
python src/services/skills/pdf-processing/scripts/create_pdf.py report.pdf --title "Report" --input content.txt
```

### 6. Fill PDF Form: `fill_form.py`

**When to use**: User needs to fill out a PDF form programmatically.

**Usage**:
```bash
python src/services/skills/pdf-processing/scripts/fill_form.py <form_path> <data_file> <output_path>
```

**Parameters**:
- `form_path`: (required) Path to PDF form
- `data_file`: (required) JSON file with field data
- `output_path`: (required) Path for filled PDF

**Example**:
```bash
# Fill form with data from JSON
python src/services/skills/pdf-processing/scripts/fill_form.py application.pdf data.json filled_application.pdf
```

**See**: `./references/form-filling.md` for data format details.

### 7. OCR PDF: `ocr_pdf.py`

**When to use**: User has scanned/image-based PDF and needs searchable text.

**Usage**:
```bash
python src/services/skills/pdf-processing/scripts/ocr_pdf.py <pdf_path> [--lang LANG] [--output OUTPUT]
```

**Parameters**:
- `pdf_path`: (required) Path to scanned PDF
- `--lang`: (optional) OCR language code. Default: eng
- `--output`: (optional) Output PDF path. Default: <original>_ocr.pdf

**Example**:
```bash
# OCR English document
python src/services/skills/pdf-processing/scripts/ocr_pdf.py scanned.pdf

# OCR with different language
python src/services/skills/pdf-processing/scripts/ocr_pdf.py document.pdf --lang chi_sim
```

**Note**: Requires tesseract-ocr installed on system.

## Interaction Guidelines

### ✅ Good Interaction Pattern

```
User: "I need to read a PDF"
Agent: "I'll help you extract the text. What's the PDF file path?"
User: "report.pdf"
Agent: [Executes] python src/services/skills/pdf-processing/scripts/extract_text.py report.pdf
Agent: "✅ Extracted text from 5 pages. Here's a preview: [shows first 500 chars]
       Would you like to see the full text or extract tables?"
```

### ❌ Poor Interaction Pattern

```
User: "I need to read a PDF"
Agent: "I can extract text, tables, images, metadata. For text I can preserve layout or not, 
       use different encodings, extract specific pages using ranges or lists, 
       output to txt/docx/html. For tables I need to know format (Excel/CSV/JSON), 
       encoding, empty cell handling... Please specify all parameters."
```

### Decision Tree for PDF Tasks

```
User mentions PDF
  ├─ "read", "extract text", "content" → execute extract_text.py
  ├─ "table", "data", "spreadsheet" → execute extract_tables.py
  ├─ "merge", "combine", "join" → execute merge_pdfs.py
  ├─ "split", "separate", "break" → execute split_pdf.py
  ├─ "create", "generate", "make" → execute create_pdf.py
  ├─ "form", "fill out", "application" → execute fill_form.py
  └─ "scanned", "image", "OCR" → execute ocr_pdf.py
```

## Error Handling

If a script fails:
1. Show the error message clearly
2. Suggest common fixes (file not found, missing dependencies)
3. Don't repeat the same command - try alternatives or ask for clarification

**Common errors**:
- `FileNotFoundError`: Check if file path is correct, suggest listing directory
- `ImportError` (pdfplumber, pypdf, etc.): Install missing package
- `No tables found`: Suggest OCR if PDF might be scanned
- `Permission denied`: Check file permissions

## Dependencies

Scripts require these Python packages:
```bash
pip install pdfplumber pypdf reportlab pandas openpyxl pytesseract pdf2image
```

System dependencies:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils

# macOS
brew install tesseract poppler
```

## Reference Documents

- `./references/form-filling.md`: Detailed guide for PDF form data format
- `./references/ocr-guide.md`: OCR best practices and language codes
- `./references/troubleshooting.md`: Common issues and solutions

## Examples

See `./examples/` directory for complete workflow examples:
- `example_extract_report.sh`: Extract text and tables from report
- `example_merge_quarterly.sh`: Merge quarterly PDFs
- `example_batch_process.sh`: Process multiple PDFs

## Key Principles

1. **Scripts are self-contained**: Each script handles one task completely
2. **Minimal parameters**: Use smart defaults, accept overrides
3. **Clear output**: Scripts print success/error messages with emojis
4. **Exit codes**: 0 for success, non-zero for errors
5. **Help text**: All scripts support `--help` flag

## Tips for Agent

- **Start simple**: Use minimal parameters first
- **Show progress**: For long operations, show that work is happening
- **Be helpful**: Suggest next logical steps after each operation
- **Learn preferences**: If user always wants CSV, remember that
- **Handle errors gracefully**: Don't just show stack traces, explain the problem

## Skill Maintenance

This skill follows the Agent Skills standard:
- **name** and **description** in frontmatter enable discovery
- **SKILL.md** provides full instructions for activation
- **scripts/** contain executable code for execution
- **Progressive disclosure** reduces cognitive load

Version: 1.0.0
Last updated: 2025-02-02