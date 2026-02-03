# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-02-04

### Added
- **Visual Zone Noise Detection**: Replaced fixed-line noise detection with coordinate-based targeting (top/bottom 10% of page height).
- **Fuzzy Noise Matching**: implemented robust header/footer deduplication using optimized fuzzy matching logic (tiered `difflib` similarity checks).
- **Caching Mechanism**: Added internal caching for the expensive noise discovery process within `PDFParser`, significantly improving performance for subsequent operations like segmentation.
- **Partial Processing**: Added `max_pages` support to `analyze_page_structure` and `detect_headings_from_content` for faster previews on large documents.

### Optimized
- **Extraction Speed**: Migrated noise candidate extraction to `get_text("blocks")`, reducing overhead on large PDFs compared to full dictionary extraction.

## [1.1.0] - 2026-02-02

### Changed
- **Refactoring**: Renamed `parser.py` to `pdf_parser.py` to avoid potential library conflicts.
- **Heuristic Logic**: Swapped priority in `get_text_by_topics()`: now prioritizes **TOC (Table of Contents)** over font-based detection for more accurate topic segmentation.
- **API**: Updated `/analyze-pdf` endpoint to accept an optional `lang` parameter (default: "en"). This allows switching between `config.json` and `config.tr.json`.

## [1.0.0] - 2026-02-01

### Added
- **FastAPI Backend**: Introduced `main.py` containing a FastAPI application with an `/analyze-pdf` endpoint for processing PDF uploads.
- **Hierarchy & Level Support**: Enhanced output to include a `level` field for headings. Supports both TOC-based levels and heuristic content-based levels (Level 1 for ALL CAPS, Level 2 for Title Case).
- **Visual TOC Extraction**: implemented `_extract_visual_toc()` to intelligently detect and extract Table of Contents entries from page content when metadata is missing.
- **Robustness Test Suite**: Added `test_robustness.py` to verify config fallback, case-insensitivity, and hierarchy assignments.

### Fixed
- **Configuration Crash**: Implemented a silent fallback mechanism. The parser now defaults to built-in settings with a warning if `config.json` is missing or invalid, preventing crashes.
- **Traceability & Normalization**: Fixed issues with Turkish character normalization (e.g., 'İ' vs 'i') in blacklist filtering by ensuring consistent case-insensitive comparison using Unicode normalization.

### Security
- **CORS Middleware**: Applied CORS configuration in the FastAPI app to safely handle cross-origin requests, enabling integration with mobile clients (Flutter).
- **Resource Management**: Implemented automatic temporary file cleanup for uploaded PDFs, ensuring no sensitive files remain on the server after processing.
