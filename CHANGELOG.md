# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
