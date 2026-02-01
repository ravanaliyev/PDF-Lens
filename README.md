# 🔍 PDF-Lens

**PDF-Lens** is a high-performance Python engine designed to deconstruct unstructured PDF documents, extract hierarchical topic structures, and serve this data through a modern API for mobile (Flutter) and web applications.

---

## ✨ Key Features

- **Hierarchical Analysis**: Segments text into meaningful topics using the `get_text_by_topics()` engine rather than just simple page blocks.
- **Smart-Level Support**: Automatically classifies headings into Level 1 (Main Topic) and Level 2 (Sub-Topic) based on structural heuristics.
- **Surgical Text Cleaning**: Normalizes line breaks, fixes hyphenation, and removes control characters to provide "clean" data.
- **Hybrid Heading Detection**:
    - Utilizes native **PDF TOC (Table of Contents)** metadata form when available.
    - Features a **fallback Visual TOC scanner** and **font-size analyzer** to detect structure in documents without metadata.
- **Table & Image Intelligence**: Identifies tables within pages, converts them to Markdown format, and analyzes visual density (image counts) per page.

---

## 🛠 Technical Architecture

The project is architected into three primary layers:

| Component | Type | Responsibility |
|-----------|------|----------------|
| **`parser.py`** | **Engine** | The core logic handler built on `PyMuPDF` (`fitz`). It performs low-level data extraction and heuristic analysis. |
| **`main.py`** | **API** | A `FastAPI` powered server layer that acts as the bridge between the Python engine and external clients (like Flutter). |
| **`cli.py`** | **Dev Tools** | A command-line interface for rapid testing and detailed analysis reporting in the terminal. |

---

## 🚀 Quick Start

### 1. Installation
Ensure you are in your virtual environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the API Server
Start the FastAPI server to listen for incoming requests:

```bash
python main.py
# Server running at http://127.0.0.1:8000
```

### 3. CLI Testing
To get a quick structural report of a PDF directly in your terminal:

```bash
python cli.py [optional_path_to_pdf]
```

---

## 📦 Project Structure

| File | Description |
|------|-------------|
| `parser.py` | Core PDF processing and heuristic logic. |
| `main.py` | REST API (`FastAPI`) implementation and entry point. |
| `cli.py` | Command-line interface for debugging and local reports. |
| `config.json` | Centralized configuration for heading detection and blacklists. |
| `requirements.txt` | Python dependency list. |
| `CHANGELOG.md` | Detailed version history and technical milestones. |