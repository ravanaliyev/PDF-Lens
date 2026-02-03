import fitz
import os
import re
import json
import unicodedata
from datetime import datetime
import difflib
from typing import List, Dict, Optional, Any, Set


class PDFParserError(Exception):
    """Custom exception class for generic PDFParser errors."""


class PDFParser:
    """High-performance PDF parser using PyMuPDF (fitz).

    Config-driven system: loads settings from config.json.

    Args:
        path: File path.
        config_path: Config file path (default: config.json).

    Example:
        parser = PDFParser("/path/to/file.pdf")
        parser = PDFParser("/path/to/file.pdf", config_path="/path/to/config.json")
    """

    def __init__(self, path: str, config_path: str = "config.json"):
        self.path = path
        self.doc: Optional[fitz.Document] = None
        self.config: Dict[str, Any] = {}
        self.blacklist: Set[str] = set()
        self._noise_cache: Optional[Set[str]] = None
        
        # Load config file
        self._load_config(config_path)
        
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            if not os.access(path, os.R_OK):
                raise PermissionError(f"File not readable (permission denied): {path}")
            self.doc = fitz.open(path)
        except FileNotFoundError:
            raise
        except PermissionError:
            raise
        except RuntimeError as e:
            # PyMuPDF raises RuntimeError for encrypted/corrupted files
            raise PDFParserError(f"PDF could not be opened (may be encrypted/corrupted): {e}")
        except Exception as e:  # pragma: no cover - unexpected errors
            raise PDFParserError(f"Unexpected error opening PDF: {e}")

    def close(self) -> None:
        """Closes the PDF document. Can be called when done to prevent resource leaks."""
        if self.doc is not None:
            try:
                self.doc.close()
            except Exception:
                pass
            self.doc = None

    def __enter__(self) -> "PDFParser":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _load_config(self, config_path: str) -> None:
        """Load config file and set up blacklist.
        
        Robust fallback: if config.json is missing or unreadable, use default settings.
        Blacklist is case-insensitive: stored as lowercase.
        """
        # Default config values (English defaults)
        default_config = {
            "language": "en",
            "blacklist": [],
            "heading_heuristics": {
                "min_chars": 2,
                "max_chars": 20,
                "max_words": 2,
                "min_description_chars": 50,
                "filter_bullet_chars": ["•", "-", "*"]
            },
            "font_detection": {
                "detect_by_size_increase": True,
                "require_numeric_content": True,
                "size_increase_threshold": 1.2,
            },
            "noise_filter": {
                "min_pages": 3,
                "top_n": 3,
                "bottom_n": 3,
            },
            "caption_prefixes": ["figure", "table", "chart"],
            "toc_keywords": ["contents", "index", "table of contents"],
            "copyright_keywords": [
                "copyright", "all rights reserved", "isbn", "legal notice", 
                "authorized adaptation", "license", "licensing agency", 
                "printed and bound"
            ]
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # Merge: merge loaded config with defaults
                    self.config = {**default_config, **loaded_config}
                    # Merge nested dicts (heading_heuristics, font_detection)
                    if "heading_heuristics" in loaded_config:
                        self.config["heading_heuristics"] = {**default_config["heading_heuristics"], **loaded_config["heading_heuristics"]}
                    if "font_detection" in loaded_config:
                        self.config["font_detection"] = {**default_config["font_detection"], **loaded_config["font_detection"]}
                    if "noise_filter" in loaded_config:
                        self.config["noise_filter"] = {**default_config["noise_filter"], **loaded_config["noise_filter"]}
                    # caption_prefixes, toc_keywords, copyright_keywords come via top-level merge
                    
                    # Blacklist: convert to lowercase case-insensitive
                    blacklist_raw = self.config.get("blacklist", [])
                    self.blacklist = set(word.lower() for word in blacklist_raw)
            else:
                # Config file not found: use defaults
                print(f"⚠️  Warning: '{config_path}' not found. Using default settings.")
                self.config = default_config
                self.blacklist = set()
        except json.JSONDecodeError as e:
            # JSON parsing error: use defaults
            print(f"⚠️  Warning: '{config_path}' invalid JSON ({e}). Using default settings.")
            self.config = default_config
            self.blacklist = set()
        except Exception as e:
            # Other errors (permissions etc): use defaults
            print(f"⚠️  Warning: Error reading config file ({e}). Using default settings.")
            self.config = default_config
            self.blacklist = set()

    def clean_text(self, text: str) -> str:
        """Cleaning pipeline.

        - Normalizes line endings.
        - Fixes hyphenation at line breaks.
        - Removes control characters.
        - Collapses multiple spaces and tabs.
        - Limits consecutive newlines to two.

        Args:
            text: Raw text.

        Returns:
            Cleaned text.
        """
        if not text:
            return ""
        # Normalize line endings
        txt = text.replace('\r\n', '\n').replace('\r', '\n')
        # Remove control chars except newline and tab
        txt = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", "", txt)
        # Fix hyphenation at line breaks: "exam-\nple" -> "example"
        txt = re.sub(r"(\w)-\n(\w)", r"\1\2", txt)
        # Collapse multiple spaces/tabs on same line
        txt = re.sub(r"[ \t]+", " ", txt)
        # Limit consecutive newlines to maximum two
        txt = re.sub(r"\n{3,}", "\n\n", txt)
        return txt.strip()

    def _normalize_line(self, s: str) -> str:
        """Normalize a line for robust equality checks (collapse whitespace, lower).

        Used for detecting repeating headers/footers across pages.
        """
        if not s:
            return ""
        return re.sub(r"\s+", " ", s).strip().lower()

    def detect_repeating_headers_footers(
        self, min_pages: Optional[int] = None, top_n: Optional[int] = None, bottom_n: Optional[int] = None
    ) -> set:
        """Detect repeating header/footer lines that appear identically on many pages.

        Scans the first `top_n` and last `bottom_n` non-empty lines of every page,
        normalizes them and returns the set of normalized lines that occur on at
        least `min_pages` pages. These are only removed when they appear in
        header/footer position (see get_pages), so in-content occurrences are kept.

        Args:
            min_pages: Repeat on at least this many pages (default: config noise_filter.min_pages).
            top_n: Top N lines of each page to consider header (default: config).
            bottom_n: Bottom N lines of each page to consider footer (default: config).

        Returns:
            set: normalized lines to treat as noise (only when in header/footer position).
        """
        if not self.doc:
            raise PDFParserError("PDF file is not open.")

        if self._noise_cache is not None and min_pages is None and top_n is None and bottom_n is None:
            return self._noise_cache

        nf = self.config.get("noise_filter", {})
        min_pages = min_pages if min_pages is not None else nf.get("min_pages", 3)
        top_n = top_n if top_n is not None else nf.get("top_n", 3)
        bottom_n = bottom_n if bottom_n is not None else nf.get("bottom_n", 3)

        from collections import Counter

        top_counts: Any = Counter()
        bottom_counts: Any = Counter()

        for i in range(self.doc.page_count):
            try:
                page = self.doc.load_page(i)
                height = page.rect.height
                # "blocks" is much faster than "dict"
                blocks = page.get_text("blocks") or []
            except Exception:
                continue

            for b in blocks:
                if b[6] != 0:  # block_type 0 is text
                    continue
                
                y0, y1 = b[1], b[3]
                txt = b[4].strip()
                if not txt:
                    continue
                
                # Split blocks into lines for granular header matching
                for line_text in txt.splitlines():
                    ln = line_text.strip()
                    if not ln: continue
                    
                    norm = self._normalize_line(ln)
                    if not norm:
                        continue

                    # Top zone: first 10% (approximate block bbox for lines inside)
                    if y1 < height * 0.10:
                        top_counts[norm] += 1
                    # Bottom zone: last 10%
                    elif y0 > height * 0.90:
                        bottom_counts[norm] += 1

        # Collect all candidates with their counts
        top_candidates = list(top_counts.items())
        bottom_candidates = list(bottom_counts.items())

        repeated = set()

        def gather_fuzzy_matches(candidates, threshold=0.9):
            # candidates: list of (line, count)
            # We want to group similarities.
            # Simple approach: A greedy clustering.
            # Sort by frequency (descending) to pick most common forms as cluster centers.
            sorted_c = sorted(candidates, key=lambda x: x[1], reverse=True)
            
            processed = set()
            
            for i, (line1, count1) in enumerate(sorted_c):
                if line1 in processed:
                    continue
                
                # Start a cluster
                cluster = {line1}
                total_count = count1
                processed.add(line1)
                
                len1 = len(line1)
                # Optimization: create Matcher once for line1
                sm = difflib.SequenceMatcher(None, line1, "")
                
                for j in range(i + 1, len(sorted_c)):
                    line2, count2 = sorted_c[j]
                    if line2 in processed:
                        continue
                    
                    # Pre-filter: length must be similar
                    len2 = len(line2)
                    if abs(len1 - len2) > max(len1, len2) * 0.2:
                        continue
                        
                    # Fast similarity check
                    sm.set_seq2(line2)
                    if sm.real_quick_ratio() < threshold:
                        continue
                    if sm.quick_ratio() < threshold:
                        continue
                        
                    # Expensive exact check
                    if sm.ratio() >= threshold:
                        cluster.add(line2)
                        total_count += count2
                        processed.add(line2)
                
                # If total count of cluster >= min_pages, mark ALL variations as noise
                if total_count >= min_pages:
                    repeated.update(cluster)

        gather_fuzzy_matches(top_candidates)
        gather_fuzzy_matches(bottom_candidates)
        
        # Cache the result if using defaults
        if min_pages is None and top_n is None and bottom_n is None:
            self._noise_cache = repeated

        return repeated

    def get_metadata(self) -> Dict[str, Optional[str]]:
        """Returns PDF metadata.

        Keys: `title`, `author`, `producer`, `creation_date`, `mod_date`.

        Returns:
            Dict: Metadata fields (None if not found).
        """
        if not self.doc:
            raise PDFParserError("PDF file is not open.")
        md = self.doc.metadata or {}
        # PyMuPDF date format is usually 'D:YYYYMMDDHHmmSS'
        def _parse_date(d: Optional[str]) -> Optional[str]:
            if not d:
                return None
            try:
                # Strip leading 'D:' if present
                d2 = d[2:] if d.startswith('D:') else d
                # try parsing common formats
                dt = datetime.strptime(d2[:14], "%Y%m%d%H%M%S")
                return dt.isoformat()
            except Exception:
                return d

        return {
            "title": md.get("title"),
            "author": md.get("author"),
            "producer": md.get("producer"),
            "creation_date": _parse_date(md.get("creationDate") or md.get("creatordate")),
            "mod_date": _parse_date(md.get("modDate") or md.get("moddate")),
        }

    def get_page_count(self) -> int:
        """Returns the number of pages in the PDF."""
        if not self.doc:
            return 0
        return self.doc.page_count

    def get_pages(self) -> List[Dict[str, Optional[str]]]:
        """Returns page-based text as a list of dictionaries.

        Keys per dict: `page_number`, `raw_text`, `clean_text`, `is_copyright`.
        """
        if not self.doc:
            raise PDFParserError("PDF file is not open.")
        pages: List[Dict[str, Optional[str]]] = []

        # Noise: remove only if in first/last N lines AND in noise_set (content occurrences kept)
        try:
            noise_set = self.detect_repeating_headers_footers()
        except Exception:
            noise_set = set()
        nf = self.config.get("noise_filter", {})
        # Increase window to be sure we catch everything in the 10% zone
        top_n = max(nf.get("top_n", 3), 10)
        bottom_n = max(nf.get("bottom_n", 3), 10)

        for i in range(self.doc.page_count):
            try:
                page = self.doc.load_page(i)
                raw = page.get_text("text") or ""
            except Exception as e:  # pragma: no cover - page read error
                raw = f"(Page could not be read: {e})"

            # Extract tables and convert to Markdown to preserve structure
            try:
                tables = self.extract_tables_from_page(page)
                table_markdowns = [self.table_to_markdown(t) for t in tables]
            except Exception:
                table_markdowns = []

            # Noise: remove lines only if they are in header/footer POSITION and in noise_set
            if noise_set and raw:
                lines = raw.splitlines()
                total_non_empty = sum(1 for ln in lines if ln.strip())
                non_empty_idx = 0
                filtered_lines = []
                for ln in lines:
                    if not ln.strip():
                        filtered_lines.append(ln)
                        continue
                    non_empty_idx += 1
                    in_header = non_empty_idx <= top_n
                    in_footer = non_empty_idx > total_non_empty - bottom_n
                    if (in_header or in_footer) and self._normalize_line(ln) in noise_set:
                        continue
                    filtered_lines.append(ln)
                filtered_raw = "\n".join(filtered_lines)
            else:
                filtered_raw = raw

            clean = self.clean_text(filtered_raw)
            if table_markdowns:
                clean = clean + "\n\n" + "\n\n".join(table_markdowns)

            # Copyright/legal pages: strip text (clean_text empty)
            is_copyright = self._is_copyright_legal_page(raw)
            if is_copyright:
                clean = ""

            pages.append({
                "page_number": i + 1,
                "raw_text": raw,
                "clean_text": clean,
                "is_copyright": is_copyright,
            })
        return pages

    def extract_tables_from_page(self, page) -> List[List[List[str]]]:
        """Extracts tables from a page.

        First attempts to use PyMuPDF's `find_tables()` (if available) API.
        If not found or fails, estimates tables using a simple heuristic based on text blocks.

        Returns:
            List of tables; each table is a list of rows, rows are lists of cell text.
        """
        tables: List[List[List[str]]] = []

        # 1) Use PyMuPDF's structured table API if available
        try:
            if hasattr(page, "find_tables"):
                found = page.find_tables() or []
                for f in found:
                    try:
                        # Handle different PyMuPDF versions flexibly
                        # If object is dict-like check 'cells' or 'rows' keys
                        if isinstance(f, dict) and "rows" in f:
                            rows = f.get("rows") or []
                            tables.append([[str(cell) for cell in row] for row in rows])
                            continue

                        # If we can get cells from table object
                        if hasattr(f, "cells"):
                            cells = f.cells
                            # cells provider is list of (r,c,text)
                            rows_map = {}
                            max_col = 0
                            for c in cells:
                                # destructure if tuple-like
                                try:
                                    r, col, txt = c
                                except Exception:
                                    # attempt dict-like
                                    r = c.get("row") if isinstance(c, dict) else None
                                    col = c.get("col") if isinstance(c, dict) else None
                                    txt = c.get("text") if isinstance(c, dict) else str(c)
                                if r is None or col is None:
                                    continue
                                rows_map.setdefault(r, {})[col] = str(txt)
                                if col > max_col:
                                    max_col = col
                            rows = []
                            for r_idx in sorted(rows_map.keys()):
                                row = [rows_map[r_idx].get(c_idx, "") for c_idx in range(max_col + 1)]
                                rows.append(row)
                            tables.append(rows)
                            continue

                        # Another API: obje.extract() -> rows
                        if hasattr(f, "extract"):
                            try:
                                rows = f.extract() or []
                                if isinstance(rows, list) and rows:
                                    tables.append([[str(cell) for cell in row] for row in rows])
                                    continue
                            except Exception:
                                pass
                    except Exception:
                        continue
        except Exception:
            # If find_tables call fails, ignore safely
            pass

        # 2) Fallback: text block based simple table extraction (previous heuristic)
        try:
            d = page.get_text("dict") or {}
            blocks = d.get("blocks", [])
        except Exception:
            try:
                blocks = page.get_text("blocks") or []
            except Exception:
                blocks = []

        for b in blocks:
            text_block = ""
            if isinstance(b, dict):
                text_block = b.get("text", "") or ""
            elif isinstance(b, (list, tuple)) and len(b) >= 5:
                text_block = b[4] or ""
            else:
                continue

            lines = [ln for ln in text_block.splitlines() if ln.strip()]
            if len(lines) < 3:
                continue

            cols_per_line = []
            rows_tokens: List[List[str]] = []
            for ln in lines:
                tokens = [t for t in re.split(r"\s{2,}|\t|\|", ln.strip()) if t.strip()]
                cols_per_line.append(len(tokens))
                rows_tokens.append(tokens)

            if not cols_per_line:
                continue

            avg_cols = sum(cols_per_line) / len(cols_per_line)
            pct_multi = sum(1 for c in cols_per_line if c >= 2) / len(cols_per_line)

            if avg_cols >= 2 and pct_multi >= 0.6:
                # Normalize rows to same width
                max_cols = max(len(r) for r in rows_tokens)
                norm_rows = [r + [""] * (max_cols - len(r)) for r in rows_tokens]
                tables.append(norm_rows)

        return tables

    def table_to_markdown(self, table: List[List[str]]) -> str:
        """Converts given 2D table data to Markdown format.

        Rows are normalized to the header column count (missing cells empty, excess truncated).

        Args:
            table: List of rows, each row is a list of cell text.

        Returns:
            Markdown string representation.
        """
        if not table:
            return ""
        header = table[0]
        cols = len(header)
        # Escape pipes and trim
        def esc(cell: str) -> str:
            return cell.replace("|", "\uFF5C").strip()

        def norm_row(row: List[str]) -> List[str]:
            """Norm row to cols length: empty if missing, cut if too long."""
            r = [esc(c) for c in row[:cols]]
            r.extend([""] * (cols - len(r)))
            return r[:cols]

        header_line = "| " + " | ".join(esc(c) for c in header) + " |"
        sep_line = "| " + " | ".join("---" for _ in range(cols)) + " |"
        body_lines = []
        for row in table[1:]:
            norm = norm_row(row)
            body_lines.append("| " + " | ".join(norm) + " |")

        md = "\n".join([header_line, sep_line] + body_lines)
        return md

    def _is_copyright_legal_page(self, text: str) -> bool:
        """Checks if the page is a copyright or legal page.
        
        Detects pages containing copyright symbols, ISBN, legal notices etc.
        
        Args:
            text: Page text
            
        Returns:
            True if copyright/legal page, False otherwise
        """
        if not text:
            return False
        
        text_lower = text.lower()
        
        # Keywords for copyright and legal pages (from config)
        copyright_keywords = self.config.get("copyright_keywords", [
            'copyright', 'all rights reserved', 'isbn', 'legal notice', 'license'
        ])
        
        # Must contain at least 2 keywords
        keyword_count = 0
        for keyword in copyright_keywords:
            if keyword.lower() in text_lower:
                keyword_count += 1
        
        # Diagnosis: 2+ keywords OR (ISBN + 1 keyword)
        has_multiple_keywords = keyword_count >= 2
        has_isbn = 'isbn' in text_lower
        
        return has_multiple_keywords or (has_isbn and keyword_count >= 1)

    def get_toc(self) -> List[Dict[str, Any]]:
        """Extracts Table of Contents (TOC).

        Strategy:
            1. Try Metadata TOC (PDF bookmarks)
            2. If failed or empty, try Visual TOC extraction
            3. If both fail, return empty list

        Returns:
            List of TOC entries. Each entry dict:
              - `level` (int): Heading level (1=main, 2=sub, etc.)
              - `title` (str): Heading text
              - `page` (int): Target page number (1-indexed)
        """
        if not self.doc:
            raise PDFParserError("PDF file is not open.")
        
        # STEP 1: Metadata TOC (PDF bookmarks)
        toc_raw = self.doc.get_toc(simple=False) or []
        page_count = self.get_page_count()
        results: List[Dict[str, Any]] = []

        for item in toc_raw:
            try:
                level, title, page = item[0], item[1], item[2]
                if len(title) > 100:
                    continue
                # Page number must be in range (1-indexed)
                if not (isinstance(page, (int, float)) and 1 <= int(page) <= page_count):
                    continue
                results.append({"level": level, "title": title, "page": int(page)})
            except (IndexError, TypeError):
                continue

        # STEP 2: Return Metadata TOC if found
        if results:
            return results
        
        # STEP 3: If no metadata TOC, try visual TOC
        try:
            visual_toc = self._extract_visual_toc()
            if visual_toc:
                return visual_toc
        except Exception:
            # If error during Visual TOC, return empty list
            pass
        
        # STEP 4: Return results (empty if failed)
        return results

    def _extract_visual_toc(self) -> List[Dict[str, Any]]:
        """Extracts headings from a visually laid out Table of Contents page.

        Algorithm:
            1. Find TOC page: search for 'Contents', 'Index' etc. (from config)
            2. Extract lines matching format "Title .... Page" via regex
            3. Determine hierarchy (Level 1, 2) by comparing x-coordinates
            4. Return extracted headings

        Returns:
            List of extracted headings. Each entry dict:
              - `level` (int)
              - `title` (str)
              - `page` (int)
        """
        if not self.doc:
            raise PDFParserError("PDF file is not open.")

        visual_toc = []
        
        # TOC page search keywords (from config)
        toc_keywords = self.config.get("toc_keywords", [
            'contents', 'index', 'table of contents'
        ])
        
        # Helper function: Unicode-normalize text
        def normalize_text(text: str) -> str:
            """Unicode normalize and lowercase (safe for international characters)"""
            # NFD: decompose combining marks
            normalized = unicodedata.normalize('NFKD', text.lower())
            # Remove combining marks
            return ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
        
        # Scan all pages
        for page_idx in range(self.doc.page_count):
            try:
                page = self.doc.load_page(page_idx)
                raw_text = page.get_text("text") or ""
            except Exception:
                continue
            
            # Check if TOC page
            raw_text_normalized = normalize_text(raw_text)
            
            is_toc_page = False
            for keyword in toc_keywords:
                keyword_normalized = normalize_text(keyword)
                if keyword_normalized in raw_text_normalized:
                    is_toc_page = True
                    break
            
            if not is_toc_page:
                continue
            
            # TOC page found - get lines with coords
            try:
                dict_text = page.get_text("dict") or {}
                blocks = dict_text.get("blocks", []) or []
            except Exception:
                continue

            lines_with_coords: List[Dict[str, Any]] = []
            for block in blocks:
                if not isinstance(block, dict) or block.get("type") != "text":
                    continue
                block_lines = block.get("lines") or []
                for line in block_lines:
                    spans = line.get("spans") or []
                    if not spans:
                        continue
                    first = spans[0]
                    x0 = first.get("x0", 0)
                    bbox = first.get("bbox")
                    if bbox and len(bbox) >= 4:
                        y0 = bbox[1]
                    else:
                        y0 = line.get("bbox", [0, 0, 0, 0])[1] if isinstance(line.get("bbox"), (list, tuple)) else 0
                    line_text = "".join(s.get("text", "") for s in spans).strip()
                    if line_text and len(line_text) > 1:
                        lines_with_coords.append({"text": line_text, "x0": x0, "y0": y0})

            toc_entry_pattern = re.compile(r'^(.+?)\s+[\.\s]*(\d+)\s*$')
            # Multi-column layout handling
            column_gap_threshold = 50.0
            if not lines_with_coords:
                pass
            else:
                lines_sorted = sorted(lines_with_coords, key=lambda L: (L["x0"], L["y0"]))
                columns: List[List[Dict[str, Any]]] = []
                current_column: List[Dict[str, Any]] = []
                prev_x0: Optional[float] = None
                for L in lines_sorted:
                    x0 = L["x0"]
                    if prev_x0 is not None and (x0 - prev_x0) > column_gap_threshold:
                        if current_column:
                            columns.append(current_column)
                        current_column = []
                    current_column.append(L)
                    prev_x0 = x0
                if current_column:
                    columns.append(current_column)

                # Heuristic mapping for hierarchy levels
                all_x0 = sorted(set(L["x0"] for L in lines_with_coords))
                x_to_level = {x: min(i + 1, 2) for i, x in enumerate(all_x0)}

                for column_lines in columns:
                    # Sort by y0 within column
                    column_lines_sorted = sorted(column_lines, key=lambda L: L["y0"])
                    for item in column_lines_sorted:
                        line_text = item["text"]
                        x0 = item["x0"]
                        match = toc_entry_pattern.match(line_text)
                        if match:
                            title = match.group(1).strip()
                            page_str = match.group(2).strip()
                            try:
                                target_page = int(page_str)
                            except ValueError:
                                continue
                            if not (1 <= target_page <= self.doc.page_count):
                                continue
                            if len(title) > 100 or len(title) < 2:
                                continue
                            if title.lower() in toc_keywords or title.lower().replace(" ", "") in [k.replace(" ", "") for k in toc_keywords]:
                                continue
                            level = x_to_level.get(x0, 1)
                            visual_toc.append({"level": level, "title": title, "page": target_page})

            # Use only the first TOC page found? (Original logic had break, kept for fidelity but noted)
            if visual_toc:
                break

        return visual_toc

    def get_headings_by_font(self) -> List[Dict[str, Any]]:
        """Detects headings on all pages based on font size.

        Algorithm:
            1. Collect font info for text blocks on all pages.
            2. Calculate most frequent font size (body text size).
            3. Mark blocks with larger font size as headings.

        Returns:
            List of detected headings. Each entry dict:
              - `page_number` (int)
              - `font_size` (float)
              - `text` (str)
              - `bbox` (tuple): Block bounds (x0, y0, x1, y1)
        """
        if not self.doc:
            raise PDFParserError("PDF file is not open.")

        # Collect all font sizes
        font_sizes: List[float] = []
        block_data: List[Dict[str, Any]] = []

        for page_idx in range(self.doc.page_count):
            try:
                page = self.doc.load_page(page_idx)
                d = page.get_text("dict") or {}
                blocks = d.get("blocks", []) or []
            except Exception:
                continue

            for b in blocks:
                if not isinstance(b, dict) or b.get("type") != "text":
                    continue

                lines = b.get("lines") or []
                for line in lines:
                    spans = line.get("spans") or []
                    for span in spans:
                        size = span.get("size", 0)
                        text = span.get("text", "").strip()
                        if size > 0 and text:
                            font_sizes.append(size)
                            block_data.append({
                                "page_number": page_idx + 1,
                                "font_size": size,
                                "text": text,
                                "bbox": b.get("bbox"),
                            })

        # Calculate main text font size (mode)
        if not font_sizes:
            return []

        from collections import Counter
        size_counts = Counter(font_sizes)
        main_font_size = size_counts.most_common(1)[0][0]

        # Caption exclusion
        caption_prefixes = [
            p.strip().lower()
            for p in self.config.get("caption_prefixes", ["figure", "table", "chart"])
            if isinstance(p, str) and p.strip()
        ]

        # Select headings larger than main font size
        headings: List[Dict[str, Any]] = []
        for item in block_data:
            if item["font_size"] <= main_font_size:
                continue
            text_lower = (item.get("text") or "").strip().lower()
            if any(text_lower.startswith(prefix) for prefix in caption_prefixes):
                continue
            headings.append(item)

        return headings

    def detect_headings_from_content(
        self, 
        debug: bool = False, 
        force_content: bool = False,
        keywords: Optional[List[str]] = None,
        max_pages: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Detects headings from page content (if TOC missing or `force_content=True`).

        Config-driven and language-agnostic. Behavior controlled by self.config and keywords.

        Behavior:
          - If `force_content=True`, content-based detection runs regardless.
          - If `force_content=False` and TOC exists, TOC is used directly (more reliable).

        Conservative fallback: accepts only short candidates followed by description.

        Smart Blacklist: filters lines consisting ONLY of blacklist words without context.
        E.g. "Example" -> Blocked, "Example of Revolution" -> Accepted.

        Args:
            debug (bool): If True, print accept/reject reasons.
            force_content (bool): If True, run detection even if TOC exists.
            keywords (Optional[List[str]]): Specific text blacklist.
                If provided, overrides config blacklist.

        Returns:
            List of detected headings. Each entry dict:
              - `page_number` (int)
              - `text` (str)
              - `confidence` (float)
        """
        if not self.doc:
            raise PDFParserError("PDF file is not open.")
        
        # Get heading heuristics from config
        heading_config = self.config.get("heading_heuristics", {})
        max_chars = heading_config.get("max_chars", 20)
        max_words = heading_config.get("max_words", 2)
        min_description_chars = heading_config.get("min_description_chars", 50)
        bullet_chars = heading_config.get("filter_bullet_chars", ['•', '-', '*'])
        caption_prefixes = [
            p.strip().lower()
            for p in self.config.get("caption_prefixes", ["figure", "table", "chart"])
            if isinstance(p, str) and p.strip()
        ]

        # Blacklist setup
        if keywords:
            active_blacklist = set(word.lower() for word in keywords)
        else:
            active_blacklist = self.blacklist
        
        def is_blacklisted(text: str, blacklist_set: Set[str]) -> bool:
            """
            Block lines consisting ONLY of blacklist words with no context.
            Case-insensitive.
            """
            words = text.strip().split()
            if not words:
                return False
            
            all_in_blacklist = all(w.lower() in blacklist_set for w in words)
            return all_in_blacklist

        # Use TOC if available and not forced
        if not force_content:
            try:
                toc_entries = self.get_toc()
                if toc_entries:
                    return [
                        {
                            "page_number": entry.get("page", 1),
                            "text": entry["title"],
                            "confidence": 0.95,
                            "level": entry.get("level", 1),
                        }
                        for entry in toc_entries
                    ]
            except Exception:
                pass

        pages = self.get_pages()
        if max_pages:
            pages = pages[:max_pages]

        # Compute noise set
        try:
            noise_set = self.detect_repeating_headers_footers()
        except Exception:
            noise_set = set()
        detected_headings: List[Dict[str, Any]] = []
        seen_headings = set()

        for page in pages:
            raw = page["raw_text"]
            lines = raw.split("\n")

            # Header/footer removal
            filtered_lines = []
            for line in lines:
                ln = line.strip()
                if not ln or len(ln) <= 2 or ln.isdigit():
                    continue
                if self._normalize_line(ln) in noise_set:
                    continue
                filtered_lines.append(ln)

            # Heuristic checks
            for i, line in enumerate(filtered_lines):
                if not line:
                    continue

                line_len = len(line)
                word_count = len(line.split())

                is_within_char_limit = 2 <= line_len <= max_chars
                is_within_word_limit = word_count <= max_words

                if not (is_within_char_limit and is_within_word_limit):
                    if debug:
                        reason = f"too long (>{max_chars})" if line_len > max_chars else f"too many words (>{max_words})"
                        print(f"  Page {page['page_number']}: SKIP '{line[:30]}...' ({reason})")
                    continue

                # FILTER 1: Bullet Points
                if line.strip() and line.strip()[0] in bullet_chars:
                    if debug:
                        print(f"  Page {page['page_number']}: SKIP '{line}' (starts with bullet/dash)")
                    continue

                # FILTER 1.4: Caption Exclusion
                line_lower = line.strip().lower()
                if any(line_lower.startswith(prefix) for prefix in caption_prefixes):
                    if debug:
                        print(f"  Page {page['page_number']}: SKIP '{line}' (caption: Figure/Table/Chart)")
                    continue

                # FILTER 1.5: Smart Blacklist
                if active_blacklist and is_blacklisted(line, active_blacklist):
                    if debug:
                        print(f"  Page {page['page_number']}: SKIP '{line}' (blacklisted word without context)")
                    continue

                # FILTER 2: Followed by description?
                has_description_after = False
                next_line_len = 0
                if i + 1 < len(filtered_lines):
                    next_line = filtered_lines[i + 1]
                    next_line_len = len(next_line)
                    if len(next_line) > min_description_chars:
                        has_description_after = True
                
                # FILTER 3: Punctuation
                if line.endswith(('.', ',')):
                    if debug:
                        print(f"  Page {page['page_number']}: SKIP '{line}' (ends with period/comma)")
                    continue
                
                if line and line[-1].islower() and not has_description_after:
                    if debug:
                        print(f"  Page {page['page_number']}: SKIP '{line}' (ends with lowercase, no description after)")
                    continue
                
                # FILTER 4: Case Sensitivity
                uppercase_count = sum(1 for c in line if c.isupper())
                lowercase_count = sum(1 for c in line if c.islower())
                first_char_upper = line[0].isupper() if line else False
                
                if not has_description_after:
                    # Strict: need UPPERCASE or mostly uppercase
                    if lowercase_count > uppercase_count or not first_char_upper:
                        if debug:
                            print(f"  Page {page['page_number']}: SKIP '{line}' (case mismatch: first_upper={first_char_upper}, upper={uppercase_count}, lower={lowercase_count})")
                        continue
                else:
                    # Lenient: Title case enough
                    if not first_char_upper:
                        if debug:
                            print(f"  Page {page['page_number']}: SKIP '{line}' (first letter must be uppercase)")
                        continue

                if has_description_after:
                    heading_key = line.lower()
                    if heading_key not in seen_headings and not heading_key.isdigit():
                        confidence = 0.85
                        
                        # Heuristic level determination
                        upper_count = sum(1 for c in line if c.isupper())
                        lower_count = sum(1 for c in line if c.islower())
                        if lower_count == 0:
                            heading_level = 1
                        else:
                            heading_level = 2

                        detected_headings.append({
                            "page_number": page["page_number"],
                            "text": line,
                            "confidence": confidence,
                            "level": heading_level,
                        })
                        seen_headings.add(heading_key)
                        if debug:
                            print(f"  Page {page['page_number']}: ACCEPT '{line}' (level={heading_level}) ✅")

        return detected_headings

    def _topics_from_heading_lines(
        self,
        full_text_by_lines: List[str],
        heading_lines: List[tuple],
        content_starts_after_heading: bool,
        is_copyright_topic: Any,
    ) -> List[Dict[str, Any]]:
        """Generates list of topics from heading lines.

        Args:
            full_text_by_lines: All text lines.
            heading_lines: List of (line_no, title, level, page_num), sorted by line_no.
            content_starts_after_heading: If True, content starts line_no+1.
            is_copyright_topic: (title, page_num) -> bool; if True, skip topic.

        Returns:
            List of dicts containing topic_title, start_line, end_line, content.
        """
        topics: List[Dict[str, Any]] = []
        for i, (line_no, title, level, page_num) in enumerate(heading_lines):
            if is_copyright_topic(title, page_num):
                continue
            start_line = line_no + 1 if content_starts_after_heading else line_no
            if i + 1 < len(heading_lines):
                end_line = heading_lines[i + 1][0] - 1
            else:
                end_line = len(full_text_by_lines) - 1
            if start_line > end_line:
                continue
            content = "\n".join(full_text_by_lines[start_line : end_line + 1])
            topic_dict: Dict[str, Any] = {
                "topic_title": title,
                "start_line": start_line,
                "end_line": end_line,
                "content": content,
                "page_number": page_num,
            }
            if level is not None:
                topic_dict["level"] = level
            topics.append(topic_dict)
        return topics

    def get_text_by_topics(self) -> List[Dict[str, Any]]:
        """Segments text into Topics based on detected headings.

        Uses line-based segmentation: finds heading location, reads until next heading.
        Prioritizes Font-based -> TOC -> Content-based detections.
        Excludes copyright pages.

        Returns:
            List of topics. Each topic dict:
              - `topic_title` (str)
              - `start_line` (int)
              - `end_line` (int)
              - `content` (str)
        """
        if not self.doc:
            raise PDFParserError("PDF file is not open.")

        # Collect all lines
        pages = self.get_pages()
        all_lines = []
        page_line_map = {}
        copyright_page_numbers = set()
        current_line = 0

        for page in pages:
            start_line = current_line
            lines = page["clean_text"].split("\n")
            for line in lines:
                all_lines.append(line)
                current_line += 1
            page_line_map[page["page_number"]] = (start_line, current_line - 1)
            if page.get("is_copyright"):
                copyright_page_numbers.add(page["page_number"])

        full_text_by_lines = all_lines

        def find_heading_line(heading_text, page_num):
            """Find line number of heading on page."""
            if page_num not in page_line_map:
                return None
            
            start_line, end_line = page_line_map[page_num]
            heading_lower = heading_text.lower().strip()
            
            search_start = start_line
            search_end = end_line + 1
            
            # 1. Exact match
            for i in range(search_start, search_end):
                if i < len(full_text_by_lines):
                    line = full_text_by_lines[i].lower().strip()
                    if line == heading_lower:
                        return i
            
            # 2. Partial match
            heading_words = heading_lower.split()
            if heading_words:
                for i in range(search_start, search_end):
                    if i < len(full_text_by_lines):
                        line = full_text_by_lines[i].lower()
                        if all(word in line for word in heading_words):
                            return i
            
            # 3. First word match
            first_word = heading_words[0] if heading_words else ""
            if first_word:
                for i in range(search_start, search_end):
                    if i < len(full_text_by_lines):
                        line = full_text_by_lines[i].lower()
                        if first_word in line:
                            return i
            
            return None

        def is_copyright_topic(title: str, page_num: int) -> bool:
            return page_num in copyright_page_numbers

        # 1. TOC (Metadata & Visual) - Prioritize explicit structure
        toc = self.get_toc()
        if toc:
            h_lines = []
            for toc_item in toc:
                found_line = find_heading_line(toc_item["title"], toc_item["page"])
                if found_line is not None:
                    h_lines.append((found_line, toc_item["title"], toc_item.get("level", 1), toc_item["page"]))
            if h_lines:
                h_lines.sort(key=lambda x: x[0])
                unique = []
                seen = set()
                for t in h_lines:
                    if t[0] not in seen:
                        unique.append(t)
                        seen.add(t[0])
                topics = self._topics_from_heading_lines(
                    full_text_by_lines, unique, content_starts_after_heading=True, is_copyright_topic=is_copyright_topic
                )
                if topics:
                    return topics

        # 2. Font-based detection
        headings = self.get_headings_by_font()
        if headings:
            headings_sorted = sorted(headings, key=lambda h: (h["page_number"], h["bbox"][1] if h["bbox"] else 0))
            h_lines = []
            for h in headings_sorted:
                found_line = find_heading_line(h["text"], h["page_number"])
                if found_line is not None:
                    h_lines.append((found_line, h["text"], None, h["page_number"]))
            if h_lines:
                h_lines.sort(key=lambda x: x[0])
                topics = self._topics_from_heading_lines(
                    full_text_by_lines, h_lines, content_starts_after_heading=False, is_copyright_topic=is_copyright_topic
                )
                if topics:
                    return topics

        # 3. Content-based
        detected = self.detect_headings_from_content(force_content=True)
        if detected:
            h_lines = []
            for h in detected:
                found_line = find_heading_line(h["text"], h["page_number"])
                if found_line is not None:
                    h_lines.append((found_line, h["text"], h.get("level", 1), h["page_number"]))
            if h_lines:
                h_lines.sort(key=lambda x: x[0])
                topics = self._topics_from_heading_lines(
                    full_text_by_lines, h_lines, content_starts_after_heading=True, is_copyright_topic=is_copyright_topic
                )
                if topics:
                    return topics

        # 4. Fallback: all content
        return [{
            "topic_title": "Full Content",
            "start_line": 0,
            "end_line": len(full_text_by_lines) - 1,
            "content": "\n".join(full_text_by_lines),
            "level": 1,
        }]


    def get_text_all(self) -> str:
        """Returns all text as a single string (cleaned)."""
        if not self.doc:
            raise PDFParserError("PDF file is not open.")
        parts: List[str] = []
        for p in self.get_pages():
            if p.get("clean_text"):
                parts.append(p["clean_text"])
        return "\n\n".join(parts)

    def analyze_page_structure(self, max_pages: Optional[int] = None) -> List[Dict[str, Any]]:
        """Analyzes structure of each page: image/table count, OCR need.

        Heuristics:
            - `images_count`: using `page.get_images(full=True)`.
            - `tables_count`: patterns of multi-column text blocks.
            - `is_ocr_needed`: True if clean text is empty but images exist.

        Returns:
            List of dicts per page.
        """
        if not self.doc:
            raise PDFParserError("PDF file is not open.")

        results: List[Dict[str, Any]] = []

        # --- Collect all image blocks across pages ---
        all_images: List[Dict[str, Any]] = []
        limit = self.doc.page_count
        if max_pages:
            limit = min(limit, max_pages)
            
        for page_idx in range(limit):
            try:
                page = self.doc.load_page(page_idx)
                d = page.get_text("dict") or {}
                blocks = d.get("blocks", []) or []
            except Exception:
                continue

            page_w = page.rect.width
            page_h = page.rect.height

            for b in blocks:
                if not isinstance(b, dict):
                    continue
                btype = b.get("type")
                if btype not in ("image", 1):
                    # Some PDFs place image info in block regardless; also check 'image' key
                    if not b.get("image"):
                        continue

                bbox = b.get("bbox")
                if not bbox or len(bbox) < 4:
                    continue

                x0, y0, x1, y1 = bbox
                w = max(0.0, x1 - x0)
                h = max(0.0, y1 - y0)
                area = w * h
                area_pct = area / (page_w * page_h) if (page_w * page_h) else 0.0

                img_info = b.get("image") or {}
                img_xref = img_info.get("xref") if isinstance(img_info, dict) else None

                all_images.append({
                    "page": page_idx + 1,
                    "bbox": (x0, y0, x1, y1),
                    "w": w,
                    "h": h,
                    "area_pct": area_pct,
                    "xref": img_xref,
                })

        # --- Group images to find repeating ones ---
        from collections import defaultdict
        groups = defaultdict(list)
        for img in all_images:
            key = img["xref"] if img["xref"] is not None else (round(img["area_pct"], 4), round(img["bbox"][0], 1), round(img["bbox"][1], 1))
            groups[key].append(img)

        decorative_keys = set()
        for key, imgs in groups.items():
            pages_seen = set(i["page"] for i in imgs)
            if len(pages_seen) < 3:
                continue
            # check coordinate similarity (centers within 5% page dims)
            centers = []
            for im in imgs:
                x0, y0, x1, y1 = im["bbox"]
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0
                centers.append((im["page"], cx, cy))
            # compute average center
            avg_cx = sum(c[1] for c in centers) / len(centers)
            avg_cy = sum(c[2] for c in centers) / len(centers)
            similar = True
            for (pg, cx, cy) in centers:
                page = self.doc.load_page(pg - 1)
                tol_x = page.rect.width * 0.05
                tol_y = page.rect.height * 0.05
                if abs(cx - avg_cx) > tol_x or abs(cy - avg_cy) > tol_y:
                    similar = False
                    break
            if similar:
                decorative_keys.add(key)

        # --- Produce per-page results excluding decorative images ---
        for page_idx in range(limit):
            page = self.doc.load_page(page_idx)
            raw = page.get_text("text") or ""
            clean = self.clean_text(raw)

            page_w = page.rect.width
            page_h = page.rect.height

            imgs_on_page = [img for img in all_images if img["page"] == page_idx + 1]
            decorative_count = 0
            meaningful_count = 0

            for img in imgs_on_page:
                x0, y0, x1, y1 = img["bbox"]
                area_pct = img["area_pct"]
                top_zone = y0 < (page_h * 0.10)
                bottom_zone = y1 > (page_h * 0.90)
                key = img["xref"] if img["xref"] is not None else (round(area_pct, 4), round(x0, 1), round(y0, 1))

                is_decorative = False
                if area_pct < 0.03:
                    is_decorative = True
                if top_zone or bottom_zone:
                    is_decorative = True
                if key in decorative_keys:
                    is_decorative = True

                if is_decorative:
                    decorative_count += 1
                else:
                    meaningful_count += 1

            # fallback: if no image blocks found, fall back to page.get_images count
            if not imgs_on_page:
                try:
                    total_images = len(page.get_images(full=True))
                except Exception:
                    total_images = 0
                meaningful_count = total_images

            # Table detection
            tables_count = 0
            try:
                blocks = page.get_text("blocks") or []
            except Exception:
                blocks = []

            for b in blocks:
                text_block = ""
                if isinstance(b, dict):
                    text_block = b.get("text", "") or ""
                elif isinstance(b, (list, tuple)) and len(b) >= 5:
                    text_block = b[4] or ""
                else:
                    continue

                lines = [ln for ln in text_block.splitlines() if ln.strip()]
                if len(lines) < 3:
                    continue

                cols_per_line = []
                for ln in lines:
                    tokens = [t for t in re.split(r"\s{2,}|\t|\|", ln.strip()) if t.strip()]
                    cols_per_line.append(len(tokens))

                if not cols_per_line:
                    continue

                avg_cols = sum(cols_per_line) / len(cols_per_line)
                pct_multi = sum(1 for c in cols_per_line if c >= 2) / len(cols_per_line)

                if avg_cols >= 2 and pct_multi >= 0.6:
                    tables_count += 1

            # Extra heuristic: drawings
            try:
                drawings = []
                if hasattr(page, "get_drawings"):
                    drawings = page.get_drawings() or []
                line_like = 0
                for d in drawings:
                    t = d.get("type") if isinstance(d, dict) else None
                    if t in ("line", "rect"):
                        line_like += 1
                if line_like >= 6 and tables_count == 0:
                    tables_count += 1
            except Exception:
                pass

            is_ocr_needed = (len(clean.strip()) == 0 and meaningful_count > 0)

            results.append({
                "page_number": page_idx + 1,
                "images_count": meaningful_count,
                "decorative_images": decorative_count,
                "tables_count": tables_count,
                "is_ocr_needed": is_ocr_needed,
                "raw_text": raw,
                "clean_text": clean,
            })

        return results


if __name__ == '__main__':
    # Simple example usage / test scenario
    sample = "test.pdf"
    print("PDFParser example run — file:", sample)
    try:
        parser = PDFParser(sample)
        meta = parser.get_metadata()
        print("\nMetadata:")
        for k, v in meta.items():
            print(f"- {k}: {v}")

        print(f"\nPage count: {parser.get_page_count()}")
        pages = parser.get_pages()
        if pages:
            print("\nPreview of cleaned text from first page:\n")
            print(pages[0]["clean_text"][:1000])
        else:
            print("(PDF is empty or could not be read)")
    except Exception as e:
        print(f"Error: {e}")