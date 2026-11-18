"""
Document text extraction.

Supports: PDF (PyMuPDF), TXT, Markdown, CSV, XLSX.
Each extractor returns a list of Page objects (text + page number).
"""

from __future__ import annotations

import csv
import io
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from educore.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ExtractedPage:
    """One logical page/section of extracted text."""
    text: str
    page_number: int
    metadata: dict = field(default_factory=dict)


class DocumentExtractor:
    """Dispatches extraction based on file type."""

    SUPPORTED_TYPES = {"pdf", "txt", "md", "csv", "xlsx"}

    @classmethod
    def extract(cls, file_path: str, file_type: str) -> list[ExtractedPage]:
        """
        Extract text from a file.

        Args:
            file_path: Absolute path to the stored file.
            file_type: Extension without dot (pdf, txt, md, csv, xlsx).

        Returns:
            List of ExtractedPage with text and page numbers.

        Raises:
            ValueError: If the file type is not supported.
        """
        file_type = file_type.lower().strip(".")
        if file_type not in cls.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported file type: {file_type}. "
                f"Supported: {', '.join(sorted(cls.SUPPORTED_TYPES))}"
            )

        handler = {
            "pdf": cls._extract_pdf,
            "txt": cls._extract_text,
            "md": cls._extract_text,
            "csv": cls._extract_csv,
            "xlsx": cls._extract_xlsx,
        }[file_type]

        pages = handler(file_path)
        logger.info("Extracted %d pages from %s (%s)", len(pages), file_path, file_type)
        return pages

    @staticmethod
    def _extract_pdf(file_path: str) -> list[ExtractedPage]:
        """Extract selectable PDF text, using OCR for scanned/image-only pages."""
        import pymupdf

        pages = []
        ocr_config = None
        with pymupdf.open(file_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                has_text = bool(text.strip())

                # Image-only scans have no text layer. Also OCR image-heavy pages
                # with only a small text layer so labels inside figures are kept.
                if not has_text or (len(text.strip()) < 40 and page.get_images(full=True)):
                    if ocr_config is None:
                        ocr_config = DocumentExtractor._prepare_ocr(pymupdf)
                    try:
                        text_page = page.get_textpage_ocr(
                            language=ocr_config["language"],
                            dpi=settings.OCR_DPI,
                            full=not has_text,
                            tessdata=str(ocr_config["tessdata"]),
                        )
                        ocr_text = page.get_text("text", textpage=text_page)
                    except Exception as exc:
                        raise RuntimeError(
                            "OCR failed for this scanned PDF. Check that Tesseract OCR and the "
                            f"required language data ({settings.OCR_LANGUAGES}) are installed. "
                            f"Details: {exc}"
                        ) from exc

                    if len(ocr_text.strip()) > len(text.strip()):
                        text = ocr_text

                if text.strip():
                    pages.append(ExtractedPage(
                        text=text.strip(),
                        page_number=page_num,
                        metadata={"total_pages": len(doc)},
                    ))

        if not pages:
            raise ValueError(
                "No readable text was found in this PDF, even after OCR. "
                "The scan may be blank or too low quality."
            )
        return pages

    @staticmethod
    def _prepare_ocr(pymupdf_module) -> dict[str, object]:
        """Find Tesseract and the best available subset of configured OCR languages."""
        configured_exe = settings.TESSERACT_CMD.strip()
        executable = Path(configured_exe).expanduser() if configured_exe else None
        if not executable or not executable.is_file():
            found = shutil.which("tesseract")
            executable = Path(found) if found else None

        if not executable or not executable.is_file():
            program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            candidates = [
                program_files / "Tesseract-OCR" / "tesseract.exe",
                Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
            ]
            executable = next((candidate for candidate in candidates if candidate.is_file()), None)

        if not executable:
            raise RuntimeError(
                "This PDF is scanned and has no selectable text. Install Tesseract OCR, "
                "then install the required language packs and retry indexing."
            )

        # Tesseract was installed after the current shell started, or is installed
        # outside PATH. Make it visible to MuPDF in this worker process.
        executable = executable.resolve()
        os.environ["PATH"] = str(executable.parent) + os.pathsep + os.environ.get("PATH", "")

        requested_languages = [
            language.strip()
            for language in settings.OCR_LANGUAGES.split("+")
            if language.strip()
        ]
        candidate_dirs: list[Path] = []
        if settings.TESSDATA_DIR.strip():
            candidate_dirs.append(Path(settings.TESSDATA_DIR).expanduser())
        if os.environ.get("TESSDATA_PREFIX"):
            candidate_dirs.append(Path(os.environ["TESSDATA_PREFIX"]).expanduser())
        candidate_dirs.append((settings.storage_path / "tessdata").resolve())
        candidate_dirs.append(executable.parent / "tessdata")
        try:
            discovered = Path(pymupdf_module.get_tessdata())
            candidate_dirs.append(discovered)
        except (RuntimeError, OSError):
            pass

        seen: set[str] = set()
        best_dir: Path | None = None
        best_languages: list[str] = []
        for directory in candidate_dirs:
            directory = directory.expanduser().resolve()
            key = str(directory).casefold()
            if key in seen or not directory.is_dir():
                continue
            seen.add(key)
            available = [
                language
                for language in requested_languages
                if (directory / f"{language}.traineddata").is_file()
            ]
            if len(available) > len(best_languages):
                best_dir, best_languages = directory, available

        if not best_dir or not best_languages:
            raise RuntimeError(
                "Tesseract is installed, but none of the requested OCR language packs "
                f"({settings.OCR_LANGUAGES}) were found. Run scripts/setup_ocr.ps1 or set "
                "TESSDATA_DIR to a folder containing the matching .traineddata files."
            )

        os.environ["TESSDATA_PREFIX"] = str(best_dir)
        language = "+".join(best_languages)
        if len(best_languages) < len(requested_languages):
            logger.warning(
                "Some OCR language packs are missing; using %s from %s",
                language,
                best_dir,
            )
        else:
            logger.info("Using Tesseract OCR languages %s from %s", language, best_dir)

        return {"language": language, "tessdata": best_dir}

    @staticmethod
    def _extract_text(file_path: str) -> list[ExtractedPage]:
        """Extract text from plain text or markdown files."""
        path = Path(file_path)
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            return []
        return [ExtractedPage(text=text.strip(), page_number=1)]

    @staticmethod
    def _extract_csv(file_path: str) -> list[ExtractedPage]:
        """
        Extract CSV — one page per row.

        Each row becomes a structured text with column headers,
        making it easy to retrieve individual records.
        """
        pages = []
        path = Path(file_path)
        with path.open(encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, start=2):  # row 2 = first data row
                parts = [f"{col}: {val}" for col, val in row.items() if val]
                if parts:
                    pages.append(ExtractedPage(
                        text=" | ".join(parts),
                        page_number=row_num,
                        metadata={"source_type": "csv_row"},
                    ))
        return pages

    @staticmethod
    def _extract_xlsx(file_path: str) -> list[ExtractedPage]:
        """
        Extract XLSX — one page per row, across all sheets.

        Row numbers match Excel's numbering (header = row 1).
        """
        from openpyxl import load_workbook

        pages = []
        wb = load_workbook(file_path, read_only=True, data_only=True)

        for sheet in wb.worksheets:
            rows = list(sheet.iter_rows(values_only=True))
            if len(rows) < 2:
                continue

            # First row = headers
            headers = [str(h) if h else f"col_{i}" for i, h in enumerate(rows[0])]

            for row_num, row in enumerate(rows[1:], start=2):
                parts = []
                for col_name, val in zip(headers, row):
                    if val is not None:
                        parts.append(f"{col_name}: {val}")
                if parts:
                    pages.append(ExtractedPage(
                        text=" | ".join(parts),
                        page_number=row_num,
                        metadata={
                            "source_type": "xlsx_row",
                            "sheet_name": sheet.title,
                        },
                    ))

        wb.close()
        return pages
