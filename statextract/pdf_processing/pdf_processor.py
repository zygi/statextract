
import abc
from pathlib import Path
import subprocess
from typing import Callable, Protocol

from statextract.typedefs import PaperMD
import pymupdf
import pymupdf4llm
class MalformedPDFException(Exception):
    """Wraps another exception"""
    def __init__(self, e: Exception):
        self.e = e

class PDFProcessingFn(Protocol):
    def __call__(self, md: PaperMD, pdf_path: Path, md_output_path: Path, image_output_path_prefix: Path) -> None:
        ...
        
        
def process_pdf_pymupdf(md: PaperMD, pdf_path: Path, md_output_path: Path, image_output_path_prefix: Path) -> None:
    if not pdf_path.exists():
        raise RuntimeError(f"PDF not found for {md.id}")
    if md_output_path.exists():
        return # already processed

    try:
        doc = pymupdf.Document(pdf_path)
    except Exception as e:
        raise MalformedPDFException(e)


    # first, check if the pdf needs to be OCRed
    full_text = ""
    try:
        for page in doc:
            full_text += page.get_text()
            pixmap = page.get_pixmap(dpi=120)
            pixmap.save(f"{image_output_path_prefix}-{page.number}.jpg", jpg_quality=80)
    except Exception as e:
        # stringify
        raise RuntimeError(f"Error extracting text from {md.id}: {e}")

    
    if len(full_text) < 1000:
        # redo ocr by calling `python -m ocrmypdf <pdf> <output> --redo-ocr`
        res = subprocess.run(["python", "-m", "ocrmypdf", pdf_path, pdf_path, "--redo-ocr"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            raise Exception(f"Error OCRing {md.id}: {res.returncode}: {res.stderr.decode('utf-8')}")
        doc = pymupdf.Document(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
            
    # now extract text using pymupdf4llm
    doc = pymupdf4llm.to_markdown(pdf_path)
    md_output_path.write_text(doc)
    return None


# def _pymupdf_extract_text(md: PaperMD, pdf_path: Path, output_path: Path, image_path: Path) -> None:
#     full_output_path = output_path / f"{form_path_base(md)}.md"
#     if full_output_path.exists():
#         return
#     if not pdf_path.exists():
#         raise RuntimeError(f"PDF not found for {md.id}")
    
    
#     # first, check if the pdf needs to be OCRed
#     full_text = ""
#     try:
#         for page in doc:
#             full_text += page.get_text()
#             pixmap = page.get_pixmap(dpi=120)
#             pixmap.save(image_path / f"{form_path_base(md)}-{page.number}.jpg", jpg_quality=80)
#     except Exception as e:
#         # stringify
#         raise RuntimeError(f"Error extracting text from {md.id}: {e}")
        
#     if len(full_text) < 1000:
#         # redo ocr by calling `python -m ocrmypdf <pdf> <output> --redo-ocr`
#         res = subprocess.run(["python", "-m", "ocrmypdf", pdf_path / f"{form_path_base(md)}.pdf", pdf_path / f"{form_path_base(md)}.pdf", "--redo-ocr"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         if res.returncode != 0:
#             raise Exception(f"Error OCRing {md.id}: {res.returncode}: {res.stderr.decode('utf-8')}")
#         doc = pymupdf.Document(pdf_path / f"{form_path_base(md)}.pdf")
#         full_text = ""
#         for page in doc:
#             full_text += page.get_text()
            
#     # now extract text using pymupdf4llm
#     doc = pymupdf4llm.to_markdown(pdf_path / f"{form_path_base(md)}.pdf")
#     full_output_path.write_text(doc)
#     return None
