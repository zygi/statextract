from pathlib import Path
from typing import Protocol

from helpers import form_path_base

class PaperFetcher(Protocol):
    async def fetch(self, doi: tuple[str, str]) -> bytes:
        ...
        
        
class CachingPaperFetcher:
    def __init__(self, fetcher: PaperFetcher, pdf_dir: Path = Path('./data/pdfs')):
        self.fetcher = fetcher
        self.pdf_dir = pdf_dir
        if not self.pdf_dir.exists():
            self.pdf_dir.mkdir(parents=True)
            
    def _get_path(self, doi: tuple[str, str]) -> Path:
        return self.pdf_dir / f"{form_path_base(doi)}.pdf"
    
    async def fetch(self, doi: tuple[str, str]) -> bytes:
        path = self._get_path(doi)
        if not path.exists():
            pdf = await self.fetcher.fetch(doi)
            path.write_bytes(pdf)
        return path.read_bytes()
        