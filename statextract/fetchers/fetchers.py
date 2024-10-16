from pathlib import Path
from typing import Protocol

from statextract.helpers import form_path_base
from statextract.typedefs import PaperMD

class PaperFetcher(Protocol):
    async def fetch(self, md: PaperMD) -> bytes | None:
        ...
        
    async def get_pdf_url(self, md: PaperMD) -> str | None:
        ...
        
        
class CachingPaperFetcher:
    def __init__(self, fetcher: PaperFetcher, pdf_dir: Path = Path('./data/pdfs')):
        self.fetcher = fetcher
        self.pdf_dir = pdf_dir
        if not self.pdf_dir.exists():
            self.pdf_dir.mkdir(parents=True)
            
    def _get_path(self, md: PaperMD) -> Path:
        return self.pdf_dir / f"{form_path_base(md)}.pdf"
    
    async def fetch(self, md: PaperMD) -> bytes | None:
        path = self._get_path(md)
        if not path.exists():
            pdf = await self.fetcher.fetch(md)
            if pdf is None:
                return None
            path.write_bytes(pdf)
        return path.read_bytes()
    
    async def get_pdf_url(self, md: PaperMD) -> str | None:
        return await self.fetcher.get_pdf_url(md)
        
        
class CombinedPaperFetcher:
    def __init__(self, fetchers: list[PaperFetcher]):
        self.fetchers = fetchers
        
    async def fetch(self, md: PaperMD) -> bytes | None:
        for fetcher in self.fetchers:
            res = await fetcher.fetch(md)
            if res is not None:
                return res
        return None
    
    async def get_pdf_url(self, md: PaperMD) -> str | None:
        for fetcher in self.fetchers:
            res = await fetcher.get_pdf_url(md)
            if res is not None:
                return res
        return None