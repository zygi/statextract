import asyncio
from bs4 import BeautifulSoup
import httpx
from joblib import Memory
from pyalex import Works, Work
import pyalex

from statextract.helpers import assert_not_none
from statextract.typedefs import PaperMD
from statextract.md_retriever import parse_work
@Memory('./.cache/openalex_fetcher').cache
async def fetch_work(oalex_id: str) -> Work | None:
    try:
        work = Works()[oalex_id]
    except Exception as e:
        print(e)
        return None
    if work is None:
        return None
    assert isinstance(work, Work)
    return work

@Memory('./.cache/openalex_fetcher_pdfs').cache
async def fetch_pdf_url(url: str):
    # follow redirects
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, follow_redirects=True)
            return response.content
    except Exception as e:
        print(e)
        return None
    
# async def fetch_from_unpaywall(doi: tuple[str, str], email: str):
#     library_id, item_id = doi
#     url = f"https://api.unpaywall.org/v2/{library_id}/{item_id}?email={email}"
#     # async with httpx.AsyncClient() as client:
#     print(url)
#     response = await fetch_url(url)
#     if response.status_code == 404:
#         return None
#     else:
#         # parse the html, get the url at div#download h2 a
#         data = response.json()
#         print(data)
#         # return response.json()
        

class OpenAlexFetcher:
    async def get_pdf_url(self, md: PaperMD) -> str | None:
        try:
            res = await fetch_work(md.id)
        except Exception as e:
            print(e)
            return None
        if res is None:
            # raise ValueError(f"Record not found for {doi}")
            return None
        locs = res["locations"]
        pdf_url = None
        assert isinstance(locs, list)
        for loc in locs:
            assert isinstance(loc, dict)
            # special case for arxiv
            if loc["landing_page_url"].startswith("https://arxiv.org/abs/"):
                pdf_url = loc["landing_page_url"].replace("/abs/", "/pdf/") + ".pdf"
                break
            if "pdf_url" in loc and loc["pdf_url"] is not None:
                pdf_url = loc["pdf_url"]
                break
        return pdf_url
    
    
    async def fetch(self, md: PaperMD) -> bytes | None:
        pdf_url = await self.get_pdf_url(md)
        if pdf_url is None:
            return None
        res = await fetch_pdf_url(pdf_url)
        return res
    
    
if __name__ == "__main__":
    TEST_DOI = ("10.48550", "arXiv.1712.05812")
    
    async def work():
        # print(await fetch_work(TEST_DOI))
        fetcher = OpenAlexFetcher()
        print(await fetcher.fetch(assert_not_none(parse_work(pyalex.Works()[f"https://doi.org/{TEST_DOI[0]}/{TEST_DOI[1]}"]))))
    
    asyncio.run(work())
    