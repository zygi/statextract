from bs4 import BeautifulSoup
import httpx
from joblib import Memory

@Memory('./.cache/lol_fetcher').cache
async def fetch_url(url: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response
    
async def fetch_from_library_lol(doi: tuple[str, str]):
    library_id, item_id = doi
    url = f"https://library.lol/scimag/{library_id}/{item_id}"
    # async with httpx.AsyncClient() as client:
        # print(url)
    response = await fetch_url(url)
    if response.status_code == 404:
        return None
    else:
        # parse the html, get the url at div#download h2 a
        soup = BeautifulSoup(response.text, 'html.parser')
        download_a = soup.select_one('div#download h2 a')
        if download_a is None:
            raise ValueError("Download URL not found, html: " + response.text)
        download_url = download_a['href']
        assert isinstance(download_url, str), "Download URL is not a string"
        
        # download the pdf
        async with httpx.AsyncClient() as client:
            pdf_response = await client.get(download_url)
        bts = pdf_response.content
        assert isinstance(bts, bytes), "PDF content is not bytes"
        return bts
        # return response.json()
        

class LibraryLolFetcher:
    async def fetch(self, doi: tuple[str, str]) -> bytes:
        res = await fetch_from_library_lol(doi)
        if res is None:
            raise ValueError(f"PDF not found for {doi}")
        return res
    
    