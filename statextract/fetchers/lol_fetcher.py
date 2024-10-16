from bs4 import BeautifulSoup
import httpx
from joblib import Memory

from statextract.typedefs import PaperMD

@Memory('./.cache/lol_fetcher').cache
async def fetch_url(url: str):
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(url)
        if response.status_code == 503:
            raise Exception("503: " + response.text)
        return response
    
# async def fetch_from_library_lol(doi: tuple[str, str]):
#     library_id, item_id = doi
#     url = f"https://library.lol/scimag/{library_id}/{item_id}"
#     # async with httpx.AsyncClient() as client:
#         # print(url)
#     response = await fetch_url(url)
#     if response.status_code == 404:
#         return None
#     else:
#         try:
#             # parse the html, get the url at div#download h2 a
#             soup = BeautifulSoup(response.text, 'html.parser')
#             download_a = soup.select_one('div#download h2 a')
#             if download_a is None:
#                 raise ValueError("Download URL not found, html: " + response.text)
#             download_url = download_a['href']
#             assert isinstance(download_url, str), "Download URL is not a string"
            
#             # download the pdf
#             async with httpx.AsyncClient() as client:
#                 pdf_response = await client.get(download_url)
#             bts = pdf_response.content
#             assert isinstance(bts, bytes), "PDF content is not bytes"
#             return bts
#         except Exception as e:
#             print(f"Error fetching {doi}: {e}")
#             return None


class LibraryLolFetcher:
    
    async def get_pdf_url(self, md: PaperMD) -> str | None:
        if md.doi is None:
            return None
        library_id, item_id = md.doi
        url = f"https://library.lol/scimag/{library_id}/{item_id}"
        # async with httpx.AsyncClient() as client:
            # print(url)
        try:
            response = await fetch_url(url)
        except Exception as e:
            print(f"Error fetching {md.id}: {e}")
            return None
        if response.status_code == 404:
            return None
        else:
            try:
                # parse the html, get the url at div#download h2 a
                soup = BeautifulSoup(response.text, 'html.parser')
                download_a = soup.select_one('div#download h2 a')
                if download_a is None:
                    raise ValueError("Download URL not found, html: " + response.text)
                download_url = download_a['href']
                assert isinstance(download_url, str), "Download URL is not a string"
                return download_url
            except Exception as e:
                print(f"Error fetching {md.id}: {e}")
                return None
            
    
    async def fetch(self, md: PaperMD) -> bytes | None:
        pdf_url = await self.get_pdf_url(md)
        if pdf_url is None:
            return None
        res = await fetch_url(pdf_url)
        if res.status_code == 404:
            return None
        return res.content
