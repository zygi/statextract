from pathlib import Path
from joblib import Memory
import pyalex
import asyncio
import httpx
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm
import pymupdf4llm

from fetchers.fetchers import PaperFetcher
from helpers import DWECK, author_works, filter_mds_by_pdf, form_path_base, get_mds_with_dois
import helpers


# author = pyalex.Authors()["A5000930617"]

# print(author['works_api_url'])


# works = author_works("A5000930617")



# dois = [convert_doi(work['doi']) for work in works if 'doi' in work and work['doi'] is not None]
# print(dois)
mds = get_mds_with_dois(helpers.TSAI)



dois = [md.doi for md in mds]

# @Memory('./cache/web_response').cache


async def download_pdfs(dois: list[tuple[str, str]], fetcher: PaperFetcher, path: Path, concurrent: int = 1):
    if not path.exists():
        path.mkdir(parents=True)
    
    sem = asyncio.Semaphore(concurrent)
    async def download_pdf(doi: tuple[str, str]):
        library_id, item_id = doi
        full_path = path / f"{form_path_base(doi)}.pdf"
        # if full_path.exists():
        #     return
        async with sem:
            try:
                bts = await fetcher.fetch(doi)
                if bts is None:
                    print(f"Error downloading {doi}: not found")
                    return
            except Exception as e:
                print(f"Error downloading {doi}: {e}")
                return
            if bts is not None:
                full_path.write_bytes(bts)
    # nonexisting = [doi for doi in dois]
    nonexisting = [doi for doi in dois if not (path / f"{form_path_base(doi)}.pdf").exists()]
    return await tqdm.gather(*[download_pdf(doi) for doi in nonexisting])



def _pymupdf_extract_text(doi: tuple[str, str], pdf_path: Path, output_path: Path):
    full_output_path = output_path / f"{form_path_base(doi)}.md"
    if full_output_path.exists():
        return
    doc = pymupdf4llm.to_markdown(pdf_path / f"{form_path_base(doi)}.pdf")
    full_output_path.write_text(doc)

import multiprocessing

def extract_text(dois: list[tuple[str, str]], pdf_path: Path, output_path: Path, concurrent: int = 1):
    if not output_path.exists():
        output_path.mkdir(parents=True)
    with multiprocessing.Pool(concurrent) as pool:
        pool.starmap(_pymupdf_extract_text, [(doi, pdf_path, output_path) for doi in dois])


async def work():
    # print(await fetch_from_library_lol(dois[0]))
    await download_pdfs(dois, Path("pdfs"), concurrent=3)
    filtered_dois = [md.doi for md in filter_mds_by_pdf(mds)]
    extract_text(filtered_dois, Path("pdfs"), Path("mds"), concurrent=3)
    
    print(len(filtered_dois))
    
    # print(len(filtered_dois))
    
asyncio.run(work())
# fetch_from_library_lol(dois[0])