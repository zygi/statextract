import logging
import os
from pathlib import Path
import subprocess
from anthropic import AsyncAnthropic
import instructor
from joblib import Memory
# import multiprocess
import pyalex
import asyncio
import httpx
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm
import pymupdf4llm
from rich import print


from statextract import helpers
from statextract.fetchers.fetchers import CachingPaperFetcher, CombinedPaperFetcher, PaperFetcher
from statextract.helpers import DWECK, PaperMD, filter_mds_by_pdf, form_path_base
from statextract.fetchers.lol_fetcher import LibraryLolFetcher
from statextract.fetchers.openalex_fetcher import OpenAlexFetcher


import multiprocessing

from statextract.md_retriever import get_all_mds
from statextract.pdf_processing.pdf_processor import process_pdf_pymupdf
from statextract.prefilter import  mk_classify_fn, prefilter_regex

logger = logging.getLogger(__name__)


# author = pyalex.Authors()["A5000930617"]

# print(author['works_api_url'])


# works = author_works("A5000930617")



# dois = [convert_doi(work['doi']) for work in works if 'doi' in work and work['doi'] is not None]
# print(dois)

# print(len(dois))
# exit()

# @Memory('./cache/web_response').cache


async def download_pdfs(mds: list[PaperMD], fetcher: PaperFetcher, path: Path, concurrent: int = 1):
    if not path.exists():
        path.mkdir(parents=True)
    
    sem = asyncio.Semaphore(concurrent)
    async def download_pdf(md: PaperMD):
        full_path = path / f"{form_path_base(md)}.pdf"
        # if full_path.exists():
        #     return
        async with sem:
            try:
                bts = await fetcher.fetch(md)
                if bts is None:
                    print(f"Error downloading {md.id}: not found")
                    return
            except Exception as e:
                print(f"Error downloading {md.id}: {e}")
                return
            if bts is not None:
                full_path.write_bytes(bts)
    # nonexisting = [doi for doi in dois]
    nonexisting = [md for md in mds if not (path / f"{form_path_base(md)}.pdf").exists()]
    return await tqdm.gather(*[download_pdf(md) for md in nonexisting])



# def _pymupdf_extract_text(md: PaperMD, pdf_path: Path, output_path: Path, image_path: Path) -> None:
#     full_output_path = output_path / f"{form_path_base(md)}.md"
#     if full_output_path.exists():
#         return
#     if not pdf_path.exists():
#         raise RuntimeError(f"PDF not found for {md.id}")
    
    
#     import pymupdf
#     try:
#         doc = pymupdf.Document(pdf_path / f"{form_path_base(md)}.pdf")
#     except Exception as e:
#         # malformed?
#         assert "Failed to open file" in str(e), e
#         os.remove(pdf_path / f"{form_path_base(md)}.pdf")
#         return
        
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

def _call_process_pdf_pymupdf(md: PaperMD, pdf_path: Path, output_path: Path, image_path: Path):
    try:
        prefix = form_path_base(md)
        process_pdf_pymupdf(md, pdf_path / f"{prefix}.pdf", output_path / f"{prefix}.md", image_path / prefix)
        return True
    except Exception as e:
        print(f"Error processing {md.id}: {e}")
        return False

def extract_text(mds: list[PaperMD], pdf_path: Path, output_path: Path, image_path: Path, max_num_pages: int = 100000, concurrent: int = 10):
    if not output_path.exists():
        output_path.mkdir(parents=True)
    if not image_path.exists():
        image_path.mkdir(parents=True)
    with multiprocessing.Pool(concurrent) as pool:
        res = list(pool.starmap(_call_process_pdf_pymupdf, [(md, pdf_path, output_path, image_path) for md in mds]))
        return res

# ('10.1152', 'ajpgi.1987.253.5.g601')


async def work():
    # fetcher = CachingPaperFetcher(CombinedPaperFetcher([OpenAlexFetcher()]))
    fetcher = CachingPaperFetcher(CombinedPaperFetcher([LibraryLolFetcher(), OpenAlexFetcher()]))
    # fetcher = CombinedPaperFetcher([OpenAlexFetcher(), LibraryLolFetcher()])
    mds = get_all_mds('A5072310807', first_author=True)

    # for md in mds:
    #     print(md.id)
    #     print(md.title)
    #     print(md.doi)
    #     print(md.full['type'])
    #     print()

    # print(len(mds))
    # exit()


    # dois = [md.doi for md in mds]
    sem = asyncio.Semaphore(3)
    async def fetch_pdf(md: PaperMD):
        async with sem:
            return (md, await fetcher.fetch(md))
    
    pdf_urls = await asyncio.gather(*[fetch_pdf(md) for md in mds])
    # print(len([True for (_, pdf) in pdf_urls if pdf is not None]))
    
    print(f"total length: {len(pdf_urls)}, with pdf: {len([True for (_, pdf) in pdf_urls if pdf is not None])}")
    
    # for (md, pdf) in (res for res in pdf_urls if res[1] is not None):
    #     print(md.title)
    #     print(md.doi)
    #     print(md.full['type'])
    #     print(pdf)
    #     print()
    
    mds_with_pdfs = [md for (md, pdf) in pdf_urls if pdf is not None
                    #  if md.doi == ('10.1152', 'ajpgi.1987.253.5.g601')
                     ]
    
    print(len(mds_with_pdfs))
    
    # await download_pdfs(dois, Path("pdfs"), concurrent=3)
    # filtered_dois = [md.doi for md in filter_mds_by_pdf(mds)]
    extraction_status = extract_text(mds_with_pdfs, Path("data/pdfs"), Path("data/mds"), Path("data/images"), concurrent=3)
    
    mds_successful = [md for (md, success) in zip(mds_with_pdfs, extraction_status) if success]
    
    # import cloudpickle
    

    client = instructor.from_anthropic(AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"]))
    classify_fn = mk_classify_fn(client)
    
    async def pvalue_prefilter(md: PaperMD, text_path: Path):
        
        
        async with sem:
            if not text_path.exists():
                return None
            text = text_path.read_text()
            first_pass = prefilter_regex(text)
            if first_pass is True:
                return first_pass
            try:
                res = await classify_fn(md.title, md.first_author, text)
                return res.is_nhst
            except Exception as e:
                print(f"Error classifying {md.id}: {e}")
                raise e
                return None
            # return first_pass
    
    # with multiprocess.Pool(3) as p:
    #     res = list(p.starmap(pvalue_prefilter, [(md, Path("data/mds") / f"{form_path_base(md)}.md") for md in mds_successful]))
    
    res = await asyncio.gather(*[pvalue_prefilter(md, Path("data/mds") / f"{form_path_base(md)}.md") for md in mds_successful])
    
    mds_with_filter_status = [md for (md, status) in zip(mds_with_pdfs, res) if status is not None]
    
    # print([md.title for md in mds])
    
    
    print(len(mds_with_filter_status))
    print(res)
    # print([status for (md, status) in zip(mds_with_filter_status, res)])
    
    # print(len(filtered_dois))
    
    # print(len(filtered_dois))
    
asyncio.run(work())
# fetch_from_library_lol(dois[0])