import itertools
import os
from pathlib import Path
import shutil
from typing import Any
from joblib import Memory
import pyalex
import pydantic
import httpx

GINA = "A5000930617"
DWECK = "A5046821941"
ZLOKOVIC = "A5036566811"
TSAI = "A5042440695"


@Memory('./cache').cache
def author_works(author_id: str):
    # paginate
    
    works = pyalex.Works().filter(author={"id": author_id}).paginate(per_page=100)
    return list(itertools.chain(*works))

@Memory('./cache/authors').cache
def author(author_id: str):
    return pyalex.Authors()[author_id]


def assert_str(s: str):
    if not isinstance(s, str):
        raise ValueError("Expected a string")
    return s

class PaperMD(pydantic.BaseModel):
    title: str
    author_names: list[str]
    first_author: str
    doi: tuple[str, str]

def convert_doi(work: Any):
    # print(work)
    doi = work['doi']
    if doi is None:
        return None
    parts = doi.split('/')[-2:]
    if len(parts) != 2:
        raise ValueError("Invalid DOI format")

    title = work['title']
    author_names = [author['author']['display_name'] for author in work['authorships']]
    first_author = author_names[0]
    return PaperMD(title=title, author_names=author_names, first_author=first_author, doi=parts)


def get_mds_with_dois(author_id: str, first_author: bool = True):
    works = author_works(author_id)
    # filter to works where the author is the first author
    if first_author:
        works = [work for work in works if work['authorships'][0]['author']['id'].split('/')[-1] == author_id]
    
    mds = [convert_doi(work) for work in works if 'doi' in work]
    mds = [md for md in mds if md is not None]
    return mds

def filter_mds_by_pdf(mds: list[PaperMD]):
    def has_pdf(md: PaperMD):
        return Path(f"pdfs/{form_path_base(md.doi)}.pdf").exists()
    return [md for md in mds if has_pdf(md)]

def form_path_base(doi: tuple[str, str]):
    library_id, item_id = doi
    return f"{library_id}_{item_id}"

def normalize_to_file_name(s: str):
    return s.replace(" ", "_").replace("/", "_").replace(r"[^a-zA-Z0-9_]", "")

import sys

def update_links(author_id: str, first_author: bool = True, pdf_author_link_path: Path = Path("data/pdf_author_links"), pdf_path: Path = Path("data/pdfs")):
    mds = get_mds_with_dois(author_id, first_author)
    mds = filter_mds_by_pdf(mds)
    
    print(mds)
    exit(0)
    
    if not pdf_author_link_path.exists():
        pdf_author_link_path.mkdir()

    author_name = mds[0].first_author
    
    dest = pdf_author_link_path / f"{author_name}"
    if not dest.exists():
        dest.mkdir()

    for md in mds:
        source = pdf_path / f"{form_path_base(md.doi)}.pdf"
        dest = pdf_author_link_path / f"{author_name}" / f"{normalize_to_file_name(md.title)}.pdf"
        if not dest.exists():
            # create the symlink
            os.symlink(source, dest)

# print(get_mds_with_dois("A5000930617"))

if __name__ == "__main__":
    update_links(TSAI)