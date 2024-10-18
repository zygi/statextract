
import itertools
from typing import Any

from joblib import Memory
import pyalex

from statextract import helpers
from statextract.typedefs import FullPaperMD


@Memory('./cache').cache
def author_works(author_id: str):
    # paginate
    
    works = pyalex.Works().filter(author={"id": author_id}).paginate(per_page=100)
    return list(itertools.chain(*works))

def parse_work(work: Any):
    # print(work)
    doi = work['doi']
    if doi is None:
        parts = None
    else:
        parts = doi.split('/')[-2:]
        if len(parts) != 2:
            raise ValueError("Invalid DOI format")

    title = work['title']
    author_names = [author['author']['display_name'] for author in work['authorships']]
    first_author = author_names[0]
    return FullPaperMD(title=title if title else "<UNKNOWN>", author_names=author_names, first_author=first_author, doi=parts, full=work, id=work['id'], type=work['type'])


def get_all_mds(author_id: str, first_author: bool = True, only_articles: bool = True):
    works = author_works(author_id)
    # filter to works where the author is the first author
    if first_author:
        works = [work for work in works if work['authorships'][0]['author']['id'].split('/')[-1] == author_id]
        
    if only_articles:
        works = [work for work in works if work['type'] == 'article']
    
    mds = [parse_work(work) for work in works if 'doi' in work]
    # mds = [md for md in mds if md is not None]
    return mds

if __name__ == "__main__":
    from rich import print

    mds = get_all_mds('A5066976206')
    
    # fetch pdfs
    
    # converted = [convert_doi(md) for md in mds]
    print([md.title for md in mds])