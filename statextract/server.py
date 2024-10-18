import asyncio
import os
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import pydantic

from statextract import storage
from statextract.agent.stats_extractor_agent import FullExtractionResult, cached_run
from statextract.helpers import add_image_ruler_overlay, collect_model_inputs, fetch_author, form_path_base, fetch_work, img_to_bytes, md_image_paths
from statextract.md_retriever import get_all_mds, parse_work
from PIL import Image

from statextract.typedefs import PaperMD

app = FastAPI()
origins = [
    # "http://localhost",
    # "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/image/{work_id}/{page_num}")
async def get_image(work_id: str, page_num: int) -> Response:
    obj = parse_work(fetch_work(work_id))
    path_base = form_path_base(obj)
    img_path = f"data/images/{path_base}-{page_num}.jpg"
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        raise HTTPException(status_code=404, detail="Image not found")

    modified_img = add_image_ruler_overlay(Image.open(img_path))
    return Response(content=img_to_bytes(modified_img), media_type="image/jpeg")


AUTHOR_WHITELIST = [
    'A5072310807'
]

@app.get("/authors")
async def get_authors() -> list[tuple[str, str, int]]:
    author_objects = [fetch_author(id) for id in AUTHOR_WHITELIST]
    
    db_instance = storage.full_extraction_result_db()
    
    async def get_author_checked_work_count(author_id: str) -> int:
        mds = get_all_mds(author_id)
        async def check_md(md: PaperMD) -> bool:
            return await db_instance.get(md.id) is not None
        reses = await asyncio.gather(*[check_md(md) for md in mds])
        return sum(reses)
        
    work_counts = await asyncio.gather(*[get_author_checked_work_count(author_id) for author_id in AUTHOR_WHITELIST])
    
    return [
        (author['display_name'], author_id, work_count)
        for (author_id, author, work_count) in zip(AUTHOR_WHITELIST, author_objects, work_counts)
    ]
    
class AuthorResponse(pydantic.BaseModel):
    name: str
    id: str
    works: list[tuple[PaperMD, FullExtractionResult | None]]
    
@app.get("/author/{author_id}")
async def get_author(author_id: str, article_only: bool = True, first_author_only: bool = True) -> AuthorResponse:
    author = fetch_author(author_id)
    
    db_instance = storage.full_extraction_result_db()
    mds = get_all_mds(author_id, first_author=first_author_only, only_articles=article_only)
    md_statuses = await asyncio.gather(*[db_instance.get(md.id) for md in mds])
    
    
    return AuthorResponse(
        name=author['display_name'], # type: ignore
        id=author['id'], # type: ignore
        works=[(md.drop_full(), status) for (md, status) in zip(mds, md_statuses)]
    )

class WorkResponse(pydantic.BaseModel):
    md: PaperMD
    images: list[str]
    full_response: FullExtractionResult

@app.get("/work/{work_id}")
async def get_work(work_id: str) -> WorkResponse:
    # test_id = "W2009995022"
    db_instance = storage.full_extraction_result_db()
    print(await db_instance.get_keys())
    res = await db_instance.get('https://openalex.org/'+work_id)
    if res is None:
        raise HTTPException(status_code=404, detail="Work not found")

   
    # res, uc = await cached_run(test_id)
    
    md = parse_work(fetch_work(work_id))
    
    imgs = [img.name for img in md_image_paths(md)]
    
    return WorkResponse(md=md, images=imgs, full_response=res)
    
    
    