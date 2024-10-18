import base64
import io
import itertools
import os
from pathlib import Path
import shutil
from typing import Any
from joblib import Memory
import pyalex
import pydantic
import httpx
import sys
from PIL import Image, ImageDraw, ImageFont

from statextract.typedefs import FullPaperMD, PaperMD

pyalex.config.email = "hi@zygi.me"
GINA = "A5000930617"
DWECK = "A5046821941"
ZLOKOVIC = "A5036566811"
TSAI = "A5042440695"


@Memory('./cache/authors').cache
def fetch_author(author_id: str):
    return pyalex.Authors()[author_id]


@Memory('./cache/works').cache
def fetch_work(work_id: str):
    return pyalex.Works()[work_id]


def assert_not_none[T](s: T | None) -> T:
    if s is None:
        raise ValueError("Expected a non-none value")
    return s

def assert_str(s: str):
    if not isinstance(s, str):
        raise ValueError("Expected a string")
    return s

@Memory('./cache/all_topics').cache
def all_topics():
    iter = pyalex.Topics().filter().paginate(per_page=100)
    return list(itertools.chain(*iter))

# @Memory('./cache/all_authors').cache
# def all_authors():
#     iter = pyalex.Authors().select(["id", "display_name", "x_concepts", "topics"]).paginate(per_page=200, n_max=100000000000)
#     return list(itertools.chain(*iter))


def filter_mds_by_pdf(mds: list[PaperMD]):
    def has_pdf(md: PaperMD):
        return Path(f"data/pdfs/{form_path_base(md)}.pdf").exists()
    return [md for md in mds if has_pdf(md)]

def form_path_base(md: PaperMD):
    if md.doi is None:
        # remove the prefix
        idd = md.id.replace("https://openalex.org/", "")
        return f"NODOI_{idd}"
    library_id, item_id = md.doi
    return f"{library_id}_{item_id}"

def normalize_to_file_name(s: str):
    return s.replace(" ", "_").replace("/", "_").replace(r"[^a-zA-Z0-9_]", "")


# def update_links(author_id: str, first_author: bool = True, pdf_author_link_path: Path = Path("data/pdf_author_links"), pdf_path: Path = Path("data/pdfs")):
#     mds = get_all_mds(author_id, first_author)
#     # mds = filter_mds_by_pdf(mds)
    
#     print([m.doi for m in mds])
#     exit(0)
    
#     if not pdf_author_link_path.exists():
#         pdf_author_link_path.mkdir()

#     author_name = mds[0].first_author
    
#     dest = pdf_author_link_path / f"{author_name}"
#     if not dest.exists():
#         dest.mkdir()

#     for md in mds:
#         source = pdf_path / f"{form_path_base(md.doi)}.pdf"
#         dest = pdf_author_link_path / f"{author_name}" / f"{normalize_to_file_name(md.title)}.pdf"
#         if not dest.exists():
#             # create the symlink
#             os.symlink(source, dest)

# from PIL import Image


def add_image_ruler_overlay(image: Image.Image, ruler_length: int = 20, font_size: int = 10):
    # draw vertical and horizontal bars every 100 pixels
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Draw vertical lines
    for x in range(0, width, 100):
        draw.line([(x, 0), (x, ruler_length)], fill="red", width=1)

    # Draw horizontal lines
    for y in range(0, height, 100):
        draw.line([(0, y), (ruler_length, y)], fill="red", width=1)

    # Add labels
    font = ImageFont.load_default(size=font_size)
    for x in range(0, width, 100):
        draw.text((x+5, 10), str(x), fill="red", font=font)
    for y in range(0, height, 100):
        draw.text((10, y+5), str(y), fill="red", font=font)

    return image

def img_to_bytes(img: Image.Image):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()

def md_image_paths(md: PaperMD):
    image_files = list(Path("data/images").glob(f"{form_path_base(md)}-*.jpg"))
    image_files.sort(key=lambda x: int(x.stem.split("-")[-1]))
    return image_files

def collect_model_inputs(md: PaperMD, md_path: Path = Path("data/mds"), image_path: Path = Path("data/images")):
    # fetch the text
    md_text = (Path(md_path) / f"{form_path_base(md)}.md").read_text()

    # fetch the images
    # image_path_prefix = image_path / form_path_base(md)
    # find all the images stored as `image_path / f"{form_path_base(md)}-{page.number}.jpg"`
    image_files = md_image_paths(md)
    
    # image bytes
    image_bytes = [img.read_bytes() for img in image_files]
    # image_objs = [Image.open(io.BytesIO(img)) for img in image_bytes]
    # sizes = [img.size for img in image_objs]
    
    # read the images as base64
    image_objs = [Image.open(io.BytesIO(img)) for img in image_bytes]
    images_transformed = [add_image_ruler_overlay(img) for img in image_objs]

    image_data = [(base64.b64encode(img_to_bytes(img)).decode(), img.size) for img in images_transformed]

    return md_text, image_data
    
# print(get_mds_with_dois("A5000930617"))

# ('10.1152', 'ajpgi.1987.253.5.g601')
if __name__ == "__main__":
    # update_links(TSAI)
    # import pyalex
    # work = pyalex.Works()["https://doi.org/10.1152/ajpgi.1987.253.5.g601"]
    # md = convert_doi(work)
    # print(collect_model_inputs(md))
    # at = all_authors()
    # print(len(at))
    # print(at[0])
    from rich import print
    print(fetch_author(GINA))