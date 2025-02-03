from pathlib import Path
from statextract.typedefs import PaperMD

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

converter = PdfConverter(
    artifact_dict=create_model_dict(),
)
rendered = converter("FILEPATH")
text, _, images = text_from_rendered(rendered)


def process_pdf_marker(md: PaperMD, pdf_path: Path, md_output_path: Path, image_output_path_prefix: Path) -> None:
    pass