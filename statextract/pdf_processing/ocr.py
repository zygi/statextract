
import asyncio
import logging
import pathlib
import re
from typing import Any
from anthropic import AsyncAnthropic
import anthropic.types as atypes
import pyalex

from statextract.agent.agent import anthropic_call_tool
from statextract.helpers import collect_model_images, collect_model_text, form_path_base
from statextract.md_retriever import parse_work
from statextract.typedefs import PaperMD

claude_prompt = """You are given a PDF file of a research paper. Your goal is to extract the text from the PDF given to you as images.
The name of the article is "{article_name}" by {authors}. You should only return the text of the article, not the rest of the document. Specifically, if this is a scan of journal pages, you should not include the page numbers or other metadata, or any article fragments before or after the main article.

Your full output should be surrounded by tags <article> and </article>.

You should return the text in natural reading order. For any tables or other formatted but textual content, you should return it as html:

<table name="Table 1">
<tr>
<td>...</td>
</tr>
...
</table>

For figures that are primarily visual, you should output a figure tag and give a short description. If the figure contains text, you should transcribe it. For example, if the figure is a graph with text labels and p-values, you should output:

<figure name="Figure 1">
<description>Graph of ...</description>
<text_literals>
<text> Control </text>
<text> 34 </text>
<text> Treatment </text>
<text> 55 </text>
<text> p = 0.045 </text>
</text_literals>
</figure> 

______


To help you, we have provided you the machine-extracted text of the article. Remember that it might be badly formatted or otherwise inaccurate. Your goal is to provide a more accurate and better structured transcription. So remember, DO NOT blindly copy newlines. DO NOT drop figures, tables or text just because they are not in the machine-extracted text.

<machine_ocr_input>
{machine_ocr_input}
</machine_ocr_input>

______

Ok, you can begin.
"""

async def do_ocr(anthropic_client: AsyncAnthropic, md: PaperMD, max_images: int = 20, max_tokens_per_round=1024) -> Any:
    imgs = collect_model_images(md)
    if len(imgs) == 0:
        return ""
    if len(imgs) > max_images:
        raise RuntimeError(f"Too many images for {md.id}: {len(imgs)}")
    
    text = collect_model_text(md)
    
    prompt = claude_prompt.format(article_name=md.title, authors=', '.join(md.author_names), machine_ocr_input=text)
    
    # stitching the responses together
    
    debug_individual_responses = []
    full_response = []
    
    msgs = [
        {"role": "user", "content": [atypes.ImageBlockParam(type="image", source={"data": img[0], "media_type": "image/jpeg", "type": "base64"}) for img in imgs] + [atypes.TextBlockParam(text=prompt, type="text", cache_control={"type": "ephemeral"}), ]},
        {"role": "assistant", "content": "<article>"}
    ]
    
    while True:
        responses, _ = await anthropic_call_tool(anthropic_client, 
                                        tools=[], 
                                        #  [], 
                                        system_prompt="You are an image transcriber.", 
                                        # model="claude-3-haiku-20240307", 
                                        model="claude-3-5-sonnet-latest", 
                                        max_tokens=4096,
                                        uncached_init_messages=msgs,
                                        init_messages=[])
        assert len(responses) == 1, f"Expected 1 response, got {len(responses)}: {responses}"
        response = responses[0]
        debug_individual_responses.append(response)
        
        msgs[-1]["content"] = msgs[-1]["content"] + response.content[0].text
        
        if response.stop_reason == "stop_sequence" or response.stop_reason == "end_turn":
            break
        if len(debug_individual_responses) > 10:
            raise RuntimeError("Too many responses")
        
        # print(response)
        
        
    return msgs[-1]["content"], debug_individual_responses
    
    
if __name__ == "__main__":
    import hishel
    # from statextract.helpers import TSAI
    from rich import print
    paper_md = parse_work(pyalex.Works()["https://doi.org/10.1152/ajpgi.1987.253.5.g601"])
    
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("hishel.controller").setLevel(logging.DEBUG)

    hc = hishel.Controller(
        cacheable_methods=["GET", "POST"],
        allow_stale=True,
        always_revalidate=False,
        force_cache=True,
    )
    storage = hishel.AsyncFileStorage()
    client = hishel.AsyncCacheClient(controller=hc, storage=storage)
    
    async def main():
        res, _ = await do_ocr(AsyncAnthropic(http_client=client), paper_md)
        
        pdf_text = collect_model_text(paper_md)
    
        
        NUMBER_REGEX = re.compile(r"[\d\.]+")
        
        # find all long number matches
        long_numbers_in_res = set([m for m in NUMBER_REGEX.findall(res) if len(m) > 4])
        long_numbers_in_gold = set([m for m in NUMBER_REGEX.findall(pdf_text) if len(m) > 4])
        
        # print(long_numbers_in_res)
        # print(long_numbers_in_gold)
        
        print(res)
        
        # print(long_numbers_in_res - long_numbers_in_gold)
        print(long_numbers_in_gold - long_numbers_in_res)
        
        pdf_path = pathlib.Path("data/pdfs") / f"{form_path_base(paper_md)}.pdf"
        print(pdf_path)
        
        # print(res)
    
    asyncio.run(main())