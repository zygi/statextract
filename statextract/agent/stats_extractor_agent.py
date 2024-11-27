import asyncio
import json
import subprocess
import sys
import typing
import pyalex
from typing_extensions import final
import pydantic

from statextract.agent.agent import anthropic_call_tool
from statextract.agent.tools import RExecInput, RExecTool, Tool
from statextract.cache_typed import Cache
from statextract.helpers import UsageCounter, collect_model_inputs, form_path_base
from statextract.md_retriever import parse_work
from statextract.typedefs import PaperMD
from anthropic import AsyncAnthropic, types as atypes

from rich import print


class ClaimSummary(pydantic.BaseModel):
    """
    A core claim of the paper that is supported by quantitative analysis and a statistical test.
    """

    summary: str = pydantic.Field(description="Claim summary.")
    position_in_text: str = pydantic.Field(
        description="The position of the claim in the text, described verbally."
    )


class Claims(pydantic.BaseModel):
    """
    A collection of core claims of the paper that are supported by quantitative analysis and a statistical test.
    Please only choose the most important claims of the paper. They should be 1) mentioned in the abstract, AND 2) supported by some kind of statistical test.

    Output no more than 3 claims.
    """

    chain_of_thought: str = pydantic.Field(
        description="The chain of thought you can use for reasoning."
    )
    claims: list[ClaimSummary] = pydantic.Field(
        description="The list of claims. No more than 3."
    )

class AbortRequest(pydantic.BaseModel):
    reason: str = pydantic.Field(
        description="The reason why the task is impossible to complete to complete and should be aborted."
    )
    
class AnswerQuestionTool[T: pydantic.BaseModel](Tool[T, None]):
    def __init__(self, cb: typing.Callable[[T], typing.Awaitable[None]], TT: typing.Type[T]):
        self.cb = cb
        super().__init__(TT, type(None))
    
    def set_callback(self, cb: typing.Callable[[T], typing.Awaitable[None]]):
        self.cb = cb
    
        
@final
class AbortTool(AnswerQuestionTool[AbortRequest]):
    def __init__(self, cb: typing.Callable[[AbortRequest], typing.Awaitable[None]]):
        super().__init__(cb, AbortRequest)

    @property
    def name(self) -> str:
        return "Abort"
    
    @property
    def description(self) -> str:
        return "If it is impossible to complete the task, call this tool and include the reason why."
    
    async def execute(self, request: AbortRequest) -> None:
        await self.cb(request)
        

# class ExactPValue(pydantic.BaseModel):
#     tag: typing.Literal["exact"] = "exact"
#     value: float = pydantic.Field(
#         description="The exact p-value of the claim: p = <value>"
#     )

# class PValueBound(pydantic.BaseModel):
#     tag: typing.Literal["bound"] = "bound"
#     value: str = pydantic.Field(
#         description="The bound on the p-value of the claim: p < <value>"
#     )
    
# class RCalculation(pydantic.BaseModel):
#     tag: typing.Literal["calculation"] = "calculation"
#     code: str = pydantic.Field(
#         description="The calculation should be a valid R script that can be used to compute the p-value. It should take no external input - you should enter the numbers inline. Your script's final line should assign the p-value to a variable named `pvalue`."
#     )


class DetailedClaimRequest(pydantic.BaseModel):
    chain_of_thought_md: str = pydantic.Field(
        description="First, you can choose to think through your answer here. You can use markdown formatting."
    )
    claim_summary: str = pydantic.Field(
        description="The summary of the claim being made."
    )
    statistical_support_summary_md: str = pydantic.Field(
        description="The summary of the statistical tests that support the claim, as well as the key test statistics and values. Specifically mention which statistical test was used by saying 'Statistical test: <test name>'. If you can't determine the test name, output 'Statistical test: UNKNOWN'. You can use markdown formatting."
    )
    
    statistical_support_page_number: int = pydantic.Field(
        description="The page number of the article where the statistical support is located. This should be a number between 1 and the total number of pages (provided images) in the article."
    )
    
    statistical_support_bounding_box: tuple[int, int, int, int] = pydantic.Field(
        description="The bounding box of the statistical support in the article, contained in the page provided in `statistical_support_page_number`. The box should be a tight fitting box around the text of the claim, with 4 numbers, in the format (x1, y1, x2, y2). The box should be in the coordinate system of the article, with the top left corner being (0, 0) and the bottom right corner being (width, height). Use the rulers for reference. BE VERY SPECIFIC AND ONLY INCLUDE THE SUPPORTING SENTENCES, NOT HUGE BLOCKS OF TEXT."
    )

#     p_value: typing.Annotated[ExactPValue | RCalculation | PValueBound, pydantic.Field(discriminator="tag")] = pydantic.Field(
#         description="""\
# The extracted p-value of the claim. If the paper includes an exact p-value, prioritize that and output an ExactPValue object. If the paper does not report an exact p-value, but does report test statistics that can be used to calculate it, output a RCalculation object with an R script that computes it. Otherwise, as a last resort, if the paper reports a bound on the p-value, output a PValueBound object."""
#     )


    p_value_exact: typing.Optional[float] = pydantic.Field(
        description="If the paper reports an exact p-value for the claim (p = <value>), output it here. Otherwise, if it reports a bound like p < <value> or simply doesn't mention it, output None."
    )

    p_value_computed_R_script: typing.Optional[str] = pydantic.Field(
        description="If the paper does NOT report an exact p-value, but does report test statistics that can be used to calculate it AND SPECIFICALLY MENTIONS WHICH STATISTICAL TEST WAS USED, output the calculation here. The calculation should be a valid R script that can be used to compute the p-value. It should take no external input - you should enter the numbers inline. Your script's final line should assign the p-value to a variable named `pvalue`. FOLLOW THE ARTICLE'S METHOD AS DESCRIBED AND DON'T MAKE ASSUMPTIONS. If you need to guess the statistical test, output null instead of guessing."
    )
    
    
    p_value_bound: typing.Optional[float] = pydantic.Field(
        description="As a last resort, if you cannot find the p-value in the article, and can't compute it from the test statistics, but the article reports a bount like p < <value>, return it here."
    )

    # finding_p_impossible_reason: typing.Optional[str] = pydantic.Field(
    #     description="If the paper does NOT report any test statistics that can be used to calculate the p-value, fill this field with the reason why. Otherwise, if any of the fields above are non-empty, output None."
    # )


@final
class ClaimListingTool(AnswerQuestionTool[Claims]):
    def __init__(self, cb: typing.Callable[[Claims], typing.Awaitable[None]]):
        self.cb = cb
        super().__init__(cb, Claims)

    @property
    def name(self) -> str:
        return "ListClaims"

    @property
    def description(self) -> str:
        return "List the core claims of the paper that are supported by quantitative analysis and a statistical test. Please only choose the most important claims of the paper. They should be 1) mentioned in the abstract, AND 2) supported by some kind of statistical test. Output no more than 3 claims.\nIf the paper does not perform null hypothesis significance testing, or if it's just an abstract without any quantitative analysis, or if there's something else wrong with the paper, output an empty list and output True for the `abort` field. Otherwise, output False for `abort`."

    async def execute(self, request: Claims) -> None:
        # if len(request.claims)
        await self.cb(request)


@final
class DetailedClaimTool(Tool[DetailedClaimRequest, None]):
    def __init__(
        self,
        cb: typing.Callable[
            [DetailedClaimRequest, typing.Optional[float]], typing.Awaitable[None]
        ],
    ):
        self.timeout = 10
        self.cb = cb
        super().__init__(DetailedClaimRequest, type(None))
        
    def set_callback(self, cb: typing.Callable[
            [DetailedClaimRequest, typing.Optional[float]], typing.Awaitable[None]
        ]):
        self.cb = cb

    @property
    def name(self) -> str:
        return "DetailedClaim"

    @property
    def description(self) -> str:
        return "This tool allows you to execute R scripts."

    async def execute(self, input: DetailedClaimRequest) -> None:
        # if input.finding_p_impossible_reason is not None:
        #     await self.cb(input, None)
        #     return
        if input.p_value_exact is not None:
            await self.cb(input, input.p_value_exact)
            return
        # otherwise, we need to calculate it
        if input.p_value_computed_R_script is not None:
            # raise Exception(
            #     "DetailedClaimTool received a DetailedClaimRequest with `finding_p_impossible` False and `exact_p_value` None and `computed_p_R_source` None. One of the fields should be non-empty - please debug."
            # )

            source_with_print = input.p_value_computed_R_script + "\nprint(pvalue)"

            process = await asyncio.create_subprocess_exec(
                "timeout",
                str(self.timeout),
                "Rscript",
                "-e",
                source_with_print,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                raise Exception(
                    f"R script failed with return code {process.returncode}:\n{stderr.decode()}"
                )
            # parse the output
            res = [l for l in stdout.decode().split("\n") if l.startswith("[1]")]
            last_print = res[-1]
            pvalue = float(last_print.split(" ")[-1])
            await self.cb(input, pvalue)
            return
        
        if input.p_value_bound is not None:
            await self.cb(input, input.p_value_bound)
            return

        raise Exception(
                "DetailedClaimTool received a DetailedClaimRequest with none of the p-value fields filled. Please fix the answer."
            )


# class DetailedClaim(pydantic.BaseModel):
#     """
#     A detailed claim that is supported by quantitative analysis and a statistical test.
#     """
#     summary: str = pydantic.Field(description="Claim summary.")

#     original_test_statistics: list[str] = pydantic.Field(..., description="""\
# The test statistics found in the paper that support the claim. Each string should be in one of the following formats:
# `F(_, _) = _`
# `t(_) = _`
# `chi2(_) = _`
# `r(_) = _`
# `z = _`
# `p = _`
# `p < _`
# where _ is a numeric value. Each test statistic should be in a separate string. Every string should be formatted exactly as shown above. Only include the statistics from the paper, not calculations you made yourself.""")

#     nearby_citation: str = pydantic.Field(..., description="A short exact substring of text (not numbers or equations) that is close or at the test statistics in the text. This will be used to locate the region where the claim exists so copy it EXACTLY from the paper contents, including formatting, any mistyping, etc.")

#     final_p_value: typing.Optional[float] = pydantic.Field(..., description="The final exact p-value of the claim. This field requires a precise value, NOT a bound like `p < 0.05`. If the paper reports such an exact p-value, like `p = 0.0317`, output it. If the paper reports other test statistics that let you calculate a p-value, output that p-value after calculating it exactly. If the paper only reports a bound, and doesn't give a way to calculate an exact p-value, leave the field empty.")

class Holder[T]:
    _value: T | None
    
    def __init__(self):
        self._value = None

    @property
    def value(self) -> T | None:
        return self._value
    
    @value.setter
    def value(self, value: T):
        self._value = value

    # def set(self, value: T):
    #     self._value = value
    
async def _process_file_claim(claim: ClaimSummary, client: AsyncAnthropic, tools: list[Tool], init_messages: list[atypes.MessageParam], usage_counter: UsageCounter) -> tuple[tuple[DetailedClaimRequest, float | None] | None, AbortRequest | None]:
    claim_res: Holder[tuple[DetailedClaimRequest, float | None]] = Holder()
    abort_result: Holder[AbortRequest] = Holder()
    
    async def detailed_claim_cb(
        request: DetailedClaimRequest, pvalue: typing.Optional[float]
    ):
        claim_res.value = (request, pvalue)
        
    async def abort_cb(request: AbortRequest):
        abort_result.value = request
        
    detailed_claim_tool = [t for t in tools if isinstance(t, DetailedClaimTool)][0]
    detailed_claim_tool.set_callback(detailed_claim_cb)
    abort_tool = [t for t in tools if isinstance(t, AbortTool)][0]
    abort_tool.set_callback(abort_cb)

    async def should_stop(i: int, msgs: list[atypes.MessageParam]) -> bool:
        return claim_res.value is not None or abort_result.value is not None

    task_message: atypes.MessageParam = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "Your goal, given the previous article, is to look into one specific claim and extract the statistical support for it. Your goal is to provide an exact p-value for the claim, if possible, or to provide a calculation that can be used to get an exact p-value if the paper does not provide one. The claim you are looking into is:\n<claim>\n" + claim.summary + "\n</claim>.\n\nIf you can provide the answer, submit it using the DetailedClaim tool. If you cannot provide the answer, call the Abort tool."
            }
        ],
    }

    msgs, calls = await anthropic_call_tool(
        client,
        tools,
        init_messages,
        "You are a helpful scientific paper reviewer. You are an expert in statistical techniques and hypothesis testing.",
        must_call=True,
        should_stop=should_stop,
        uncached_init_messages=[task_message],
    ) 
    
    for msg in msgs:
        usage_counter.add_cache_creation_input_tokens(msg.usage)
    
    return claim_res.value, abort_result.value
    
def _build_tools_init_messages(md: PaperMD) -> tuple[list[Tool], list[atypes.MessageParam]]:
    text, pics_with_sizes = collect_model_inputs(md)
    pics, sizes = zip(*pics_with_sizes)
    
    size_text = "<image_sizes>\n" + "\n".join([f"Page {i+1}: {w}x{h}" for i, (w, h) in enumerate(sizes)]) + "\n</image_sizes>"
    
    text_block: atypes.TextBlockParam = {
        "type": "text",
        "text": f"The following is the article {md.title} by {', '.join(md.author_names)}. You will need to perform the task described below.\n\n<article_text>"
        + text
        + "</article_text>\n\nFurthermore, you're given the pages of the article as images. The images are augmented with rulers to make it easier to describe positions of objects. Full image sizes are as follows:\n"
        + size_text
        +"\n\nI'll give you the task in the next message. Are you ready?",
    }
    # text_block: atypes.TextBlockParam = {
    #     "type": "text",
    #     "text": f"The following is the article {md.title} by {', '.join(md.author_names)}, given to you as a series of images, one per page, in order. The images are augmented with rulers to make it easier to describe positions of objects. Full image sizes are as follows:\n"
    #     + size_text
    #     +"\n\n Using this article, you will need to perform the task described below. I'll give you the task in the next message. Are you ready?",
    # }

    image_blocks: list[atypes.ImageBlockParam] = [
        {
            "type": "image",
            "source": {"data": pic, "media_type": "image/jpeg", "type": "base64"},
        }
        for pic in pics
    ]

    assistant_reply: atypes.MessageParam = {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Yes, I'm ready.",
            }
        ],
    }

    initial_message: atypes.MessageParam = {
        "role": "user",
        "content": [text_block] + image_blocks,
    }
    
    async def noop_cb_detailed(request: DetailedClaimRequest, pvalue: typing.Optional[float]):
        pass
    
    async def noop_cb_abort(request: AbortRequest):
        pass
    
    async def noop_cb_claims(request: Claims):
        pass
        
    claim_listing_tool = ClaimListingTool(noop_cb_claims)
    detailed_claim_tool = DetailedClaimTool(noop_cb_detailed)
    abort_tool = AbortTool(noop_cb_abort)
    
    return [claim_listing_tool, detailed_claim_tool, abort_tool], [initial_message, assistant_reply]

class WrappedAbort(AbortRequest):
    tag: typing.Literal["abort"] = "abort"
    
    @classmethod
    def from_abort(cls, abort: AbortRequest):
        return cls(reason=abort.reason)

    # def __init__(self, abort: AbortRequest):
    #     super().__init__(reason=abort.reason)
        
# class WrappedD

class ClaimDetailsSuccess(pydantic.BaseModel):
    tag: typing.Literal["claim_details_success"] = "claim_details_success"
    claim: DetailedClaimRequest
    pvalue: float | None

class FullExtractionResultSuccess(pydantic.BaseModel):
    tag: typing.Literal["success"] = "success"
    claims: Claims
    detailed_claim_results: list[typing.Annotated[ClaimDetailsSuccess | WrappedAbort, pydantic.Field(discriminator="tag")] | None]

class FullExtractionResult(pydantic.BaseModel):
    inner: typing.Annotated[FullExtractionResultSuccess | WrappedAbort, pydantic.Field(discriminator="tag")]

async def process_file(client: AsyncAnthropic, md: PaperMD, usage_counter: UsageCounter) -> FullExtractionResult:
    tools, init_messages = _build_tools_init_messages(md)

    claim_list: Holder[Claims] = Holder()
    abort_result: Holder[AbortRequest] = Holder()

    async def list_claims_cb(claims: Claims):
        claim_list.value = claims
        
    async def abort_cb(request: AbortRequest):
        abort_result.value = request
    
    async def detailed_claim_cb(request: DetailedClaimRequest, pvalue: typing.Optional[float]):
        pass
    
    claim_listing_tool = [t for t in tools if isinstance(t, ClaimListingTool)][0]
    detailed_claim_tool = [t for t in tools if isinstance(t, DetailedClaimTool)][0]
    abort_tool = [t for t in tools if isinstance(t, AbortTool)][0]
    
    claim_listing_tool.set_callback(list_claims_cb) 
    detailed_claim_tool.set_callback(detailed_claim_cb)
    abort_tool.set_callback(abort_cb)

    task_message: atypes.MessageParam = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Your goal, given the previous article, is to list the core claims of the paper that are supported by quantitative analysis and a statistical test. Please only choose the most important claims of the paper. They should be 1) mentioned in the abstract, AND 2) supported by some kind of statistical test. Output no more than 3 claims by using the ListClaims tool.\n\nIf the paper does not perform null hypothesis significance testing, or if it's just an abstract without any quantitative analysis, please use the Abort tool.",
            }
        ],
    }

    # first, run the claim listing tool
    async def should_stop(i: int, msgs: list[atypes.MessageParam]) -> bool:
        return claim_list.value is not None or abort_result.value is not None

    msgs, calls = await anthropic_call_tool(
        client,
        tools,
        init_messages,
        "You are a helpful scientific paper reviewer. You are an expert in statistical techniques and hypothesis testing.",
        must_call=True,
        should_stop=should_stop,
        uncached_init_messages=[task_message],
    )
    
    for msg in msgs:
        usage_counter.add_cache_creation_input_tokens(msg.usage)
    
    if abort_result.value is not None:
        return FullExtractionResult(inner=WrappedAbort.from_abort(abort_result.value))
    
    assert claim_list.value is not None, "Claim listing tool did not return a result"
    
    detailed_claim_results: list[ClaimDetailsSuccess | WrappedAbort | None] = []
    
    for claim in claim_list.value.claims:
        claim_res, abort_res = await _process_file_claim(claim, client, tools, init_messages, usage_counter)
        
        # if abort_res is not None:
        #     return (claim_list.value, None, detailed_claim_results, abort_res)
        if abort_res is not None:
            detailed_claim_results.append(WrappedAbort.from_abort(abort_res))
        # assert claim_res.value is not None
        elif claim_res is not None:
            detailed_claim_results.append(ClaimDetailsSuccess(claim=claim_res[0], pvalue=claim_res[1]))
    
    return FullExtractionResult(inner=FullExtractionResultSuccess(claims=claim_list.value, detailed_claim_results=detailed_claim_results))

        
    #     print(claim_res.value)

    # print(claim_list)
    # print(detailed_claim_results)

# def show_html(html: str):
#     from PySide6.QtCore import QUrl
#     from PySide6.QtWidgets import QApplication, QHBoxLayout, QLineEdit
#     from PySide6.QtWidgets import QMainWindow, QPushButton, QVBoxLayout
#     from PySide6.QtWidgets import QWidget
#     from PySide6.QtWebEngineWidgets import QWebEngineView
#     app = QApplication()

#     # first, just create a test webview
#     webview = QWebEngineView()
#     layout = QVBoxLayout()
#     layout.addWidget(webview)

#     container = QWidget()
#     container.setLayout(layout)

#     window = QMainWindow()
#     window.setCentralWidget(container)
#     window.setWindowTitle("Full Extraction Result")
#     window.resize(1920, 1080)  # Set default width to 1920 and height to 1080
#     webview.setHtml(html)
#     window.show()
#     sys.exit(app.exec())
    
def show_html_in_native_browser(html: str):
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as temp_file:
        print(temp_file.name)
        temp_file.write(html.encode())
        temp_file_path = temp_file.name

    subprocess.run(["/mnt/c/Windows/explorer.exe", f"\\\\wsl.localhost\\Ubuntu-22.04{temp_file_path.replace('/', '\\')}"], env={"BROWSER": "/mnt/c/Windows/explorer.exe"})
    
def represent_full_extraction(md: PaperMD, res: FullExtractionResult):  
    print(f"PDF path: {sys.path[0]}/data/pdfs/{form_path_base(md)}.pdf")
    text, pics_with_sizes = collect_model_inputs(md)
    # pics, sizes = zip(*pics_with_sizes)
    
    html = """
    <html>
        <head><title>Extraction Result</title>
        <script type="module" src="https://md-block.verou.me/md-block.js"></script>

        </head>
        <body>{INNER}
        </body></html>
    """
    
    if res.inner.tag == "abort":
        msg = f"The extraction was aborted because {res.inner.reason}"
        return show_html_in_native_browser(html.format(INNER=msg))
    
    INNER = ""
    
    for claim, detailed_claim in zip(res.inner.claims.claims, res.inner.detailed_claim_results, strict=True):
        INNER += f"<h2>Claim: {claim.summary}</h3>"
        INNER += f"<p>Position in text: {claim.position_in_text}</p>"

        if isinstance(detailed_claim, tuple):
            request, pvalue = detailed_claim
            INNER += f"<h3>Detailed Claim: {request.claim_summary}</h4>"
            INNER += f"<p><b>Chain of thought:</b> <md-block>{request.chain_of_thought_md}</md-block></p>"
            INNER += f"<p><b>Stats explanation:</b> <md-block>{request.statistical_support_summary_md}</md-block></p>"
            INNER += f"<p><b>R Source:</b> <pre>{request.p_value_computed_R_script}</pre></p>" if request.p_value_computed_R_script is not None else ""
            INNER += f"<p><b>P-value:</b> {f'p < {request.p_value_bound}' if request.p_value_bound is not None else f'p = {pvalue}'} </p>"
            INNER += f"<p><b>Bounding Box:</b> {request.statistical_support_bounding_box}</p>"
            INNER += f"<p><b>Page Number:</b> {request.statistical_support_page_number}</p>"
            
            image_data, image_size = pics_with_sizes[request.statistical_support_page_number-1]
            # image_html
            image_html = f"<img src='data:image/jpeg;base64,{image_data}' width='{image_size[0]}' height='{image_size[1]}' />"
            
            bounding_box = request.statistical_support_bounding_box
            assert len(bounding_box) == 4, f"Bounding box should have 4 elements, got {bounding_box}"
            # draw a red box across the image
            # Construct a bounding box on the image
            box_html = f"""
            <div style="
                position: absolute;
                left: {bounding_box[0]}px;
                top: {bounding_box[1]}px;
                width: {bounding_box[2] - bounding_box[0]}px;
                height: {bounding_box[3] - bounding_box[1]}px;
                border: 2px solid red;
                pointer-events: none;
            "></div>
            """
            
            # Wrap the image and bounding box in a container
            image_html = f"""
            <div style="position: relative; display: inline-block;">
                {image_html}
                {box_html}
            </div>
            """
            
            
            INNER += f"<p>{image_html}</p>"
        else:
            assert isinstance(detailed_claim, WrappedAbort)
            INNER += f"<p>Abort: {detailed_claim.reason}</p>"
        
    return show_html_in_native_browser(html.format(INNER=INNER))
        
    

    
    
    # text, pics_with_sizes = collect_model_inputs(md)
    # pics, sizes = zip(*pics_with_sizes)
    
cc = Cache()
@cc()
async def cached_run(id: str):
    import pyalex

    work = pyalex.Works()[id]
    md = parse_work(work)
    
    # print(AbortRequest.model_json_schema())
    # exit()

    client = AsyncAnthropic()
    uc = UsageCounter()
    return (await process_file(client, md, uc), uc)
    
    
if __name__ == "__main__":
    # tool = RExecTool()

    # # test R script to do a simple t-test analysis
    # TEST = """
    # print('asdf')
    # pvalue <- t.test(rnorm(100), rnorm(100, mean = 1))$p.value
    # print(pvalue)
    # """

    # import pyalex

    # work = pyalex.Works()["W2009995022"]
    # md = convert_doi(work)
    
    # # print(AbortRequest.model_json_schema())
    # # exit()

    # client = AsyncAnthropic()
    # uc = UsageCounter()
    # async def test_tool():
    #     print(await process_file(client, md, uc))
    #     print(uc)
    
    # print(DetailedClaimRequest.model_json_schema())
    
    test_id = "W2009995022"
    
    async def run_test():
        res, uc = await cached_run(test_id)
        print(res)
        print(uc)
        represent_full_extraction(parse_work(pyalex.Works()[test_id]), res)
    asyncio.run(run_test())

    # asyncio.run(test_tool())
