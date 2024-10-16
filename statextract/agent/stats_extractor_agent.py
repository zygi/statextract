import asyncio
import typing
from typing_extensions import final
import pydantic

from statextract.agent.agent import anthropic_call_tool
from statextract.agent.tools import RExecInput, RExecTool, Tool
from statextract.helpers import collect_model_inputs
from statextract.md_retriever import convert_doi
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
    If the paper doesn't perform null hypothesis significance testing, output an empty list.
    """

    chain_of_thought: str = pydantic.Field(
        description="The chain of thought you can use for reasoning."
    )
    claims: list[ClaimSummary] = pydantic.Field(
        description="The list of claims. No more than 3."
    )

    abort: bool = pydantic.Field(
        description="If the paper does not perform null hypothesis significance testing, output True and leave `claims` empty. Otherwise, output False."
    )


class AbortRequest(pydantic.BaseModel):
    reason: str = pydantic.Field(
        description="The reason why the task is impossible to complete to complete and should be aborted."
    )


class DetailedClaimRequest(pydantic.BaseModel):
    chain_of_thought: str = pydantic.Field(
        description="First, you can choose to think through your answer here."
    )
    claim_summary: str = pydantic.Field(
        description="The summary of the claim being made."
    )
    statistical_support_summary: str = pydantic.Field(
        description="The summary of the statistical tests that support the claim, as well as the key test statistics and values. This should be copied or paraphrased from the text."
    )

    exact_p_value: typing.Optional[float] = pydantic.Field(
        description="If the paper reports an exact p-value for the claim, output it here. Otherwise, if it reports a bound like p < 0.05 or simply doesn't mention it, output None."
    )

    computed_p_R_source: typing.Optional[str] = pydantic.Field(
        description="If the paper does NOT report an exact p-value, but does report test statistics that can be used to calculate it, output the calculation here. The calculation should be a valid R script that can be used to compute the p-value. It should take no external input - you should enter the numbers inline. Your script's final line should assign the p-value to a variable named `pvalue`."
    )

    finding_p_impossible_reason: typing.Optional[str] = pydantic.Field(
        description="If the paper does NOT report any test statistics that can be used to calculate the p-value, fill this field with the reason why. Otherwise, if any of the fields above are non-empty, output None."
    )


@final
class ClaimListingTool(Tool[Claims, None]):
    def __init__(self, cb: typing.Callable[[Claims], typing.Awaitable[None]]):
        self.cb = cb
        super().__init__(Claims, type(None))

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

    @property
    def name(self) -> str:
        return "DetailedClaim"

    @property
    def description(self) -> str:
        return "This tool allows you to execute R scripts."

    async def execute(self, input: DetailedClaimRequest) -> None:
        if input.finding_p_impossible_reason is not None:
            await self.cb(input, None)
            return
        if input.exact_p_value is not None:
            await self.cb(input, input.exact_p_value)
            return

        # otherwise, we need to calculate it
        if input.computed_p_R_source is None:
            raise Exception(
                "DetailedClaimTool received a DetailedClaimRequest with `finding_p_impossible` False and `exact_p_value` None and `computed_p_R_source` None. One of the fields should be non-empty - please debug."
            )

        source_with_print = input.computed_p_R_source + "\nprint(pvalue)"

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


async def process_file(md: PaperMD):
    text, pics = collect_model_inputs(md)
    text_block: atypes.TextBlockParam = {
        "type": "text",
        "text": f"The following is the article {md.title} by {', '.join(md.author_names)}. You will need to perform the task described below.\n\n<article_text>"
        + text
        + "</article_text>",
    }

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
                "text": "Got it, I'm ready to answer your question.",
            }
        ],
    }

    initial_message: atypes.MessageParam = {
        "role": "user",
        "content": [text_block] + image_blocks,
    }

    claim_list: list[Claims] = []

    async def list_claims_cb(claims: Claims):
        claim_list.append(claims)

    detailed_claim_results = []
    
    async def detailed_claim_cb(request: DetailedClaimRequest, pvalue: typing.Optional[float]):
        pass

    tools = [ClaimListingTool(list_claims_cb), DetailedClaimTool(detailed_claim_cb)]

    client = AsyncAnthropic()

    task_message: atypes.MessageParam = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Your goal, given the previous article, is to list the core claims of the paper that are supported by quantitative analysis and a statistical test. Please only choose the most important claims of the paper. They should be 1) mentioned in the abstract, AND 2) supported by some kind of statistical test. Output no more than 3 claims.\nIf the paper does not perform null hypothesis significance testing, or if it's just an abstract without any quantitative analysis, or if there's something else wrong with the paper, output an empty list and output True for the `abort` field. Otherwise, output False for `abort`.",
            }
        ],
    }

    # first, run the claim listing tool
    async def should_stop(i: int, msgs: list[atypes.MessageParam]) -> bool:
        return len(claim_list) > 0

    msgs, calls = await anthropic_call_tool(
        client,
        tools,
        [initial_message, assistant_reply, task_message],
        "You are a helpful scientific paper reviewer. You are an expert in statistical techniques and hypothesis testing.",
        must_call="ListClaims",
        should_stop=should_stop,
    )
    
    # now, run the detailed claim tool
    for claim in claim_list[0].claims:
        
        
        claim_res = []
        
        async def detailed_claim_cb(
            request: DetailedClaimRequest, pvalue: typing.Optional[float]
        ):
            claim_res.append((request, pvalue))
    
        dct = DetailedClaimTool(detailed_claim_cb)
            
        tools = [ClaimListingTool(list_claims_cb), dct]

        async def should_stop(i: int, msgs: list[atypes.MessageParam]) -> bool:
            return len(claim_res) > 0

        task_message: atypes.MessageParam = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Your goal, given the previous article, is to look into one specific claim and extract the statistical support for it. Your goal is to provide an exact p-value for the claim, if possible, or to provide a calculation that can be used to get an exact p-value if the paper does not provide one. The claim you are looking into is:\n<claim>\n" + claim.summary + "\n</claim>"
                }
            ],
        }

        msgs, calls = await anthropic_call_tool(
            client,
            tools,
            [initial_message, assistant_reply, task_message],
            "You are a helpful scientific paper reviewer. You are an expert in statistical techniques and hypothesis testing.",
            must_call=dct.name,
            should_stop=should_stop,
        )
        
        assert len(claim_res) == 1
        detailed_claim_results.append(claim_res[0])
        print(claim_res[0])

    print(claim_list)
    print(detailed_claim_results)

if __name__ == "__main__":
    # tool = RExecTool()

    # # test R script to do a simple t-test analysis
    # TEST = """
    # print('asdf')
    # pvalue <- t.test(rnorm(100), rnorm(100, mean = 1))$p.value
    # print(pvalue)
    # """

    import pyalex

    work = pyalex.Works()["https://doi.org/10.1152/ajpgi.1987.253.5.g601"]
    md = convert_doi(work)

    async def test_tool():
        await process_file(md)

    asyncio.run(test_tool())
