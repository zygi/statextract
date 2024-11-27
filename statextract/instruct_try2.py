
import csv
import itertools
import json
from pathlib import Path
import re
from typing import Sequence, TypedDict
import pyalex
import tqdm
from typing_extensions import Literal
import instructor

# instruct.instruct("What is the capital of France?")
from instructor import AsyncInstructor, Instructor, Mode, OpenAISchema, patch
from anthropic import Anthropic, AsyncAnthropic
from pydantic import BaseModel
import pydantic

from statextract import prefilter
from statextract import helpers
from statextract.agent.agent import anthropic_call_tool
from statextract.agent.tools import AnswerTool
from statextract.cache_typed import Cache
from statextract.helpers import UsageCounter, collect_model_inputs, fetch_work, form_path_base
from statextract.md_retriever import parse_work
from statextract.typedefs import PaperMD
from rich import print

P_VALUE_REGEX = re.compile(r"^[pP]\s*[=<]\s*[0-9\.×\^\-e]+$", re.IGNORECASE)

class ClaimSummaries(OpenAISchema):
    """
    A collection of tested main hypotheses and their test statistics extracted from a paper. Please only include the hypotheses that are the core focus of the paper.
    Specifically, only include them if they are referenced in the paper's abstract. At most 3.
    If the paper doesn't perform NHST, return an empty list.
    """

    claims: list[str] = pydantic.Field(
        ..., description="The main claims.", max_length=3
    )


class Claim(OpenAISchema):
    """
    An p-value-supported claim and its associated p-value.
    """

    chain_of_thought: str = pydantic.Field(
        ..., description="The chain of thought you can use for reasoning. "
    )

    # claim: str = pydantic.Field(
    #     ...,
    #     description="The paraphrased alternative (not null) hypothesis that is being tested",
    # )

    specific_choice: str | None = pydantic.Field(
        default=None,
        description="If the claim provided is not specific, i.e. it doesn't refer to a single hypothesis test, choose one specific hypothesis tested that represents the claim, and describe it here. For example, if the claim is 'The differences in the groups' Big 5 scores are statistically significant', and the paper has a table with 5 regression coefficients for each group, arbitrarily choose one of them and describe it here. DO NOT INCLUDE TEST STATISTICS OR P-VALUES IN THIS FIELD.\nIf the claim is already specific, leave this blank.",
        json_schema_extra={
            "type": "string",
        },
    )

    @pydantic.field_validator("specific_choice")
    @classmethod
    def check_specific_choice(cls, v: str | None) -> str | None:
        if v is None or v.strip() == "":
            return None
        return v

    p_value: str | None = pydantic.Field(
        default=None,
        description="The p-value of the claim. Enter either p<[number] if only a bound is provided, or p=[number] if an exact p-value is specified in the paper. If the paper doesn't give a p-value, set this to an empty string.",
        json_schema_extra={
            "type": "string",
        },
    )

    # @pydantic.computed_field
    # @property
    # def p_value_computed(self) -> str | None:
    #     if self.p_value.strip() == "":
    #         return None
    #     return self.p_value

    # validator
    @pydantic.field_validator("p_value")
    @classmethod
    def check_p_value(cls, v: str | None) -> str | None:
        if v is None or v.strip() == "":
            return None
        if "<UNKNOWN>" in v:
            return None
        if not P_VALUE_REGEX.match(v):
            raise ValueError("P-value must be in the format p<... or p=...")
        return v

    claim_location_type: Literal["abstract", "text", "table", "figure"] | None = (
        pydantic.Field(
            default=None,
            description="Where in the paper the claim is located.",
            json_schema_extra={
                "type": "string",
            },
        )
    )

    claim_bounding_box_page_num: int = pydantic.Field(
        ...,
        description="The number of the page the claim is on. 0-indexed. Note that this is based on the number of THE IMAGES PASSED IN, not the actual page labels in the paper.",
    )
    claim_bounding_box: list[int] = pydantic.Field(
        ...,
        description="The bounding box of the claim in the paper. [top, left, bottom, right]. In pixels.",
        min_length=4,
        max_length=4,
    )


class Claims(pydantic.BaseModel):
    claims: list[Claim] = pydantic.Field(
        ...,
        description="The main claims and their test statistics extracted from the paper",
    )


SYSTEM_PROMPT = """\
You are an expert at analyzing scientific papers. Your goal will be to extract the significant 
"""

from anthropic import types as atypes
# from anthropic.types.chat import ChatCompletionMessageParam

# from openai.types.chat import ChatCompletionMessageParam

import copy


def rm_images(o) -> dict:
    oo = copy.deepcopy(o)

    def recurse(o):
        if isinstance(o, dict):
            if "type" in o and o["type"] == "image":
                return None
            return {k: recurse(v) for k, v in o.items() if v is not None}
        elif isinstance(o, list):
            return [recurse(v) for v in o if v is not None]
        else:
            return o

    res = recurse(oo)
    assert res is not None
    return res


def mk_summaries_prompt() -> str:
    return """First, your goal is to extract the main claims statistically supported claims of the paper. Submit them using the `submit_claims` tool. Please only call the tool once."""


def mk_specific_prompt(claim: str, specific_choice: str | None = None) -> str:
    specific_choice_prompt = ""
    if specific_choice is not None:
        specific_choice_prompt = f"\nIn particular, you should evaluate the following specific claim:\n<specific_choice>\n{specific_choice}\n</specific_choice>"

    return f"""
A previous agent has extracted several important claims from the paper. Your job is to investigate a specific claim, and to extract the p-value and the claim location from the paper. Use the `investigate_claim` tool to do this.
In particular, you should only report an explicitly stated p-value. It might be stated in many ways, e.g.
- Next to the claim itself
- In a table, signified by the number of asterisks next to the test statistic
- In a figure
- Not explicitly stated next to the claim, but there is a sentence saying something like "all presented results are statistically significant at the p<0.05 level".

DO NOT guess a p-value. If the paper just says "statistically significant" without giving a p-value ever, leave it blank instead of just putting p<0.05.

With that in mind: the claim is\n<claim>\n{claim}\n</claim>{specific_choice_prompt}
"""


def mk_base_prompt() -> str:
    return """The paper is in the following images. I'll give you the question in the next message. Is that okay?"""


def mk_message_base_images(md: PaperMD) -> list[atypes.MessageParam]:
    text, images = collect_model_inputs(md)
    # formed_images = [instructor.Image.from_raw_base64(data=base64) for base64, _ in images[:1]]
    formed_images: list[atypes.ImageBlockParam] = [
        {
            "type": "image",
            "source": {"type": "base64", "data": base64, "media_type": "image/jpeg"},
        }
        for base64, _ in images
    ]  # TODO fix

    anthropic_messages_base: list[atypes.MessageParam] = [
        atypes.MessageParam(
            {
                "role": "user",
                "content": [
                    atypes.TextBlockParam(
                        {
                            "type": "text",
                            "text": mk_base_prompt(),
                        }
                    )
                ]
                + formed_images,
            }
        ),
        {
            "role": "assistant",
            "content": [
                atypes.TextBlockParam(
                    {
                        "type": "text",
                        "text": "Got it, I'm ready to start.",
                        "cache_control": {"type": "ephemeral"},
                    }
                )
            ],
        },
    ]

    return anthropic_messages_base


def mk_message_base_text(md: PaperMD) -> list[atypes.MessageParam]:
    text, images = collect_model_inputs(md)
    # formed_images = [instructor.Image.from_raw_base64(data=base64) for base64, _ in images[:1]]
    # formed_images: list[atypes.ImageBlockParam] = [
    #     {
    #         "type": "image",
    #         "source": {"type": "base64", "data": base64, "media_type": "image/jpeg"},
    #     }
    #     for base64, _ in images
    # ]  # TODO fix

    anthropic_messages_base: list[atypes.MessageParam] = [
        atypes.MessageParam(
            {
                "role": "user",
                "content": [
                    atypes.TextBlockParam(
                        {
                            "type": "text",
                            "text": f"The paper is provided as follows:\n\n<paper>\n{text}\n</paper>\n\nI'll give you the question in the next message. Is that okay?",
                        }
                    )
                ],
            }
        ),
        {
            "role": "assistant",
            "content": [
                atypes.TextBlockParam(
                    {
                        "type": "text",
                        "text": "Got it, I'm ready to start.",
                        "cache_control": {"type": "ephemeral"},
                    }
                )
            ],
        },
    ]

    return anthropic_messages_base


async def process_md_claim_details(
    client: AsyncAnthropic,
    message_base: list[atypes.MessageParam],
    tool_list: list[AnswerTool],
    claims: list[str],
    usage_counter: UsageCounter,
    specific_choices: Sequence[str | None] | None = None,
    model: str = "claude-3-haiku-20240307",
):
    if specific_choices is None:
        specific_choices = [None] * len(claims)
    
    claims_all: list[Claim] = []

    for c, sc in zip(claims, specific_choices):
        anthropic_message_investigate_claim: atypes.MessageParam = {
            "role": "user",
            "content": mk_specific_prompt(c, sc),
        }

        holder: Claim | None = None

        async def investigate_claim_cb(x: Claim):
            nonlocal holder
            holder = x

        async def should_stop_investigate_claim(_, __):
            return holder is not None

        investigate_claim_tool = [t for t in tool_list if t.name == "investigate_claim"][0]
        investigate_claim_tool.cb = investigate_claim_cb

        msgs, calls = await anthropic_call_tool(
            client,
            tool_list,
            message_base + [anthropic_message_investigate_claim],
            SYSTEM_PROMPT,
            model=model,
            max_tokens=4096,
            must_call="investigate_claim",
            should_stop=should_stop_investigate_claim,
        )
        # print([rm_images(c) for c in calls])
        for msg in msgs:
            usage_counter.add_cache_creation_input_tokens(msg.usage)
        # assert holder is not None
        claims_all.append(holder)  # type: ignore

        # print([rm_images(c) for c in calls])
        
    return claims_all

async def process_md(
    md: PaperMD,
    client: AsyncAnthropic,
    usage_counter: UsageCounter,
    model: str = "claude-3-haiku-20240307",
):
    anthropic_messages_base = mk_message_base_images(md)
    # anthropic_messages_base = mk_message_base_text(md)

    anthropic_message_summaries: atypes.MessageParam = {
        "role": "user",
        "content": mk_summaries_prompt(),
    }

    claims_holder: ClaimSummaries | None = None

    async def submit_claims(x: ClaimSummaries):
        nonlocal claims_holder
        claims_holder = x

    async def should_stop_claims(_, __):
        return claims_holder is not None

    async def investigate_claim_cb_temp(x: Claim):
        pass

    submit_claims_tool = AnswerTool[ClaimSummaries](
        "submit_claims",
        "Submit the claims you extracted from the paper.",
        ClaimSummaries,
        submit_claims,
    )

    investigate_claim_tool = AnswerTool[Claim](
        "investigate_claim",
        "Investigate a specific claim.",
        Claim,
        investigate_claim_cb_temp,
    )

    msgs, calls = await anthropic_call_tool(
        client,
        [submit_claims_tool, investigate_claim_tool],
        anthropic_messages_base + [anthropic_message_summaries],
        SYSTEM_PROMPT,
        model=model,
        max_tokens=4096,
        must_call="submit_claims",
        should_stop=should_stop_claims,
    )
    print([rm_images(c) for c in calls])
    for msg in msgs:
        usage_counter.add_cache_creation_input_tokens(msg.usage)
    # assert claims_holder is not None

    claims_all: list[Claim] = await process_md_claim_details(
        client,
        anthropic_messages_base,
        [investigate_claim_tool],
        claims_holder.claims,
        usage_counter,
    )
    # reses_all: list[atypes.MessageParam] = []


    # for c in claims_holder.claims:
    #     anthropic_message_investigate_claim: atypes.MessageParam = {
    #         "role": "user",
    #         "content": mk_specific_prompt(c),
    #     }

    #     holder: Claim | None = None

    #     async def investigate_claim_cb(x: Claim):
    #         nonlocal holder
    #         holder = x

    #     async def should_stop_investigate_claim(_, __):
    #         return holder is not None

    #     investigate_claim_tool.cb = investigate_claim_cb

    #     msgs, calls = await anthropic_call_tool(
    #         client,
    #         [submit_claims_tool, investigate_claim_tool],
    #         anthropic_messages_base + [anthropic_message_investigate_claim],
    #         SYSTEM_PROMPT,
    #         model=model,
    #         max_tokens=4096,
    #         must_call="investigate_claim",
    #         should_stop=should_stop_investigate_claim,
    #     )
    #     print([rm_images(c) for c in calls])
    #     for msg in msgs:
    #         usage_counter.add_cache_creation_input_tokens(msg.usage)
    #     # assert holder is not None
    #     claims_all.append(holder)  # type: ignore

    #     print([rm_images(c) for c in calls])
    # # summs, resp = await client.chat.completions.create_with_completion(
    # #     model="claude-3-haiku-20240307",
    # #     messages=anthropic_messages_base + [anthropic_message_summaries],
    # #     response_model=ClaimSummaries,
    # #     max_tokens=4096,
    # # )

    return claims_holder, claims_all


cc = Cache()
@cc(ignore=["respose_model", "client"])
async def call_instructor_wrapper[T: pydantic.BaseModel](
    client: AsyncInstructor,
    model: str | None,
    messages,
    respose_model: type[T],
    model_forcache: str,
    response_model_name_forcache: str,
    max_tokens: int | None = 4096,
) -> T:
    model_param = {"model": model} if model is not None else {}
    max_tokens_param = {"max_tokens": max_tokens} if max_tokens is not None else {}
    return await client.chat.create(
        messages=messages,
        response_model=respose_model,
        **max_tokens_param,
        **model_param,
    )


def mk_instructor_messages_base_images(md: PaperMD):
    txt, images = collect_model_inputs(md)
    instructor_images = [
        instructor.Image.from_raw_base64(data=base64) for base64, _ in images
    ]

    return [
        {
            "role": "user",
            "content": instructor_images + [mk_base_prompt()],
        },
        {
            "role": "assistant",
            "content": "Got it, I'm ready to start.",
        },
    ]


def mk_instructor_messages_base_text(md: PaperMD):
    txt, images = collect_model_inputs(md)
    return [
        {
            "role": "user",
            "content": f"The paper is provided as follows:\n\n<paper>\n{txt}\n</paper>\n\nI'll give you the question in the next message. Is that okay?",
        },
        {
            "role": "assistant",
            "content": "Got it, I'm ready to start.",
        },
    ]


async def process_md_using_instructor(
    md: PaperMD,
    client: AsyncInstructor,
    claims_with_places: list[tuple[str, str | None]],
    model: str | None = None,
    max_tokens: int | None = 4096,
    input_mode: Literal["images", "text"] = "images",
):
    if input_mode == "images":
        instructor_messages_base = mk_instructor_messages_base_images(md)
    else:
        instructor_messages_base = mk_instructor_messages_base_text(md)

    reses: list[Claim] = []
    for claim, place in claims_with_places:
        instructor_messages_claim = [
            {
                "role": "user",
                "content": mk_specific_prompt(claim, place),
            },
        ]

        res = await call_instructor_wrapper(
            client,
            model,
            instructor_messages_base + instructor_messages_claim,
            Claim,
            model if model is not None else "unknown",
            str(Claim),
            max_tokens,
        )
        reses.append(res)

    return reses


class Result(pydantic.BaseModel):
    claims_holder: ClaimSummaries | None
    claims_all: list[Claim]
    usage_counter: UsageCounter | None


# class ResultRepl(pydantic.BaseModel):
#     claims_all: list[Claim | None]

if __name__ == "__main__":
    import hishel

    from dotenv import load_dotenv

    load_dotenv()

    sonnet_result_dir = Path("results/claude-3-5-sonnet-latest")
    
    
    sonnet_repl_text_output_path = Path("results/claude-3-5-sonnet-latest-text")
    sonnet_repl_text_output_path.mkdir(parents=True, exist_ok=True)

    oai_output_path = Path("results/gpt-4o")
    oai_output_path.mkdir(parents=True, exist_ok=True)

    
    oai_text_output_path = Path("results/gpt-4o-text")
    oai_text_output_path.mkdir(parents=True, exist_ok=True)


    gemini_output_path = Path("results/gemini")
    gemini_output_path.mkdir(parents=True, exist_ok=True)

    cohere_output_path = Path("results/cohere")
    cohere_output_path.mkdir(parents=True, exist_ok=True)

    hc = hishel.Controller(
        cacheable_methods=["GET", "POST"],
        allow_stale=True,
        always_revalidate=False,
        force_cache=True,
    )
    storage = hishel.AsyncFileStorage()
    client = hishel.AsyncCacheClient(controller=hc, storage=storage)
    # async def main():
    #     # client = instructor.from_anthropic(AsyncAnthropic(), enable_prompt_caching=True)

    #     client = AsyncAnthropic(http_client=client)

    #     with open("short_mds_ids.json") as f:
    #         ids = json.load(f)

    #     print(ids)

    #     result_dir = Path("results/claude-3-5-sonnet-latest")
    #     result_dir.mkdir(parents=True, exist_ok=True)

    #     import tqdm

    #     for id in tqdm.tqdm(ids):
    #         path = Path(result_dir) / f"{id.split('/')[-1]}.json"
    #         if path.exists():
    #             continue
    #         work = fetch_work(id)
    #         md = parse_work(work)
    #         usage_counter = UsageCounter()
    #         claims_holder, claims_all = await process_md(md, client, usage_counter, model="claude-3-5-sonnet-latest")

    #         with open(path, "w") as f:
    #             f.write(Result(
    #                 claims_holder=claims_holder,
    #                 claims_all=claims_all,
    #                 usage_counter=usage_counter,
    #             ).model_dump_json())

    import openai
    
    def load_results_from(result_dir: Path):
        with open("short_mds_ids.json") as f:
            ids = json.load(f)
        
        files = result_dir.glob("*.json")
        
        # order by position in ids
        files = sorted(files, key=lambda p: ids.index(f"https://openalex.org/{p.stem}"))
        
        return [
            Result.model_validate_json(p.read_text())
            for p in files
        ], ids
    
    def get_claims_with_places():
        reses, ids = load_results_from(sonnet_result_dir)
        mds = [parse_work(fetch_work(id)) for id in ids]

        claims_with_places = [
            [
                (r.claims_holder.claims[i], c.specific_choice)
                for (i, c) in enumerate(r.claims_all)
            ]
            for r in reses
        ]

        # print([len(cwp) for cwp in claims_with_places])
        return mds, claims_with_places


    async def get_instructor(
        instructor_client: AsyncInstructor,
        output_path: Path,
        model_name: str | None = None,
        max_tokens: int | None = 4096,
        input_mode: Literal["images", "text"] = "images",
    ):
        # first, fetch all results from sonnet's result_dir
        mds, claims_with_places = get_claims_with_places()
        print([len(cwp) for cwp in claims_with_places])
        # print(Claim.model_json_schema())

        # return

        for md, cwp in tqdm.tqdm(list(zip(mds, claims_with_places))):
            if (output_path / f"{md.id.split('/')[-1]}.json").exists():
                continue
            # print(cwp)
            reses = await process_md_using_instructor(
                md,
                instructor_client,
                cwp,
                model=model_name,
                max_tokens=max_tokens,
                input_mode=input_mode,
            )
            # print(reses)
            # if len(reses) > 0:
            #     print(type(reses[0]))
            # print(r.mo)

            reses = [Claim.model_validate(r.model_dump()) for r in reses]

            # print(reses)

            with open(output_path / f"{md.id.split('/')[-1]}.json", "w") as f:
                f.write(
                    Result(
                        claims_holder=None,
                        claims_all=reses,
                        usage_counter=None,
                    ).model_dump_json()
                )

            # break

    async def get_instructor_openai():
        instructor_client = instructor.from_openai(
            openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), http_client=client)
        )
        await get_instructor(instructor_client, oai_output_path, model_name="gpt-4o")
        
        
    async def get_instructor_openai_text():
        instructor_client = instructor.from_openai(
            openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), http_client=client)
        )
        await get_instructor(instructor_client, oai_text_output_path, input_mode="text", model_name="gpt-4o")

    async def get_instructor_gemini():
        import google.generativeai as genai
        import os

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        # genai.configure(client_options=)

        gai = genai.GenerativeModel(
            model_name="gemini-1.5-pro-002",
        )

        # print(Claim.model_json_schema())

        # return

        instructor_client = instructor.from_gemini(gai, use_async=True)
        await get_instructor(
            instructor_client, gemini_output_path, max_tokens=None, input_mode="text"
        )

    async def get_instructor_cohere():
        import cohere
        import os

        instructor_client = instructor.from_cohere(cohere.AsyncClient())

        await get_instructor(instructor_client, cohere_output_path, max_tokens=None)

    async def get_claude_text_verify():
        mds, claims_with_places = get_claims_with_places()
        # print(Claim.model_json_schema())
                                        
                                        
        anthropic_client = AsyncAnthropic(http_client=client)

        for md, cwp in tqdm.tqdm(list(zip(mds, claims_with_places))):
            output_path = sonnet_repl_text_output_path / f"{md.id.split('/')[-1]}.json"
            if output_path.exists():
                continue
            
            usage_counter = UsageCounter()
            
            anthropic_messages_base = mk_message_base_text(md)
            
            async def _empty_cb(x):
                pass
            
            tools = [
                AnswerTool[ClaimSummaries](
                    "submit_claims",
                    "Submit the claims you extracted from the paper.",
                    ClaimSummaries,
                    _empty_cb,
                ),
                AnswerTool[Claim](
                    "investigate_claim",
                    "Investigate a specific claim.",
                    Claim,
                    _empty_cb,
                ),
            ]
            
            
            
            claims, specific_choices = zip(*cwp) if len(cwp) > 0 else ([], [])
            
            reses: list[Claim] = await process_md_claim_details(
                anthropic_client,
                anthropic_messages_base,
                tools,
                claims,
                specific_choices=specific_choices,
                usage_counter=usage_counter,
                model="claude-3-5-sonnet-latest",
            )
            print(reses)
            # if len(reses) > 0:
            #     print(type(reses[0]))

            reses = [Claim.model_validate(r.model_dump()) for r in reses]

            with open(sonnet_repl_text_output_path / f"{md.id.split('/')[-1]}.json", "w") as f:
                f.write(
                    Result(
                        claims_holder=None,
                        claims_all=reses,
                        usage_counter=usage_counter,
                    ).model_dump_json()
                )


        # claims_all: list[Claim] = await process_md_claim_details(
        #     client,
        #     anthropic_messages_base,
        #     [investigate_claim_tool],
        #     claims_holder.claims,
        #     usage_counter,
        # )

    def check_results():
        class PValue(pydantic.BaseModel):
            num: float
            exact: bool

            @classmethod
            def parse(cls, s: str | None):
                if s is None:
                    return None
                
                # print(s)
                # remove spaces
                s = s.replace(" ", "")
                first_two, s = s[:2], s[2:]
                exact = first_two == "p="
                s = s.replace("×10^", "E")
                
                # s = s.split(',')[0]
                num = float(s)
                
                return cls(num=num, exact=exact)

        def is_num_close(p1: PValue | None, p2: PValue | None):
            # print(p1, p2)
            if p1 is None and p2 is None:
                return True
            if p1 is None or p2 is None:
                return False
            
            if p1.num == 0 and p2.num == 0:
                return True
            
            return abs(p1.num - p2.num) / max(p1.num, p2.num) < 1e-2

        anthropic_reses, _ = load_results_from(sonnet_result_dir)
        oai_reses, _ = load_results_from(oai_output_path)
        anthropic_text_reses, _ = load_results_from(sonnet_repl_text_output_path)
        oai_text_reses, _ = load_results_from(oai_text_output_path)
        
        def dump_questions():
            mds, claims_with_places = get_claims_with_places()

            WSL_PATH_BASE = Path("wsl.localhost/Ubuntu-22.04")

            res = []
            for md, cwp in tqdm.tqdm(list(zip(mds, claims_with_places))):
                for claim, place in cwp:
                    res.append([md.id, str(WSL_PATH_BASE / "mnt" / "storage" / "python" / "pcurves" / "data" / "pdfs" / f"{form_path_base(md)}.pdf"), claim, place])

            # dump csv
            with open("questions.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "path", "claim", "place"])
                writer.writerows(res)
        
        dump_questions()
        
        res_lists = [anthropic_reses, anthropic_text_reses, oai_reses, oai_text_reses]
        res_names = ["claude-images", "claude-text", "oai-images", "oai-text"]
        
        # print([len(reses) for reses in res_lists])
        
        flattened_claims = [[x for r in ls for x in r.claims_all] for ls in res_lists]

        zipped_flattened_pvals = [
            [PValue.parse(c.p_value) for c in ls]
            for ls in zip(*flattened_claims)
        ]
        
        unzipped_pvals = list(zip(*zipped_flattened_pvals))

        # print(
        #     [
        #         (PValue.parse(c1.p_value), PValue.parse(c2.p_value), PValue.parse(c3.p_value), PValue.parse(c4.p_value))
        #         for c1, c2 in zipped_flattened_claims
        #     ]
        # )
        
        # pvals = [
            
        # ]
        
        # print(pvals)
        
        # pval_listoflists = list(zip(*pvals))
        
        pairs = list(itertools.combinations(range(len(res_names)), 2))
        pair_concordances = {
            (res_names[i], res_names[j]): sum(
                1
                for px, py in zip(unzipped_pvals[i], unzipped_pvals[j], strict=True)
                if is_num_close(px, py)
            ) / len(unzipped_pvals[i])
            for i, j in pairs
        }
        
        # print(pair_concordances)
        
        import pandas as pd
        
        columns = res_names
        array_of_arrays = [
            [pair_concordances.get((r1, r2), float("nan")) for r2 in res_names]
            for r1 in res_names
        ]
        df = pd.DataFrame(array_of_arrays, columns=columns, index=columns)
        print(df)
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        
        # set backend to Agg
        # plt.switch_backend("Agg")
        
        # fig = sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu")
        # # plt.show()
        # plt.savefig("concordance.png")
        
        # now format it as a table
        # import tabulate
        # print(tabulate.tabulate(pair_concordances, headers="keys"))
        
        # close_count = sum(
        #     1
        #     for c1, c2 in zipped_flattened_claims
        #     if is_num_close(PValue.parse(c1.p_value), PValue.parse(c2.p_value))
        #     and c1.p_value is not None
        #     and c2.p_value is not None
        # )

        # print(close_count / len(zipped_flattened_claims))

        # print(len(zipped_flattened_claims))

    # async def text_debug():
    #     with open("short_mds_ids.json") as f:
    #         ids = json.load(f)
    #     mds = [parse_work(fetch_work(id)) for id in ids]
        
    #     claims = [Result.model_validate_json(p.read_text()) for p in sonnet_result_dir.glob("*.json")]
        
    #     md = mds[1]
    #     text, images = collect_model_inputs(md)
    #     pdf_path = Path(f"data/pdfs/{form_path_base(md)}.pdf")
    #     print(text[:10000])
    #     print(pdf_path)
        
    #     print(claims[1])
        
    import asyncio

    # asyncio.run(get_claude_text_verify())
    # asyncio.run(get_instructor_openai())
    # print(load_results_from(sonnet_result_dir))
    # asyncio.run(text_debug())
    # asyncio.run(get_instructor_openai_text())
    # asyncio.run(get_instructor_gemini())
    # asyncio.run(check_results())
    check_results()

# if __name__ == "__main__":


# with open("./book.txt") as f:
#     book = f.read()

# resp = client.chat.completions.create(
#     model="claude-3-haiku-20240307",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "<book>" + book + "</book>",
#                     "cache_control": {"type": "ephemeral"},
#                 },
#                 {
#                     "type": "text",
#                     "text": "Extract a character from the text given above",
#                 },
#             ],
#         },
#     ],
#     response_model=Character,
#     max_tokens=1000,
# )
