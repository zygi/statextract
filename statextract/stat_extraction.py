import asyncio
import datetime
import os
from pathlib import Path
from typing import Literal, Tuple, Union
import numpy as np
import pydantic
import instructor
from dotenv import load_dotenv
from anthropic import AsyncAnthropic
import pymupdf
import pypcurve
from rich import print
import re
import scipy
import unidecode

import matplotlib.pyplot as plt
# import perscache
from cache_typed import Cache
from tqdm import tqdm

from helpers import DWECK, PaperMD, assert_str, author, filter_mds_by_pdf, form_path_base, get_mds_with_dois
import helpers
load_dotenv()

class ClassifyNHST(pydantic.BaseModel):
    """
    Determine if the paper is a study that performs null hypothesis significance testing. That is, does the paper define one or more hypotheses, collect data for them, perform statistical tests, and report test statistics of the results.
    
    If the paper is a meta-analysis, systematic review, or other non-traditional study, return False.
    If the paper does not report test statistics for the claims, return False.
    """
    is_nhst: bool = pydantic.Field(..., description="Whether the paper is a study that performs null hypothesis significance testing")
    

client = instructor.from_anthropic(AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"]))

cache = Cache()
@cache
async def classify_nhst(title: str, author: str, contents: str) -> ClassifyNHST:
    model = ClassifyNHST
    prompt = f"""
    Please classify if the following paper, "{title}", by {author}, is a study that performs null hypothesis significance testing and reports p-values.
    
    <paper>
    {contents}
    </paper>
    """
    return await client.chat.completions.create(model="claude-3-haiku-20240307", messages=[{"role": "user", "content": prompt}], max_tokens=1000,response_model=model)



class ClaimSummary(pydantic.BaseModel):
    """
    An NHST claim and its supporting test statistics.
    """ 
    claim: str = pydantic.Field(..., description="The paraphrased alternative (not null) hypothesis that is being tested")
    # p_value: float = pydantic.Field(..., description="The p-value of the test")
    # is_exact: bool = pydantic.Field(..., description="If the p-value is provided exactly (e.g. p=0.036), set this to True. If it's a bound (e.g. p<0.001), set this to False.")
    
    test_statistics: list[str] = pydantic.Field(..., description="""\
The test statistics that support the claim. Each string should be in one of the following formats:
`F(_, _) = _`
`t(_) = _`
`chi2(_) = _`
`r(_) = _`
`z = _`
`p = _`
`p < _`
where _ is a numeric value. Each test statistic should be in a separate string. Every string should be formatted exactly as shown above.""")
    
    nearby_citation: str = pydantic.Field(..., description="A short exact substring of text (not numbers or equations) that is close or at the test statistics in the text. This will be used to locate the region where the claim exists so copy it EXACTLY from the paper contents, including formatting, any mistyping, etc.")
    
class ClaimSummaryCollection(pydantic.BaseModel):
    """
    A collection of tested main hypotheses and their test statistics extracted from a paper. Please only include the hypotheses that are the core focus of the paper.
    Specifically, only include them if they are referenced in the paper's abstract. At most 3.
    If the paper doesn't perform NHST, return an empty list.
    """
    chain_of_thought: str = pydantic.Field(..., description="The chain of thought you can use for reasoning. ")
    claims: list[ClaimSummary] = pydantic.Field(..., description="The main claims and their test statistics extracted from the paper")

@cache
async def extract_pvalues(title: str, author: str, contents: str) -> ClaimSummaryCollection:
    model = ClaimSummaryCollection
    prompt = f"""
    Please extract the main claims and their p-values from the following paper, "{title}", by {author}.
    
    <paper>
    {contents}
    </paper>
    """
    return await client.chat.completions.create(model="claude-3-5-sonnet-20240620", messages=[{"role": "user", "content": prompt}], max_tokens=8192,response_model=model)

def display_supporting_images(md: PaperMD, res: ClaimSummaryCollection):
    pdf = pymupdf.open(Path("pdfs") / f"{form_path_base(md.doi)}.pdf")
    blocks = [page.get_text(sort=True) for page in pdf]
    # blocks = [page.get_text("blocks") for page in pdf]
    # flatten blocks
    # blocks_with_page = [(block, i) for i, page in enumerate(blocks) for block in page]
    
    
    def normalize(text: str) -> str:
        text = unidecode.unidecode(text)
        text = text.lower()
        text = re.sub(r"[0-9]+", "", text)
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"-", " ", text)
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"=", "", text)
        return text
    
    blocks_with_page = [(block, normalize(block), i) for i, block in enumerate(blocks)]
    # blocks_with_page = [(block, re.sub(r"\s*\n\s*", " ", re.sub(r"-\s*\n\s*", "", block)), i) for i, block in enumerate(blocks)]
    
        
    for block, block_clean, page_idx in blocks_with_page:
        print(block)
        # print(block_clean)
    
    for p_value_res in res.claims:
        print(p_value_res)
        for block, block_clean, page_idx in blocks_with_page:
            # print(block.encode("utf-8"))
            # print(re.sub(r"-?\s*\n\s*", " ", block))
            if normalize(p_value_res.nearby_citation) in block_clean or p_value_res.nearby_citation.lower() in block.lower():
                print(f"Claim: {p_value_res.claim}, p-value: {p_value_res.p_value}, is_exact: {p_value_res.is_exact}, nearby_citation: {p_value_res.nearby_citation}")
                print(f"Page {page_idx}: {block[4]}")
                print()
                break
        else:
            print(f"Claim not found")
            print(normalize(p_value_res.nearby_citation))
            print()
    
# thispercentagewassignificantlyhigherintheingroupidentitycondition(24.3%,9outof37)thanintheoutgroupidentitycondition(3.6%,1outof28),w2(1,n565)55.27,p<.05.experiment2:effectsofsaliency
# thispercentagewassignificantlyhigherintheingroupidentitycondition(24.3%,9outof37)thanintheoutgroupidentitycondition(3.6%,1outof28),w[2](1,n565)55.27,p<.05.
    
StatType = Literal["F", "t", "chi2", "z", "r", "p", "p_bound"]
    
    
def parse_test_statistics(test_statistics: list[str], prefer_not_p: bool = False) -> str | None:
    
    def parse_test_statistic(test_statistic: str) -> Tuple[StatType, str] | None:
        # remove spaces 
        test_statistic = re.sub(r"\s+", "", test_statistic)
        
        # if there is a comma that's not inside a paren, split by it
        if "," in test_statistic:
            # find positions of commas that are not inside parens
            poses = []
            paren_count = 0
            for i, c in enumerate(test_statistic):
                if c == "(" or c == "[":
                    paren_count += 1
                elif c == ")" or c == "]":
                    paren_count -= 1
                elif c == "," and paren_count == 0:
                    poses.append(i)
            
            if len(poses) > 0:
                poses = [0] + poses + [len(test_statistic)]
                test_statistic = [test_statistic[i:j] for i, j in zip(poses, poses[1:])][0]
        try:
            # split on the first = or < from the right side 
            parts = re.split(r"[=<]", test_statistic[::-1], 1)
            lhs = parts[1][::-1]
            rhs = parts[0][::-1]
            rhs_num = float(rhs)
        
            def eat_parens(s: str) -> str:
                # eat parens
                if s[0] == "(":
                    assert s[-1] == ")"
                    return s[1:-1]
                elif s[0] == "[":
                    assert s[-1] == "]"
                    return s[1:-1]
                else:
                    return s
            
            # first, zero arity fns
            if lhs == "p" and "<" in test_statistic:
                return "p_bound", f"p < {rhs_num}"
            elif lhs == "p" and "=" in test_statistic:
                return "p", f"p = {rhs_num}"
            elif lhs == "z":
                return "z", f"z = {rhs_num}"
            # now, positive arity fns
            elif lhs[0] == "F":
                lhs = eat_parens(lhs[1:])
                nums = [int(n) for n in lhs.split(",")]
                assert len(nums) == 2
                return "F", f"F({nums[0]}, {nums[1]}) = {rhs_num}"
            elif lhs[0] == "t":
                # if lhs == "t":
                #     return "t", f"t = {rhs_num}"
                lhs = eat_parens(lhs[1:])
                nums = [int(n) for n in lhs.split(",")]
                assert len(nums) == 1
                return "t", f"t({nums[0]}) = {rhs_num}"
            elif lhs.startswith("chi2") or lhs.startswith("χ2"):
                # lhs = eat_parens(lhs[4:])
                first_paren = max(test_statistic.find("("), test_statistic.find("["))
                lhs = eat_parens(lhs[first_paren:])
                nums = [int(n) for n in lhs.split(",") if not n.lower().startswith("n=")]
                assert len(nums) == 1
                return "chi2", f"chi2({nums[0]}) = {rhs_num}"
            elif lhs[0] == "r":
                # if lhs == "r":
                #     return "r", f"r = {rhs_num}"
                lhs = eat_parens(lhs[1:])
                nums = [int(n) for n in lhs.split(",")]
                assert len(nums) == 1
                return "r", f"r({nums[0]}) = {rhs_num}"
            else:
                print(f"Unknown test statistic: {test_statistic}")
                return None
        
        except Exception as e:
            print(f"Error parsing test statistic: {test_statistic}")
            # raise e
            return None
    
    stats = [parse_test_statistic(ts) for ts in test_statistics]
    
    # if there's a test statistic that's not a p-value, return it
    
    not_p = None
    for stat in stats:
        if stat is None:
            continue
        if stat[0] != "p" and stat[0] != "p_bound":
            not_p = stat[1]
    
    p = None
    for stat in stats:
        if stat is None:
            continue
        if stat[0] == "p" or stat[0] == "p_bound":
            p = stat[1]
            
            
    if prefer_not_p:
        if not_p is not None:
            return not_p
        else:
            return p
    else:
        if p is not None:
            return p
        else:
            return not_p
        
        # parse lhs
        
    
    
async def work():
    # all_mds = list(Path("mds").glob("*.md"))
    author_name = assert_str(author(helpers.TSAI)['display_name'])
    all_mds = filter_mds_by_pdf(get_mds_with_dois(helpers.TSAI))
    contents = [(Path("mds") / f"{form_path_base(md.doi)}.md").read_text() for md in all_mds]
    classifications = [await classify_nhst(md.title, author_name, content) for md, content in tqdm(list(zip(all_mds, contents)))]
    
    good_mds_contents = list((md, c, cl) for (md, c, cl) in zip(all_mds, contents, classifications) if cl.is_nhst)
    
    results: list[ClaimSummaryCollection] = []
    for md, c, cl in tqdm(good_mds_contents[:9999]):
        print(md.title)
        p_values = await extract_pvalues(md.title, author_name, c)
        print(p_values)
        
        # print(len(p_values.claims))   
        results.append(p_values)
        # print(p_values)
        
        # display_supporting_images(md, p_values)
        
        # fetch pdf and try 

    all_p_values = [pv for res in results for pv in res.claims]
    
    # for pv in all_p_values:
    #     print(parse_test_statistics(pv.test_statistics))
    test_stats_flat_not_p = [parse_test_statistics(pv.test_statistics, prefer_not_p=True) for pv in all_p_values]
    test_stats_flat = [parse_test_statistics(pv.test_statistics, prefer_not_p=False) for pv in all_p_values]
    test_stats_flat = [ts for ts in test_stats_flat if ts is not None]
    print(len(test_stats_flat))
    
    
    
    # drop p for now
    test_stats_flat_eq = [ts.replace("<", "=") for ts in test_stats_flat if ts[0] == "p"]
    test_stats_flat_lt = [ts.replace("=", "<") for ts in test_stats_flat if ts[0] == "p"]
    # test_stats_flat = [ts for ts in test_stats_flat if ts[0] != "p"]
    print(len(test_stats_flat_eq))
    print(test_stats_flat_eq)
    
    # print grouped test_stats_flat and their counts
    grouped_test_stats = {}
    for ts in test_stats_flat_lt:
        grouped_test_stats[ts] = grouped_test_stats.get(ts, 0) + 1
    sorted_by_val = sorted(grouped_test_stats.items(), key=lambda x: float(x[0].split("<")[-1]), reverse=True)
    for ts, count in sorted_by_val:
        print(f"{ts}: {count}")
    
    p_floats = [float(ts.split("=")[-1]) for ts in test_stats_flat_eq]
    Z_vals = [scipy.stats.norm.ppf(1 - p / 2) for p in p_floats]
    Z_strings = [f"z={z}" for z in Z_vals]
    for i in ([f"Z={z}" for z in Z_vals]):
        print(i)
        
    # augmented p floats
    p_buckets = [0] + np.logspace(-2, -0.5, num=10).tolist() # logarithmic range from 0 to 0.05
    p_floats_augmented: list[float] = []
    for p in p_floats:
        for i, (a, b) in enumerate(zip(p_buckets, p_buckets[1:])):
            if a <= p <= b:
                REPEATS = 100
                p_floats_augmented.extend(np.random.uniform(a, b, REPEATS).tolist())
                break
        else:
            raise ValueError(f"P-value {p} not in any bucket")
    p_floats_augmented_str = [f"p={p}" for p in p_floats_augmented]
    
    # print(p_floats_augmented)
    
    print(f"{len(all_mds)=}")
    print(f"{len(good_mds_contents)=}")
    print(f"{len([pv for pv in results if len(pv.claims) > 0])=}")
    print(f"{len(p_floats)=}")
    # import pypcurve
    pc = pypcurve.PCurve(p_floats_augmented_str)
    # pc = pypcurve.PCurve(test_stats_flat_eq)
    # pc = pypcurve.PCurve(Z_strings)
    
    ax = pc.plot_pcurve(dpi=100)
    plt.savefig("pcurve.png", dpi=300)
    
asyncio.run(work())