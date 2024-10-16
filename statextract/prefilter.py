import re
from typing import Awaitable, Callable
from anthropic import AsyncAnthropic
from instructor import AsyncInstructor
import pydantic

from statextract.cache_typed import Cache

P_VALUE_REGEX = re.compile(r"[pP]\s*[=<]\s*[0-9\.]+")

def prefilter_regex(abstract: str) -> bool | None:
    if P_VALUE_REGEX.search(abstract):
        return True
    return False


class PrefilterAnswer(pydantic.BaseModel):
    reasoning: str = pydantic.Field(description="A short explanation for why the paper was included or excluded.")
    answer: bool
    

# def prefilter_llm(abstract: str) -> bool | None:





class ClassifyNHST(pydantic.BaseModel):
    """
    Determine if the paper is a study that performs null hypothesis significance testing. That is, does the paper define one or more hypotheses, collect data for them, perform statistical tests, and report test statistics of the results.
    
    If the paper is a meta-analysis, systematic review, or other non-traditional study, return False.
    If the paper does not report test statistics for the claims, return False.
    """
    is_nhst: bool = pydantic.Field(..., description="Whether the paper is a study that performs null hypothesis significance testing")
    


# cache = Cache()
# @cache
def mk_classify_fn(client: AsyncInstructor) -> Callable[[str, str, str], Awaitable[ClassifyNHST]]:
    async def classify_nhst(title: str, author: str, contents: str) -> ClassifyNHST:
        model = ClassifyNHST
        prompt = f"""
        Please classify if the following paper, "{title}", by {author}, is a study that performs null hypothesis significance testing and reports p-values.
        
        <paper>
        {contents}
        </paper>
        """
        res: ClassifyNHST = await client.chat.completions.create(model="claude-3-haiku-20240307", messages=[{"role": "user", "content": prompt}], max_tokens=1000,response_model=model)
        return res
    return Cache()(classify_nhst)

