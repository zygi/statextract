import typing
import abc
import asyncio
import anthropic
from typing_extensions import final
from anthropic import AsyncAnthropic
import pydantic
from instructor import function_calls

import anthropic.types as atypes

InputType = typing.TypeVar("InputType", bound=pydantic.BaseModel)
OutputType = typing.TypeVar("OutputType")

class Tool(typing.Generic[InputType, OutputType], abc.ABC):
    def __init__(self, input_type: typing.Type[InputType], output_type: typing.Type[OutputType]):
        self.input_type = input_type
        self.output_type = output_type
        
    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abc.abstractmethod
    def description(self) -> str:
        pass

    @abc.abstractmethod
    async def execute(self, input: InputType) -> OutputType:
        pass
    
    async def run(self, input: str) -> OutputType:
        try:
            parsed = self.input_type.model_validate_json(input)
            return await self.execute(parsed)
        except Exception as e:
            raise Exception(f"Failed to parse input:\n{e}")
    
    
    @property
    def anthropic_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_type.model_json_schema(),
        }

class RExecInput(pydantic.BaseModel):
    script: str = pydantic.Field(description="The R script to execute.")

@final
class RExecTool(Tool[RExecInput, str]):
    def __init__(self):
        self.timeout = 10
        super().__init__(RExecInput, str)
        
    @property
    def name(self) -> str:
        return "RExecTool"
    
    @property
    def description(self) -> str:
        return "This tool allows you to execute R scripts."
        
    async def execute(self, input: RExecInput) -> str:
        process = await asyncio.create_subprocess_exec("timeout", str(self.timeout), "Rscript", "-e", input.script, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise Exception(f"R script failed with return code {process.returncode}:\n{stderr.decode()}")
        return stdout.decode()

T = typing.TypeVar("T", bound=function_calls.OpenAISchema)
@final
class AnswerTool(Tool[T, str], typing.Generic[T]):
    def __init__(self, result_type: typing.Type[T], cb: typing.Callable[[T], typing.Awaitable[None]]):
        super().__init__(result_type, str)
        self.cb = cb
        # self.result_type = result_type
        
    @property
    def name(self) -> str:
        return "AnswerTool"
    
    @property
    def description(self) -> str:
        return "This tool allows you to submit a result to the user."
        
    async def execute(self, input: T) -> str:
        await self.cb(input)
        return "Successfully submitted"

class TestAnswer(function_calls.OpenAISchema):
    bbb: list[int] = pydantic.Field(description="The list of numbers")
    answer: str = pydantic.Field(description="The answer to the question")


class ClaimSummary(pydantic.BaseModel):
    """
    A core claim of the paper that is supported by quantitative analysis and a statistical test.
    """
    summary: str = pydantic.Field(description="Claim summary.")
    position_in_text: str = pydantic.Field(description="The position of the claim in the text, described verbally.")
    
class Claims(pydantic.BaseModel):
    """
    A collection of core claims of the paper that are supported by quantitative analysis and a statistical test.
    Please only choose the most important claims of the paper. They should be 1) mentioned in the abstract, AND 2) supported by some kind of statistical test.
    
    Output no more than 3 claims.
    If the paper doesn't perform null hypothesis significance testing, output an empty list.
    """
    chain_of_thought: str = pydantic.Field(description="The chain of thought you can use for reasoning.")
    claims: list[ClaimSummary] = pydantic.Field(description="The list of claims. No more than 3.")
    
class DetailedClaim(pydantic.BaseModel):
    """
    A detailed claim that is supported by quantitative analysis and a statistical test.
    """
    summary: str = pydantic.Field(description="Claim summary.")
    
    original_test_statistics: list[str] = pydantic.Field(..., description="""\
The test statistics found in the paper that support the claim. Each string should be in one of the following formats:
`F(_, _) = _`
`t(_) = _`
`chi2(_) = _`
`r(_) = _`
`z = _`
`p = _`
`p < _`
where _ is a numeric value. Each test statistic should be in a separate string. Every string should be formatted exactly as shown above. Only include the statistics from the paper, not calculations you made yourself.""")
    
    nearby_citation: str = pydantic.Field(..., description="A short exact substring of text (not numbers or equations) that is close or at the test statistics in the text. This will be used to locate the region where the claim exists so copy it EXACTLY from the paper contents, including formatting, any mistyping, etc.")
    
    final_p_value: typing.Optional[float] = pydantic.Field(..., description="The final exact p-value of the claim. This field requires a precise value, NOT a bound like `p < 0.05`. If the paper reports such an exact p-value, like `p = 0.0317`, output it. If the paper reports other test statistics that let you calculate a p-value, output that p-value after calculating it exactly. If the paper only reports a bound, and doesn't give a way to calculate an exact p-value, leave the field empty.")
    
# class ClaimSummaryCollection(pydantic.BaseModel):
#     """
#     A collection of quantitatively-supported main claims extracted from a paper. 
#     """
#     claims: list[str] = pydantic.Field(description="The list of claims. No more than 3.")
    # chain_of_thought: str = pydantic.Field(..., description="The chain of thought you can use for reasoning. ")
    # claims: list[ClaimSummary] = pydantic.Field(..., description="The main claims and their test statistics extracted from the paper")
    
    

async def anthropic_tool_caller(client: AsyncAnthropic, tools: typing.Sequence[Tool], init_messages: list[atypes.MessageParam]):
    def append_cache_control(dct: dict):
        dct["cache_control"] = "ephemeral"
    
    tool_dict = {t.name: t for t in tools}
    
    message = await client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        system=[{"type": "text", "text": "You are a helpful assistant that can run R scripts."}],
        tools = [append_cache_control(t.anthropic_schema) for t in tools], # type: ignore
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "Hello, Claude"}]}
        ]
    )
    
    tool_results: list[atypes.ToolResultBlockParam] = []
    
    # find tool calls
    if message.stop_reason == "tool_use":
        for c in message.content:
            if c.type == "tool_use":
                if c.name not in tool_dict:
                    tool_results.append({"tool_use_id": c.id, "is_error": True, "content": "Tool not found", "type": "tool_result"})
                    continue
                
                tool = tool_dict[c.name]
                # try 
                
                #     input = tool.input_type.model_validate_json(c.input.input)
                #     output = await tool.execute(input)
                #     message = await client.messages.create(
                #         model="claude-3-5-sonnet-20240620",
                #         max_tokens=1024,
# test 
if __name__ == "__main__":
    # tool = RExecTool()
    
    # # test R script to do a simple t-test analysis
    # TEST = """
    # t.test(rnorm(100), rnorm(100, mean = 1))
    # """
    
    # async def test_tool():
    #     print(await tool.run(TEST))
    
    # asyncio.run(test_tool())
    
    # print(TestAnswer.anthropic_schema)
    
    async def print_cb(x: TestAnswer):
        print(x)
    
    st = AnswerTool(TestAnswer, print_cb)
    
    async def test_tool():
        print(await st.run("""{"bbb": [1, 2, 3], "answer": "42"}"""))
    
    asyncio.run(test_tool())