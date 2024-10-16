import copy
import typing
import abc
import asyncio
import anthropic
from typing_extensions import final
from anthropic import AsyncAnthropic
import pydantic
from instructor import function_calls
from rich import print

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

    async def format_output(self, output: OutputType) -> str:
        # return self.output_type.model_dump_json(output)
        if isinstance(self.output_type, pydantic.BaseModel):
            return self.output_type.model_dump_json(output)
        return str(output)

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
    
    



class TestCalculatorInput(pydantic.BaseModel):
    a: float = pydantic.Field(description="The first number")
    b: float = pydantic.Field(description="The second number")
    op: typing.Literal["+", "-", "*", "/"] = pydantic.Field(
        description="The operation to perform"
    )


@final
class TestCalculatorTool(Tool[TestCalculatorInput, float]):
    def __init__(self):
        super().__init__(TestCalculatorInput, float)

    async def execute(self, input: TestCalculatorInput) -> float:
        if input.op == "+":
            return input.a + input.b
        elif input.op == "-":
            return input.a - input.b
        elif input.op == "*":
            return input.a * input.b
        elif input.op == "/":
            return input.a / input.b
        else:
            raise Exception(f"Invalid operation: {input.op}")

    @property
    def name(self) -> str:
        return "TestCalculatorTool"

    @property
    def description(self) -> str:
        return "This tool allows you to perform basic arithmetic operations."

