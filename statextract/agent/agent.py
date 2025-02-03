import copy
import typing
import asyncio
from anthropic import AsyncAnthropic
import anthropic
from rich import print

import anthropic.types as atypes
from statextract.agent.tools import TestCalculatorTool, Tool
from anthropic.types.beta.prompt_caching.prompt_caching_beta_message import PromptCachingBetaMessage

CacheControlable = atypes.MessageParam | atypes.ToolParam

def append_cache_control_message(dct: atypes.MessageParam, enabled: bool) -> atypes.MessageParam:
    dct = copy.deepcopy(dct)
    if enabled and isinstance(dct["content"], list):
        # for c in dct["content"]:
        if len(dct["content"]) > 0:
            dct["content"][-1]["cache_control"] = {"type": "ephemeral"}  # type: ignore
    return dct

# def append_cache_control_tool(dct: atypes.ToolParam, enabled: bool) -> atypes.ToolParam:
#     dct = copy.deepcopy(dct)
#     if enabled:
#         dct["cache_control"] = {"type": "ephemeral"}  # type: ignore
#     return dct

    
    
async def _anthropic_tool_caller(
    client: AsyncAnthropic,
    tools: typing.Sequence[Tool],
    init_messages: list[atypes.MessageParam],
    system_prompt: str,
    model: str = "claude-3-5-sonnet-20240620",
    max_tokens: int = 1024,
    prompt_caching: bool = True,
    must_call: bool | str = False,
    temperature: float = 0.0,
):

    tool_dict = {t.name: t for t in tools}

    tool_descriptions = [t.anthropic_schema for t in tools]
    # tool_descriptions = [append_cache_control_tool(t.anthropic_schema, prompt_caching) for t in tools]

    # print(tool_descriptions)
    must_call_arg: atypes.message_create_params.ToolChoice
    if must_call == True:
        must_call_arg = {"type": "any"}
    elif must_call == False:
        must_call_arg = {"type": "auto"}
    else:
        must_call_arg = {"type": "tool", "name": must_call}
    
    msgs = client.beta.prompt_caching.messages if prompt_caching else client.messages
    message = await msgs.create(
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system_prompt}], # type: ignore
        tools=tool_descriptions,  # type: ignore
        messages=init_messages,
        tool_choice=must_call_arg if len(tools) > 0 else anthropic.NotGiven(),
        temperature=temperature,
    )
    
    # print(message)
    
    tool_results: list[atypes.ToolResultBlockParam] = []

    # find tool calls
    if message.stop_reason == "tool_use":
        for c in message.content:
            if c.type == "tool_use":
                if c.name not in tool_dict:
                    tool_results.append(
                        {
                            "tool_use_id": c.id,
                            "is_error": True,
                            "content": "Tool not found",
                            "type": "tool_result",
                        }
                    )
                    continue

                tool = tool_dict[c.name]
                try:
                    msg: object = tool.input_type.model_validate(c.input)
                    # if await should_stop(msg):
                    #     break
                    res = await tool.execute(msg)
                    tool_results.append(
                        {
                            "tool_use_id": c.id,
                            "is_error": False,
                            "content": await tool.format_output(res),
                            "type": "tool_result",
                        }
                    )
                except Exception as e:
                    error_msg = f"Error when calling tool {tool.name}:\n{e}"
                    tool_results.append(
                        {
                            "tool_use_id": c.id,
                            "is_error": True,
                            "content": error_msg,
                            "type": "tool_result",
                        }
                    )

    new_message_base: list[atypes.MessageParam] = [
        {"role": "assistant", "content": message.content},
        {"role": "user", "content": tool_results},
    ]

    return message, new_message_base


async def _false(x: int, y) -> bool:
    return False

_AnthropicMessage = typing.Union[atypes.Message, PromptCachingBetaMessage]

async def anthropic_call_tool(
    client: AsyncAnthropic,
    tools: typing.Sequence[Tool],
    init_messages: list[atypes.MessageParam],
    system_prompt: str,
    model: str = "claude-3-5-sonnet-20240620",
    max_tokens: int = 1024,
    should_stop: typing.Callable[
        [int, list[atypes.MessageParam]], typing.Awaitable[bool]
    ] = _false,
    max_steps: int = 20,
    prompt_caching: bool = True,
    must_call: bool | str = False,
    uncached_init_messages: list[atypes.MessageParam] = [],
    temperature: float = 0.0,
) -> tuple[list[_AnthropicMessage], list[atypes.MessageParam]]:
    if len(init_messages) > 0:
        init_messages[-1] = append_cache_control_message(init_messages[-1], prompt_caching) 
    responses: list[_AnthropicMessage] = []
    for i in range(max_steps):
        message, new_message_base = await _anthropic_tool_caller(
            client,
            tools,
            init_messages + uncached_init_messages,
            system_prompt,
            model,
            max_tokens,
            prompt_caching,
            must_call=must_call,
            temperature=temperature,
        )
        init_messages = init_messages + new_message_base
        responses.append(message)
        if (
            message.stop_reason == "end_turn"
            or message.stop_reason == "max_tokens"
            or message.stop_reason == "stop_sequence"
        ):
            break
        if await should_stop(i, new_message_base) or i == max_steps - 1:
            break
    return responses, init_messages



from openai import AsyncOpenAI, NotGiven
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam, ChatCompletionMessage, ChatCompletionAssistantMessageParam,ChatCompletionMessageToolCallParam, ChatCompletion
from openai.types.completion_usage import CompletionUsage, CompletionTokensDetails, PromptTokensDetails
from openai.types.chat.chat_completion import Choice
    
# class FunctionCallLog(TypedDict):
    
async def _openai_tool_caller(
    client: AsyncOpenAI,
    tools: typing.Sequence[Tool],
    init_messages: list[ChatCompletionMessageParam],
    system_prompt: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 1024,
    must_call: bool | str = False,
    temperature: float = 0.0,
) -> tuple[ChatCompletion, list[ChatCompletionMessageParam]]:
    messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": system_prompt}]
    for msg in init_messages:
        messages.append(msg)

    tools_list = [
        ChatCompletionToolParam(
            type="function",
            function={
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_type.model_json_schema()
            }
        )
        for tool in tools
    ]

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools_list,
        tool_choice="auto" if must_call else NotGiven(),
        max_tokens=max_tokens,
        temperature=temperature
    )

    message = response.choices[0]
    new_message: ChatCompletionAssistantMessageParam = ChatCompletionAssistantMessageParam(
        role="assistant",
        content=message.message.content,
        tool_calls=[ChatCompletionMessageToolCallParam(
            id=tool_call.id,
            function={
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments
            },
            type="function"
        ) for tool_call in message.message.tool_calls] if message.message.tool_calls else [],
        refusal=message.message.refusal,
    )
    
    tool_call_responses: list[ChatCompletionMessageParam] = []

    for tool_call in message.message.tool_calls or []:
        tool_name = tool_call.function.name
        tool = next(t for t in tools if t.name == tool_name)
        
        try:
            result = await tool.run(tool_call.function.arguments)
            formatted_result = await tool.format_output(result)
            
            tool_message: ChatCompletionMessageParam = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": formatted_result
            }
            
            tool_call_responses.append(tool_message)
            
        except Exception as e:
            error_message: ChatCompletionMessageParam = {
                "role": "tool", 
                "tool_call_id": tool_call.id,
                "content": f"Error: {str(e)}"
            }
            tool_call_responses.append(error_message)
            
    return response, [new_message] + tool_call_responses

async def openai_call_tool(
    client: AsyncOpenAI,
    tools: typing.Sequence[Tool],
    init_messages: list[ChatCompletionMessageParam],
    system_prompt: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 1024,
    should_stop: typing.Callable[
        [int, list[ChatCompletionMessageParam]], typing.Awaitable[bool]
    ] = _false,
    max_steps: int = 20,
    must_call: bool | str = False,
    temperature: float = 0.0,
) -> tuple[list[ChatCompletion], list[ChatCompletionMessageParam]]:
    # no explicit caching bc openai doesn't support it
    responses: list[ChatCompletion] = []
    for i in range(max_steps):
        message, new_message_base = await _openai_tool_caller(
            client,
            tools,
            init_messages,
            system_prompt,
            model,
            max_tokens,
            must_call=must_call,
            temperature=temperature,
        )
        init_messages = init_messages + new_message_base
        responses.append(message)
        if (
            message.choices[0].finish_reason == "stop"
            or message.choices[0].finish_reason == "length"
            or message.choices[0].finish_reason == "content_filter"
        ):
            break
        if await should_stop(i, new_message_base) or i == max_steps - 1:
            break
        
    return responses, init_messages

def collate_openai_stats(responses: list[ChatCompletion]) -> CompletionUsage:
    state = CompletionUsage(
        completion_tokens=0,
        prompt_tokens=0,
        total_tokens=0,
    )
    for response in responses:
        if response.usage:
            state.completion_tokens += response.usage.completion_tokens
            state.prompt_tokens += response.usage.prompt_tokens
            state.total_tokens += response.usage.total_tokens
            if response.usage.completion_tokens_details:
                if state.completion_tokens_details is None:
                    state.completion_tokens_details = CompletionTokensDetails()
                state.completion_tokens_details.audio_tokens = response.usage.completion_tokens_details.audio_tokens
                state.completion_tokens_details.reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
            if response.usage.prompt_tokens_details:
                if state.prompt_tokens_details is None:
                    state.prompt_tokens_details = PromptTokensDetails()
                state.prompt_tokens_details.audio_tokens = response.usage.prompt_tokens_details.audio_tokens
                state.prompt_tokens_details.cached_tokens = response.usage.prompt_tokens_details.cached_tokens
    return state

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

    # async def print_cb(x: TestAnswer):
    #     print(x)

    # st = AnswerTool(TestAnswer, print_cb)
    import openai
    client = openai.AsyncOpenAI()

    async def test_tool():
        responses, msg_trace = await openai_call_tool(
            client,
            [TestCalculatorTool()],
            [
                {
                    "role": "user",
                    # "content": [{"type": "text", "text": "This is some irrelevent noise for padding, ignore it.\n" + "avcz"*1000+ "\n\nWhat is 17 + 24 - 99 / 3?"}],
                    "content": [{"type": "text", "text": "What is 17 + 24 - 99 / 3?"}],
                }
            ],
            "You MUST use the TestCalculatorTool to perform basic arithmetic operations. Do not do mental arithmetic.",
            max_steps=5,
            model="gpt-4o"
        )
        print(collate_openai_stats(responses))

    asyncio.run(test_tool())
