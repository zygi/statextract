import copy
import typing
import asyncio
from anthropic import AsyncAnthropic
from rich import print

import anthropic.types as atypes
# import statextract.agent.prompt_caching_wrapper as atypes
from statextract.agent.tools import TestCalculatorTool, Tool
from anthropic.types.beta.prompt_caching.prompt_caching_beta_message import PromptCachingBetaMessage

CacheControlable = atypes.MessageParam | atypes.ToolParam

def append_cache_control_message(dct: atypes.MessageParam, enabled: bool) -> atypes.MessageParam:
    dct = copy.deepcopy(dct)
    if enabled and isinstance(dct["content"], list):
        for c in dct["content"]:
            c["cache_control"] = {"type": "ephemeral"}  # type: ignore
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
        tool_choice=must_call_arg,
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


async def _false(x: int, y: list[atypes.MessageParam]) -> bool:
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
) -> tuple[list[_AnthropicMessage], list[atypes.MessageParam]]:
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

    client = AsyncAnthropic(
    )

    async def test_tool():
        res = await anthropic_call_tool(
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
            max_steps=1,
            prompt_caching=True,
        )
        # print(res)

    asyncio.run(test_tool())
