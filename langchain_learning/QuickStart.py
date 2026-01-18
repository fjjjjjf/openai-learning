from langchain.agents import create_agent
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
import os
from dotenv import load_dotenv


from pydantic import BaseModel, Field
from typing import Literal

load_dotenv('./.env')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL =os.getenv("OPENAI_BASE_URL")

SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""

@tool
def get_weather_for_location(city:str)->str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

@tool
def get_user_location(runtime:ToolRuntime[Context])->str: #ToolRuntime表示工具在Agent运行是的执行环境,[context]表示runtime内部携带的上下文类型
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    #print(runtime)
    return "Florida" if user_id == "1" else "SF"

@dataclass
class ResponseFormat:
    """Response schema for the agent"""
    # A punny response (always required)
    punny_response:str
    weather_conditions:str|None=None



def get_weather1(location: str) -> str:
    """Get the weather at a location."""
    ...

class WeatherInput(BaseModel):
    """Input for weather queries."""
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result




# Access the current conversation state
@tool
def summarize_conversation(
    runtime: ToolRuntime
) -> str:
    """Summarize the conversation so far."""
    messages = runtime.state["messages"]

    human_msgs = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
    ai_msgs = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")
    tool_msgs = sum(1 for m in messages if m.__class__.__name__ == "ToolMessage")

    return f"Conversation has {human_msgs} user messages, {ai_msgs} AI responses, and {tool_msgs} tool results"

# Access custom state fields
@tool
def get_user_preference(
    pref_name: str,
    runtime: ToolRuntime  # 模型不可见的参数
) -> str:
    """Get a user preference value."""
    preferences = runtime.state.get("user_preferences", {})
    return preferences.get(pref_name, "Not set") #没有偏好则为not set


@tool
def get_weather3(city: str, runtime: ToolRuntime) -> str:
    """Get weather for a given city."""
    writer = runtime.stream_writer

    # Stream custom updates as the tool executes
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")

    return f"It's always sunny in {city}!"

if __name__=='__main__':
    llm = init_chat_model(
        model="gpt-4o-mini",
        temperature=0.2
    )

    #-------------------------------------------------------------------------
    #     checkpointer = InMemorySaver()

    #     agent = create_agent(
    #         model=llm,
    #         system_prompt=SYSTEM_PROMPT,
    #         tools=[get_user_location,get_weather_for_location],
    #         context_schema=Context,#运行期上下文如何在多轮、多 tool 中共享
    #         response_format=ToolStrategy(ResponseFormat),#模型最终必须按什么结构返回结果
    #         checkpointer=checkpointer#对话状态如何持久化 / 恢复 / 继续执行
    #     )
    #     # `thread_id` is a unique identifier for a given conversation.
    #     config={"configurable":{"thread_id":"1"}}

    #     response =agent.invoke(
    #         {"messages":[{"role":"user","content":"what is the weather outside?"}]},
    #         config=config,
    #         context=Context(user_id="1")
    #     )
    #     print(response)
    #     print(response['structured_response'])

    #     response = agent.invoke(
    #     {"messages": [{"role": "user", "content": "thank you!"}]},
    #     config=config,
    #     context=Context(user_id="1")
    # )
    #     print(response['structured_response'])

    # -------------------------------------------------------------------------

    
    # model_with_tools = llm.bind_tools([get_weather])
    # response = model_with_tools.invoke("What's the weather in Paris ?")

    # print(response)
    # for tool_call in response.tool_calls:
    #     print(f"Tool: {tool_call['name']}")
    #     print(f"Args: {tool_call['args']}")
    #     print(f"ID: {tool_call['id']}")


    # -------------------------------------------------------------------------

    agent = create_agent(
            model=llm,
            system_prompt=SYSTEM_PROMPT,
            tools=[get_weather3],
            context_schema=Context,#运行期上下文如何在多轮、多 tool 中共享
            response_format=ToolStrategy(ResponseFormat),#模型最终必须按什么结构返回结果

        )
    response =agent.invoke(
            {"messages":[{"role":"user","content":"what is the weather outside in Paris use celsuis?"}]},
            context=Context(user_id="1")
        )
    print(response)