import json
import os
from openai import OpenAI
from tenacity import retry ,wait_random_exponential,stop_after_attempt
from termcolor import colored
from dotenv import load_dotenv


load_dotenv('./.env')

GPT_MODEL ='gpt-5'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL =os.getenv("OPENAI_BASE_URL")
client =OpenAI()


@retry(wait=wait_random_exponential(multiplier=1,max=40),stop= stop_after_attempt(5))
def chat_completion_request(messages,tools=None,tool_choice=None,model=GPT_MODEL):
    try:
        response =client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

def pretty_print_conversation(messages):
    role_to_color={
        "system":"red",
        "user":"green",
        "assistant":"blue",
        "function":"magenta",
    }
    for message in messages:
        if messages["role"]=="system":
            print(colored(f"system: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "user":
            print(colored(f"user: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and message.get("function_call"):
            print(colored(f"assistant: {message['function_call']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and not message.get("function_call"):
            print(colored(f"assistant: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "function":
            print(colored(f"function ({message['name']}): {message['content']}\n", role_to_color[message["role"]]))

def get_current_weather(location:str,format:str):
    
    if format =='celsius':
        weather ='20 celsius'
    else :
        weather = "38 fahrenheit"
    
    print(weather)
    return {"weather": weather}
    

def get_n_day_weather_forecast(location:str,format:str,num_days:int):
    
    print (f"{location} next {num_days} is 20 {format}")
    return {'weather':f'{location} next {num_days} is 20 {format}'}
if __name__ =='__main__':

    tools=[
        {
            "type":'function',
            "function":{
                "name":"get_current_weather",
                "description": "Get the current weather",
                "parameters":{
                    "type":"object",
                    "properties":{
                        "location":{
                            "type":"string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format":{
                            "type":"string",
                            "enum":["celsius","fahrenheit"],
                            "description": "The temperature unit to use. Infer this from the users location.",
                        },
                    },
                    "required": ["location", "format"],
                }
            }
        
        },
        {
        "type": "function",
        "function": {
            "name": "get_n_day_weather_forecast",
            "description": "Get an N-day weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "The number of days to forecast",
                    }
                },
                "required": ["location", "format", "num_days"]
            },
        }
    },
    ]
    
    messages =[]
    messages.append({"role":"system","content":"Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
    messages.append({"role": "user", "content": "What's the weather like today in beijing use celsius"})
    chat_response = chat_completion_request(messages,tools=tools)

    assistant_message =chat_response.choices[0].message
    messages.append(assistant_message)

    #print(assistant_message)

    #并不会调用自己的函数，只是告诉你更适合用函数
    # ChatCompletionMessage(content=None, refusal=None, role='assistant', annotations=[], 
    # audio=None, function_call=None, 
    # tool_calls=[ChatCompletionMessageFunctionToolCall(id='call_ONUbc3fomHlT5t1TCT8LiCj7', function=Function(arguments='{"location":"Beijing, China","format":"celsius"}', name='get_current_weather'), type='function')])

    #要是返回对话
    #ChatCompletionMessage(content='I can check that for you—what’s your location (city and state/country)? Also, do you prefer Fahrenheit or Celsius?', refusal=None, 
    # role='assistant', annotations=[], audio=None, function_call=None, tool_calls=None)

    if(assistant_message.tool_calls):
        for tool_call in assistant_message.tool_calls:
            func_name =tool_call.function.name
            arguments =json.loads(tool_call.function.arguments)
            if func_name == "get_current_weather":
                result = get_current_weather(**arguments)
            elif func_name == "get_n_day_weather_forecast":
                result = get_n_day_weather_forecast(**arguments)
            else:
                result = {"error": "unknown function"}

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id, #要有tool和tool id
            'name':func_name,
            "content": json.dumps(result)
        })
        print(messages)
    
    print(chat_completion_request(messages,tools=tools))
    