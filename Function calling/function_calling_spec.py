import os 
import json
import jsonref
from openai import OpenAI
from pprint import pp
from dotenv import load_dotenv


load_dotenv('./.env')

GPT_MODEL ='gpt-5'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL =os.getenv("OPENAI_BASE_URL")
client =OpenAI()

MAX_CALLS = 5

SYSTEM_MESSAGE = """
You are a helpful assistant.
Respond to the following prompt by using function_call and then summarize actions.
Ask for clarification if a user request is ambiguous.
"""

#step 1 解析json文件，使用格式化json文件的优点在于避免在函数过多时候，每个都定义一遍
def openai_to_functions(openapi_spec):
    functions =[]
    for path ,methods in openapi_spec['paths'].items():
        for method,spec_with_ref in methods.items():
        # 1.处理json引用
            spec =jsonref.replace_refs(spec_with_ref)
        # 2.提取函数名称.
            function_name = spec.get("operationId")  
        # 3.提取description和parameters
            desc =spec.get('description') or spec.get('summary',"")
            
            schema ={'type':'object','properties':{}}

            req_body =(
                spec.get("requestBody",{})
                .get("content",{})
                .get("application/json",{})
                .get("schema")
            )  #一般是post
            if req_body:
                schema["properties"]["requestBody"] = req_body #如果存在req，则将其添加进入properties
            params =spec.get('parameters',[])
            
            if params:
                param_properties={
                    param["name"]: param["schema"]
                    for param in params
                    if "schema" in param
                }
                schema["properties"]["parameters"]={
                    'type':'object',
                    'properties':param_properties,
                }
            
            functions.append(
                {'type':'function','function':{'name':function_name,"description":desc,"parameters":schema}}

            )
    return functions

def get_openai_response(functions, messages):
    return client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        tools=functions,
        tool_choice="auto",  # "auto" means the model can pick between generating a message or calling a function.
        temperature=0,
        messages=messages,
    )

def process_user_instruction(functions, instruction):
    num_calls = 0
    messages = [
        {"content": SYSTEM_MESSAGE, "role": "system"},
        {"content": instruction, "role": "user"},
    ]

    while num_calls < MAX_CALLS:
        response = get_openai_response(functions, messages)
        message = response.choices[0].message
        print(message)
        try:
            print(f"\n>> Function call #: {num_calls + 1}\n")
            pp(message.tool_calls)   #没有调用则会报错
            messages.append(message)

            #假装调用了这些函数

            messages.append(
                {
                    "role": "tool",
                    "content": "success",
                    "tool_call_id": message.tool_calls[0].id,
                    "name":message.tool_calls[0].function.name
                }
            )

            num_calls += 1
        except:
            print("\n>> Message:\n")
            print(message.content)
            break

    if num_calls >= MAX_CALLS:
        print(f"Reached max chained function calls: {MAX_CALLS}")

if __name__ =='__main__':
    with open('Function calling/example_events_openapi.json','r') as f:
        openapi_spec =jsonref.loads(f.read())
        
    #print(openapi_spec)
    
    functions =openai_to_functions(openapi_spec)
    # for function in functions:
    #     print(function)

    USER_INSTRUCTION = """ 
    Instruction: Get all the events.
    Then create a new event named AGI Party.
    Then delete event with id 2456.
    One by one
    """

    process_user_instruction(functions,USER_INSTRUCTION)
    