# OpenAI Function Calling

AI的函数调用方法，本质上是一个**Prompt Engineer**，我们需要自己定义一个函数，在设置一个tool的json文件，传给AI，让AI来判断是否需要调用，如果需要调用，他会进行处理，返回函数所需要的参数，tool_id，等内容，之后我们在进行处理.
而为了处理多函数的调用的tool工具的书写，可以提前处理好json文件，之后根据参数的相同与不同进行分别处理，这样比单纯的编写n个tool会方便

```
tool= {
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
    }

```
一个tool中有type,function,在function，需要name，description,parameters大致结构如下,你所定义的function name中是你已经实现的函数

```

  ChatCompletionMessage(content=None, refusal=None, role='assistant', annotations=[],audio=None, function_call=None,tool_calls=[ChatCompletionMessageFunctionToolCall(id='call_ONUbc3fomHlT5t1TCT8LiCj7', function=Function(arguments='{"location":"Beijing, China","format":"celsius"}', name='get_current_weather'), type='function')])

```
**这是正常返回内容，function中有你的参数，函数名字，以及id，此时tool_calls不为None**

```
ChatCompletionMessage(content='I can check that for you—what’s your location (city and state/country)? Also, do you prefer Fahrenheit or Celsius?', refusal=None, role='assistant', annotations=[], audio=None, function_call=None, tool_calls=None)

```
**这是返回对话的内容，此时content不为None，tool_calls为None**

后续处理对话的内容
1. 需要将返回的参数输入到函数中，获得renturn内容
2. 将返回内容处理成json文本，或者是string类型,并将该内容传入回到messages中
```
messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id, #要有tool和tool id
        'name':func_name,
        "content": json.dumps(result)
    })
``` 
**不可或缺的参数是role,tool_call_id,name,content，这样openai才能根据传入的新的messages解析并确返回数据**

## 增加一个使用function calling的知识检索demo

增加arxiv进行文件的检索以及下载，保存到本地，并且进行总结内容。优化了原OpenAI的function call写法，改成新版的 tool call



