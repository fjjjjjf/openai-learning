# OpenAI Agents Learning
在这一个小节中，主要学习了怎么使用Openai的agents框架，包括并行，多智能体同时运行，如何将agent作为工具一样进行调用

## Session Memory
 Session Memory 的核心目标是在 **上下文长度受限** 的前提下，实现以下能力：

- 保留对话的**关键上下文**
- 自动丢弃或压缩过旧信息
- 支持 **多轮对话连续性**
- 在并发/异步环境下保证数据一致性
- 为 Agent / Runner 提供干净、可控的历史输入

主要有两种方式来进行对话记忆的管理，1.进行上下文裁剪，只保留一定轮次的上下文绘画，2.使用LLM对于上下文进行总结，作为新的内容插入到对话中

### 1.TrimmingSession：基于轮次裁剪的记忆
首先是对于TrimmingSession类的定义，保留max_turns轮记忆，一轮指的是一个user 到下一个user message之间，每一个item都是一轮对话

    def __init__(self,session_id:str,max_turns:int=4):
        self.session_id =session_id
        self.max_turns = max(1,int(max_turns))
        self._items:Deque[TResponseInputItem] =deque() #双端队列
        self._lock =asyncio.Lock()  #加入异步锁，避免同时修改items

其次是对于最后几轮的裁剪，只保留最后几轮的items数据

    def _trim_to_last_turns(self,items:List[TResponseInputItem]) ->List[TResponseInputItem]:
        '''
            仅保留包含最后 `max_turns` 条用户消息的后缀，以及这些用户消息中最早的一条之后的所有内容。
            如果用户消息少于 `max_turns` 条（或没有），则保留所有条目。
        '''
        if not items:
            return items
        
        count =0
        start_idx =0

        for i in range(len(items)-1,-1,-1):  #起始为最后一个，倒着走直到第一个index=0
            if _is_user_msg(items[i]):
                count+=1
                if count== self.max_turns:
                    start_idx = i
                    break
        return items[start_idx:]

### 2.LLMSummarizer：基于摘要压缩的记忆
首先将角色的内容进行小写生成，针对的是role中的tool输出，因为输出有包括user,assistant，tool/tool_result，这里面tool的输出可能是长的json内容，因此有必要对于输出的json长序列进行压缩限制，避免其超过max_tokens。最后将多轮的历史数据提取出来，并且根据prompt进行总结，返回user_shadow和summary


    async def summarize(self,messages:List[Item])-> tuple[str,str]:
        '''
        从messages中进行总结
        returens：
        Tuple【STR，STR】:用于保持对话自然行，以及模型生成的摘要文本。
        '''
        user_shadow ="Summarize the conversation we had so far"
        TOOL_ROLES ={"tool","tool_result"}

        def to_snipper(m:Item) ->str|None:
            role =(m.get("role") or "assistant").lower()  #让内容变成小写，如果没有role则用assistant替代
            content =(m.get("content")or "").strip()
            if not content:
                return None
            if role in TOOL_ROLES and len(content) >self.tool_trim_limit:
                content =content[:self.tool_trim_limit]+" …"  #如果超过内容则去掉多余的内容
            return f"{role.upper()}: {content}"
        
        history_snippets = [s for m in messages if(s:= to_snipper(m))] # :=是特殊的赋值，表示如果返回值不是none，则将s加入列表[]

        prompt_messages =[
            {"role":"system","content":SUMMARY_PROMPT},
            {"role":"user","content":"\n".join(history_snippets)}
        ]

        resp =await asyncio.to_thread(
            self.client.response.create,
            model =self.model,
            input = prompt_messages,
            max_output_tokens = self.max_tokens
        )
        summary =resp.output_text

        await asyncio.sleep(0) #yield control

        return user_shadow,summary
    
### 3.SummarizingSession：裁剪并压缩对话
在进行新的一轮对话的时候， 先记录item，如果这个时候并未超过max_turns,则返回，如果需要总结，则先保留不会被裁剪的最后几轮，然后总结要被裁剪的几轮，生成后再返回保存起来


    async def add_items(self,items:List[Dict[str,Any]]) ->None:
        '''增加item，如果需要则总结 '''
        # 1)add item
        async with self._lock:
            for it in items:
                msg, meta =self._split_msg_and_meta(it)
                self._records.append({"msg":msg,"meta":meta})
            
            need_summary,boundary = self._summarize_decision_locked()
        
        # 2) 不总结 则仅仅标准化并退出
        if not need_summary:
            async with self._lock:
                self._normalize_synthetic_flags_locked()
            return 
        # 3） 准备总结
        async with self._lock:
            snapshot =list(self._records)
            prefix_msgs =[r["msg"]for r in snapshot[:boundary]]

        user_shadow,assistant_summary =await self._summarize(prefix_msgs)

        # 4) 再确认以及总结
        async with self._lock:
            still_need,new_boundary =self._summarize_decision_locked()
            if not still_need:
                self._normalize_synthetic_flags_locked()
                return 
            
            snapshot =list(self._records)
            suffix = snapshot[new_boundary:] #最后n论

            self._records.clear()
            self._records.extend([
                    {
                        "msg":{"role":"user","content":user_shadow},
                        "meta":{
                            "synthetic":True,
                            "kind" :"hostpry_summary_prompt",
                            "summary_for_turns": f"< all before idx {new_boundary}",
                        },
                    },
                    {
                        "msg": {"role": "assistant", "content": assistant_summary},
                        "meta": {
                            "synthetic": True,
                            "kind": "history_summary",
                            "summary_for_turns": f"< all before idx {new_boundary} >",
                        },
                    },
                ])
            self._records.extend(suffix)

            self._normalize_synthetic_flags_locked()


## Parallel Agents

对于多个agent并行使用，首先是要对于agent进行生成和定义，之后就是对于agent进行同异步处理，根据返回结果可以进行整合或者其他操作。 
</br>可以将其他agent作为另外一个agent的调用工具，既可以在函数中使用astool，或者在定义的时候就使用。

    meta_agent =Agent(
        name='MetaAgent',
        instructions='111",
        
        model_settings=ModelSettings(
        parallel_tool_calls=True
    ),
    tools= [
        features_agent.as_tool(   #这是四个不同的agent
                tool_name="features",
                tool_description="Extract the key product features from the review.",   
        ),
            pros_cons_agent.as_tool(
                tool_name="pros_cons",
                tool_description="List the pros and cons mentioned in the review.",
            ),
            sentiment_agent.as_tool(
                tool_name="sentiment",
                tool_description="Summarize the overall user sentiment from the review.",
            ),
            recommend_agent.as_tool(
                tool_name="recommend",
                tool_description="State whether you would recommend this product and why.",
            ),
    ]
    )
    ----------------------------------------------------------
    def build_editor_agent(): #另外一种方式，这是把自己作为一个function_tool
        tool_retry_instructions = load_prompt("tool_retry_prompt.md")
        editor_prompt = load_prompt("editor_base.md")
        return Agent(
            name="Memo Editor Agent",
            instructions=(editor_prompt+DISCLAIMER+tool_retry_instructions),
            tools=[write_markdown,read_file,list_output_files],
            model=default_model,
            model_settings= ModelSettings(temperature=0)
    )

    def build_memo_edit_tool(editor):
        #创建一个工具，让其他 Agent可以像调用“加法函数”一样去调用“编辑器 Agent”。
        @function_tool(
            name_override="memo_editor",
            description_override="Stitch analysis sections into a Markdown memo and save it. This is the ONLY way to generate and save the final investment report. All memos must be finalized through this tool.",
        )
        async def memo_edit_tool(ctx:RunContextWrapper,input:MemoEditorInput)->str:#运行阶段执行的。每当 LLM（大模型）决定要写报告时，它才会真正运行
            result= await Runner.run(
                starting_agent=editor,
                input=json.dumps(input.model_dump()),
                context=ctx.context,
                max_turns=40,
            )
            return result.final_output
        return memo_edit_tool 
    -------------------------------------------------------------
    Agent( #这时候memo_edit_tool就是一个tool
        name="Head Portfolio Manager Agent",
        instructions=(
            load_prompt("pm_base.md") + DISCLAIMER
        ),
        model="gpt-4.1",
        tools=[fundamental_tool, macro_tool, quant_tool, memo_edit_tool, run_all_specialists_tool],
        tool_choice="auto")
        model_settings=ModelSettings(parallel_tool_calls=True, tool_choice="auto", temperature=0)
    ) 


## multi-agent-portfolio-collaboration
这一个章节主要内容则是做一个数据分析，多个智能体进行协助，根据所提要求进行分析，总结，查询，但是学的不是很明白，
</br>首先是定义function tool

    @function_tool(failure_error_function=code_interpreter_error_handler)
    def run_code_interpreter(request:str,input_files:list[str])->str:
    """  #告诉agent使用条件
    使用 OpenAI 的代码解释器（云端）执行定量分析请求。
    
    Args:
        request (str) : 清晰、定量的分析请求，描述要对所提供的数据执行的具体计算、统计分析或可视化操作。
            Examples:
                - “计算 returns.csv 文件中投资组合收益的夏普比率。”
                - “绘制 'AAPL_returns.csv' 文件中每日收益的直方图。”
                - “对 data.csv 文件中的 'y' 与 'x' 进行线性回归，并报告 R² 值。”
                - “汇总提供的 CSV 文件中每个股票代码的波动率。”
        input_files (list[str]): 请提供一个非空的文件路径列表（相对于 outputs/ 目录），这些文件是进行分析所必需的。每个文件应包含所需定量分析的数据。
            Example: ["returns.csv", "tickers.csv"]

    Returns:
        str:JSON string 包含分析摘要和可供下载的生成文件列表（例如，图表、CSV 文件）。
    """
    # Input Validation
    if not request or not isinstance(request, str):
        raise ValueError("The 'request' argument must be a non-empty string describing the analysis to perform.")
    if not input_files or not isinstance(input_files, list) or not all(isinstance(f, str) for f in input_files):
        raise ValueError("'input_files' must be a non-empty list of file paths (strings) relative to outputs/.")

    client =OpenAI()
    file_ids =[]
    for file_path in input_files:
        abs_path =output_file(file_path,make_parents=False)
        if not abs_path.exists():
            raise ValueError(
                f"File not found: {file_path}. "
                "Use the list_output_files tool to see which files exist, "
                "and the read_file tool to see the contents of CSV files."
            )
        with abs_path.open("rb") as f:
            uploaded =client.files.create(file=f,purpose="user_data")
            file_ids.append(uploaded.id)
    
    instructions = CODE_INTERPRETER_INSTRUCTIONS

    resp = client.responses.create(
        model="gpt-4.1",
        tools=[
            {
                "type":"code_interpreter",
                "container":{"type":"auto","file_ids":file_ids}
            }
        ],
        instructions= instructions,
        input= request,
        temperature=0
    )
    
    output_text =resp.output_text

    #提取容器id
    raw =resp.model_dump() if hasattr(resp,'model_dump') else resp.__dict__ #如果对象有 model_dump 方法（Pydantic V2），则调用它；否则直接读取对象的属性字典 __dict__。这样做是为了方便后续用 ["key"] 的方式查找数据。
    container_id = None
    if "output" in raw:
        for item in raw["output"]:
            if item.get("type") == "code_interpreter_call" and "container_id" in item:
                container_id = item["container_id"]

    #下载任意新文件
    downloaded_files = []
    if container_id: #模型在沙盒里生成文件后，这些文件暂时存储在 container_id 对应的目录中。
        api_key = os.environ["OPENAI_API_KEY"]
        url = f"https://api.openai.com/v1/containers/{container_id}/files"
        headers = {"Authorization": f"Bearer {api_key}"}
        resp_files = requests.get(url, headers=headers)
         #在获得container之后，要向contariner中请求/创建容器文件
        resp_files.raise_for_status() #200 OK，4xx 抛出一个 requests.exceptions.HTTPError 异常

        files = resp_files.json().get("data", [])
        for f in files:
            # 排除了user上传的原始数据，只锁定模型运行代码后新产生的文件
            if f["source"] != "user":
                filename = f.get("path", "").split("/")[-1]
                cfile_id = f["id"]
                url_download = f"https://api.openai.com/v1/containers/{container_id}/files/{cfile_id}/content"
                resp_download = requests.get(url_download, headers=headers)
                resp_download.raise_for_status()
                out_path = output_file(filename)
                with open(out_path, "wb") as out:
                    out.write(resp_download.content)
                downloaded_files.append(str(out_path)) #将下载的二进制流写入本地文件系统，并将本地路径记录在 downloaded_files 列表中。

     # 如果没有下载到文件，检查是否有 <reason> 标签
    if not downloaded_files:
        match = re.search(r'<reason>(.*?)</reason>', output_text, re.DOTALL)
        if match:
            reason = match.group(1).strip()
            raise ValueError(reason)
        raise ValueError("No downloads were generated and no <reason> was provided. Please call the tool again, and ask for downloadable files.")

    #最后将模型的文字分析结果和本次下载的所有本地文件路径封装成 JSON 字符串返回。
    return json.dumps({
        "analysis": output_text,
        "files": downloaded_files,
    })

之后在根据不同agent的要求，是他们能调用不同的tool，以下的agent能调用四个tool

    Agent(
        name="Quantitative Analysis Agent",
        instructions=(quant_prompt + DISCLAIMER + tool_retry_instructions),
        mcp_servers=[yahoo_mcp_server],
        tools=[run_code_interpreter, get_fred_series, read_file, list_output_files],
        model=default_model,
        model_settings=ModelSettings(parallel_tool_calls=True, temperature=0),
    ) 
在将多个agent进行整合配置之后，进行工作流的制作，build_investment_agents()获得所有的agent，让其他的agent连上mcp的服务器，之后再运行pm，让其自动的调用其他智能体以及工具

    async def run_workflow():
    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError(
            "OPENAI_API_KEY not set — set it as an environment variable before running."
        )

    today_str = datetime.date.today().strftime("%B %d, %Y")
    question = (
        f"Today is {today_str}. "
        "How would the planned interest rate reduction affect my holdings in GOOGLE if they were to happen? "
        "Considering all the factors affecting its price right now (Macro, Technical, Fundamental, etc.), "
        "what is a realistic price target by the end of the year?"
    )

    bundle = build_investment_agents()

    async with AsyncExitStack() as stack:
        # 连接各 agent 的 MCP servers
        for agent in [
            getattr(bundle, "fundamental", None),
            getattr(bundle, "quant", None),
        ]:
            if agent is None:
                continue
            for server in getattr(agent, "mcp_servers", []):
                await server.connect()
                await stack.enter_async_context(server)

        print("Running multi-agent workflow...\n")
        print(f"Head PM Name: {bundle.head_pm.name}")

        response = None
        try:
            response = await asyncio.wait_for(
                Runner.run(
                    bundle.head_pm,
                    question,
                    max_turns=40,
                ),
                timeout=1200,  # 20 minutes
            )
        except asyncio.TimeoutError:
            print("\n❌ Workflow timed out after 20 minutes.")
            return

        report_path = None
        try:
            if hasattr(response, "final_output"):
                output = response.final_output
                if isinstance(output, str):
                    data = json.loads(output)
                    if isinstance(data, dict) and "file" in data:
                        report_path = output_file(data["file"])
        except Exception as e:
            print(f"Could not parse investment report path: {e}")

        print(
            "Workflow Completed.\n"
            f"Agent Response: {response.final_output if hasattr(response, 'final_output') else response}\n"
            f"Investment report created: {report_path if report_path else '[unknown]'}"
        )
