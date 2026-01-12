from __future__ import annotations
from openai import OpenAI
from collections import deque
from typing import Any,Deque,List,cast,Optional,Tuple,Dict
import os
from dotenv import load_dotenv
from agents import set_tracing_disabled
from agents import Agent,Runner
from agents.memory.session import SessionABC #Abstract Base Class
from agents.items import TResponseInputItem 
import asyncio

load_dotenv('./.env')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL =os.getenv("OPENAI_BASE_URL")

ROLE_USER ="user"
client =OpenAI()

def _is_user_msg(item:TResponseInputItem)->bool:
    """兼容两种sdk，有直接返回{"role": "..."}，也有{"type": "message", "role": "..."}"""
    if isinstance(item,dict):  #判断item是否是dict类型
        role =item.get("role")
        if role is not None:
            return role==ROLE_USER
        #openai 中TResponseInputItem是一个类型集合，ResponseMessage,ToolCall,dict,str,中的一种
        if item.get("type")=="message":
            return item.get("role") == ROLE_USER
    
    return getattr(item,"role",None) ==ROLE_USER

class TrimmingSession(SessionABC):
    '''
    保留最后n轮记忆，一轮指的是user 到下一个user message之间
    '''
    def __init__(self,session_id:str,max_turns:int=4):
        self.session_id =session_id
        self.max_turns = max(1,int(max_turns))
        self._items:Deque[TResponseInputItem] =deque() #双端队列
        self._lock =asyncio.Lock()  #加入异步锁，避免同时修改items

    # ---- SessionABC API -----

    async def get_items(self, limit: int | None = None) -> List[TResponseInputItem]:
        async with self._lock:
            trimmed =self._trim_to_last_turns(list(self._items))
            return trimmed[-limit:] if (limit is not None and limit >=0) else trimmed
        
    async def add_items(self,items:List[TResponseInputItem])-> None:
        if not items:
            return
        async with self._lock:
            self._items.extend(items)
            trimmed = self._trim_to_last_turns(list(self._items))
            self._items.clear()
            self._items.extend(trimmed)
        
    async def pop_item(self) -> TResponseInputItem | None:
        """Remove and return the most recent item (post-trim)."""
        async with self._lock:
            return self._items.pop() if self._items else None
        
    async def clear_session(self) -> None:
        """Remove all items for this session."""
        async with self._lock:
            self._items.clear()

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
    
    async def set_max_turns(self, max_turns: int) -> None:
        async with self._lock:
            self.max_turns = max(1, int(max_turns))
            trimmed = self._trim_to_last_turns(list(self._items))
            self._items.clear()
            self._items.extend(trimmed)

    async def raw_items(self) -> List[TResponseInputItem]:
        """Return the untrimmed in-memory log (for debugging)."""
        async with self._lock:
            return list(self._items)
        

SUMMARY_PROMPT = """
You are a senior customer-support assistant for tech devices, setup, and software issues.
Compress the earlier conversation into a precise, reusable snapshot for future turns.

Before you write (do this silently):
- Contradiction check: compare user claims with system instructions and tool definitions/logs; note any conflicts or reversals.
- Temporal ordering: sort key events by time; the most recent update wins. If timestamps exist, keep them.
- Hallucination control: if any fact is uncertain/not stated, mark it as UNVERIFIED rather than guessing.

Write a structured, factual summary ≤ 200 words using the sections below (use the exact headings):

• Product & Environment:
  - Device/model, OS/app versions, network/context if mentioned.

• Reported Issue:
  - Single-sentence problem statement (latest state).

• Steps Tried & Results:
  - Chronological bullets (include tool calls + outcomes, errors, codes).

• Identifiers:
  - Ticket #, device serial/model, account/email (only if provided).

• Timeline Milestones:
  - Key events with timestamps or relative order (e.g., 10:32 install → 10:41 error).

• Tool Performance Insights:
  - What tool calls worked/failed and why (if evident).

• Current Status & Blockers:
  - What’s resolved vs pending; explicit blockers preventing progress.

• Next Recommended Step:
  - One concrete action (or two alternatives) aligned with policies/tools.

Rules:
- Be concise, no fluff; use short bullets, verbs first.
- Do not invent new facts; quote error strings/codes exactly when available.
- If previous info was superseded, note “Superseded:” and omit details unless critical.
"""

class LLMSummarizer:
    def __init__(self,client,model='gpt-4o',max_tokens=400,tool_trim_limit =600):
        self.client =client
        self.model= model
        self.max_tokens = max_tokens
        self.tool_trim_limit =tool_trim_limit
    
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

Record = Dict[str,Dict[str,Any]]

class SummarizingSession:
    """
    保存前n论用户对话，如果超出context_limits ,则会进行总结
    """
    _ALLOWED_MSG_KEYS ={"role","content","name"}

    def __init__(self,
                 keep_last_n_turns:int=3,
                 context_limits :int =3,
                 summaeizer:Optional["Summarizer"]= None,
                 session_id :Optional["str"]=None
                 ):
        assert context_limits>=1
        assert keep_last_n_turns>=0
        assert keep_last_n_turns <=context_limits,"keep_last_n_turns should not be greater than context_limit"

        self.keep_last_n_turns =keep_last_n_turns
        self.context_limit =context_limits
        self.summarizer =summaeizer
        self.session_id =session_id or "default"

        self._records :deque[Record] =deque()
        self._lock =asyncio.Lock()
        
    # --------- public API used by runner -------------
    async def get_items(self,limit:Optional[int] =None) -> List[Dict[str,Any]]:
        ''' 返回 安全的信息'''
        async with self._lock:
            data =list(self._records)
        msgs = [self._sanitize_for_model(rec["msg"]) for rec in data]
        return msgs[-limit:] if limit else msgs

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

async def main():
    set_tracing_disabled(True)
    
    # agent =Agent(
    #     name="Assistant",
    #     instructions="Reply very concisely"
    # )

    # support_agent = Agent(
    #     name="Customer Support Assistant",
    #     model="gpt-5",
    #     instructions=(
    #         "You are a patient, step-by-step IT support assistant. "
    #         "Your role is to help customers troubleshoot and resolve issues with devices and software. "
    #         "Guidelines:\n"
    #         "- Be concise and use numbered steps where possible.\n"
    #         "- Ask only one focused, clarifying question at a time before suggesting next actions.\n"
    #         "- Track and remember multiple issues across the conversation; update your understanding as new problems emerge.\n"
    #         "- When a problem is resolved, briefly confirm closure before moving to the next.\n"
    #     )
    # )
    # session =TrimmingSession("my_session",max_turns=3)
    # message = "There is a red light blinking on my laptop"

    # result =await Runner.run(

    #     support_agent,
    #     message,
    #     session=session
    # )
    # history =await session.get_items()

    # print(history)



asyncio.run(main())

