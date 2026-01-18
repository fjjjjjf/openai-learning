from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, AIMessage
load_dotenv('./.env')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL =os.getenv("OPENAI_BASE_URL")

@tool
def read_email_tool(email_id: str) -> str:
    """Mock function to read an email by its ID."""
    return f"Email content for ID: {email_id}"

def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Mock function to send an email."""
    return f"Email sent to {recipient} with subject '{subject}'"

config = {"configurable": {"thread_id": "thread-1"}}

agent = create_agent(
    model="gpt-4o",
    tools=[read_email_tool, send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": True,
                "read_email_tool": False,
            }
        ),
    ],
)
inputs = {"messages": [("user", "帮我读取 ID 为 123 的邮件，并回复给 alice@example.com 说收到")]}

"""
for event in agent.stream(inputs, config=config):
    # event 的格式通常是 {'agent': {...}} 或 {'tools': {...}}
    for node_name, value in event.items():
        print(f"\n[节点: {node_name}]")
        print("-------------------")
        
        # 获取消息列表中的最后一条
        if "messages" in value:
            last_msg = value["messages"][-1]
            
            # 情况 A: 模型决定调用工具
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for tool_call in last_msg.tool_calls:
                    print(f"🤖 模型打算调用工具: {tool_call['name']}")
                    print(f"📦 参数内容: {tool_call['args']}")
            
            # 情况 B: 模型输出普通文本
            elif hasattr(last_msg, "content") and last_msg.content:
                print(f"🤖 模型回复: {last_msg.content}")
        else:
            print(value)

print("\n--- ⏸️ 运行已暂停 (因为命中了 send_email_tool 的中断) ---")
"""

for event in agent.stream(inputs, config=config):
    pass 
snapshot = agent.get_state(config)
if not snapshot.next:
    print("没有触发中断，任务可能已完成。")
    exit()
last_message = snapshot.values["messages"][-1]
tool_call = last_message.tool_calls[0]
tool_call_id = tool_call["id"]
current_args = tool_call["args"]

print("\n" + "="*40)
print(last_message)
print(f"🛑 拦截到工具调用: {tool_call['name']}")
print(f"📝 预生成的邮件内容:\n{current_args.get('body')}")
print("="*40 + "\n")

while True:
    choice = input("👉 请选择操作: [a]批准 / [e]修改 / [r]拒绝: ").lower().strip()

    if choice == 'a':
        # === Approve (批准) ===
        print("✅ 已批准，继续执行...")
        # 传入 None 表示“继续之前中断的地方”
        result_stream = agent.stream(None, config=config)
        break

    elif choice == 'e':
        # === Edit (修改) ===
        new_body = input("⌨️  请输入新的邮件正文: ")
        
        # 1. 复制当前的参数并修改
        new_args = current_args.copy()
        new_args['body'] = new_body
        
        # 2. 构造新的工具调用对象
        new_tool_call = tool_call.copy()
        new_tool_call['args'] = new_args
        
        # 3. 构造新的 AIMessage，必须使用相同的 id 来覆盖旧消息
        # 注意：LangGraph 更新状态时，如果消息 ID 相同，会执行替换操作
        updated_message = AIMessage(
            content=last_message.content,
            tool_calls=[new_tool_call],
            id=last_message.id  # 关键：保持 ID 不变
        )
        print(updated_message)
        # 4. 更新状态
        agent.update_state(config, {"messages": [updated_message]})
        
        print("✏️  参数已修改，继续执行...")
        result_stream = agent.stream(None, config=config)
        break

    elif choice == 'r':
        # === Reject (拒绝) ===
        reason = input("❌ 请输入拒绝原因 (告诉模型为什么要拒绝): ")
        
        # 拒绝的本质是：我们不运行真实工具，而是手动插入一个“工具输出结果”
        # 这个结果告诉模型“用户禁止了此操作”
        
        rejection_message = ToolMessage(
            tool_call_id=tool_call_id,
            content=f"Error: User blocked the email execution. Reason: {reason}"
        )
        
        # 我们插入这条 ToolMessage，相当于告诉模型工具已经“跑完了”（虽然其实没跑真实逻辑）
        # as_node="send_email_tool" 伪装成是这个节点产生的输出
        agent.update_state(config, {"messages": [rejection_message]}, as_node="send_email_tool")
        
        print("🚫 已拒绝，正在通知模型...")
        # 继续运行，模型会收到 ToolMessage 并根据报错信息生成新的回复
        result_stream = agent.stream(None, config=config)
        break
