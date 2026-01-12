from __future__ import annotations

"""多智能体投资工作流程的共享实用程序。"""

from pathlib import Path
import json

from agents.tracing.processor_interface import TracingExporter

# ------------------------------------------------
# agent的免责声明

DISCLAIMER=(
    "DISCLAIMER: I am an AI language model, not a registered investment adviser. "
    "Information provided is educational and general in nature. Consult a qualified "
    "financial professional before making any investment decisions.\n\n"
)

# ------------------------------------------------
# 路径

ROOT_DIR: Path = Path(__file__).resolve().parent

def repo_path (rel:str |Path) ->Path:
    """返回某一个文件的绝对路径   """
    return (ROOT_DIR/rel).resolve()

def outputs_dir() ->Path:
    """输出到 outputs文件夹，如果需要则创建"""
    out =repo_path("outputs")
    out.mkdir(parents=True,exist_ok=True)

    return out

# -------------------------------------------------
# 提示词加载

PROMPTS_DIR :Path = repo_path("prompts")

def load_prompt(name:str, **subs)->str:
    """加载markdown中的提示次模板并且替换为占位符"""
    content =(PROMPTS_DIR/name).read_text()
    for key ,val in subs.items():
        content =content .replace(f"<{key}>",str(val))
    return content

# -------------------------------------------------
# 本地追踪导出

class FileSpanExporter(TracingExporter):
    """将 span/traces 写入 `logs/` 下的 JSONL 文件。"""

    def __init__(self,logfile:str|Path ="logs/agent_traces.jsonl")->None:
        path = repo_path(logfile)
        path.parent.mkdir(parents=True,exist_ok=True)
        self.logfile =path
    
    def export(self,items):
        with self.logfile.open("a", encoding="utf-8") as f: #a 追加模式。它保证了新产生的数据会写在文件末尾，而不会覆盖掉之前记录的日志。
            for item in items:
                try:
                    f.write(json.dumps(item.export(),default=str)+ "\n") #item.export(): 调用对象自身的 export 方法，将复杂的追踪对象转换成 Python 的**字典（dict）**格式。
                except Exception:
                    f.write(str(item)+ "\n")

# ---------------------------------------------------
# 输出路径帮助

def output_file(name:str|Path,*,make_parents:bool=True) ->Path:
    """
        返回共享的 outputs/ 目录下的绝对路径。
        如果 *name* 已经以字符串“outputs/”开头，则会移除该前缀，
        以避免意外嵌套第二个 outputs 文件夹（例如：
        `outputs/outputs/foo.png`）。绝对路径将保持不变地返回。
    """
    path=Path(name)

    if path.is_absolute():
        return
    
    # Strip leading "outputs/" if present
    if path.parts and path.parts[0] =="outputs":
        path =Path(*path.parts[1:])

    final =outputs_dir()/path
    if make_parents:
        final.parent.mkdir(parents=True,exist_ok=True)

    return final

# mkdir 创建一个新的文件夹（目录）,parents=True允许逐层创建，exist_ok=True如果月月一已存在则不报错
__all__ = [
    "ROOT_DIR",
    "repo_path",
    "outputs_dir",
    "load_prompt",
    "FileSpanExporter",
    "output_file",
] 