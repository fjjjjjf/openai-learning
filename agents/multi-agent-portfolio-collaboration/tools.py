# ------------------------------
# 标准库
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore",category=UserWarning)
import re

# ------------------------------
# 第三方库

import pandas as pd
import requests 
from fredapi import Fred
from openai import OpenAI

# ------------------------------
# 本地包

from agents import function_tool
from utils import outputs_dir,output_file

# -------------------------------
# 仓库位置和全局

OUTPUT_DIR =outputs_dir()
PROMPT_PATH =Path(__file__).parent / "prompts" /"code_interpreter.md"
with open(PROMPT_PATH,"r",encoding="utf-8") as f:
    CODE_INTERPRETER_INSTRUCTIONS =f.read()

# --------------------------------
# 工具运行
def code_interpreter_error_handler(ctx, error):
    """
    为 run_code_interpreter 自定义错误处理程序。向 LLM 返回一条清晰的消息，说明哪里出了问题以及如何解决。
    """
    return (
        "Error running code interpreter. "
        "You must provide BOTH a clear natural language analysis request and a non-empty list of input_files (relative to outputs/). "
        f"Details: {str(error)}"
    )

@function_tool(failure_error_function=code_interpreter_error_handler)
def run_code_interpreter(request:str,input_files:list[str])->str:
    """
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

@function_tool
def write_markdown(filename: str, content: str) -> str:
    """将content写入文件中，并返回json内容"""
    if not filename.endswith(".md"):
        filename += ".md"
    path = output_file(filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return json.dumps({"file": filename})

function_tool
def read_file(filename: str, n_rows: int = 10) -> str:
    """
        读取并预览输出目录中的文件内容。
        支持读取 CSV、Markdown (.md) 和纯文本 (.txt) 文件。对于 CSV 文件，返回最后 `n_rows` 行的 Markdown 表格预览。对于 Markdown 和文本文件，返回完整的文本内容。对于不支持的文件类型，返回错误消息。
        
        Args：
        filename：要读取的文件的名称，相对于输出目录。支持的扩展名：.csv、.md、.txt。
        n_rows：要预览的 CSV 文件行数（默认值：10）。

        Return：
        str：包含以下内容之一的 JSON 字符串：
        - 对于 CSV 文件：{"file": filename, "preview_markdown": "<markdown table>"}
        - 对于 Markdown/Text{"file": filename, "content": "<text content>"}
        - 对于错误信息：{"error": "<error message>", "file": filename}
    """
    path = output_file(filename, make_parents=False)
    if not path.exists():
        return json.dumps({"error": "file not found", "file": filename})

    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        try:
            df = pd.read_csv(path).tail(n_rows)
            table_md = df.to_markdown(index=False)
            return json.dumps({"file": filename, "preview_markdown": table_md})
        except Exception as e:
            return json.dumps({"error": str(e), "file": filename})
    elif suffix == ".md" or suffix == ".txt":
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return json.dumps({"file": filename, "content": content})
        except Exception as e:
            return json.dumps({"error": str(e), "file": filename})
    else:
        return json.dumps({"error": f"Unsupported file type: {suffix}", "file": filename})

@function_tool
def get_fred_series(series_id: str, start_date: str, end_date: str, download_csv: bool = False) -> str:
    """Fetches a FRED economic time-series and returns simple summary statistics.

    Parameters
    ----------
    series_id : str
        FRED series identifier, e.g. "GDP" or "UNRATE".
    start_date : str
        ISO date string (YYYY-MM-DD).
    end_date : str
        ISO date string (YYYY-MM-DD).

    Returns
    -------
    str
        JSON string with basic statistics (mean, latest value, etc.). Falls back to a
        placeholder if fredapi is not available or an error occurs.
    """
     # Treat empty strings as unspecified
    start_date = start_date or None  # type: ignore
    end_date = end_date or None  # type: ignore

    if Fred is None:
        return json.dumps({"error": "fredapi not installed. returning stub result", "series_id": series_id})

    try:
        fred_api_key = os.getenv("FRED_API_KEY")
        fred = Fred(api_key=fred_api_key)
        data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        if data is None or data.empty:
            return json.dumps({"error": "Series not found or empty", "series_id": series_id})

        summary = {
            "series_id": series_id,
            "observations": len(data),
            "start": str(data.index.min().date()),
            "end": str(data.index.max().date()),
            "latest": float(data.iloc[-1]),
            "mean": float(data.mean()),
        }

        # ------------------------------------------------------------------
        # Optional CSV download
        # ------------------------------------------------------------------
        if download_csv:
            # Reset index to turn the DatetimeIndex into a column for CSV output
            df = data.reset_index()
            df.columns = ["Date", series_id]  # Capital D to match Yahoo Finance

            # Build date_range string for filename (YYYYMMDD-YYYYMMDD).
            start_str = start_date if start_date else str(df["Date"].min().date())
            end_str = end_date if end_date else str(df["Date"].max().date())
            date_range = f"{start_str}_{end_str}".replace("-", "")
            file_name = f"{series_id}_{date_range}.csv"

            # Save under outputs/
            csv_path = output_file(file_name)
            df.to_csv(csv_path, index=False)

            # Add file metadata to summary
            summary["file"] = file_name
            summary["schema"] = ["Date", series_id]

        return json.dumps(summary)
    except Exception as e:
        return json.dumps({"error": str(e), "series_id": series_id})
    
@function_tool
def list_output_files(extension: str = None) -> str: #找出对应的扩展名的文件，比如csv
    """
    List all files in the outputs directory. Optionally filter by file extension (e.g., 'png', 'csv', 'md').
    Returns a JSON list of filenames.
    """
    out_dir = outputs_dir()
    if extension:
        files = [f.name for f in out_dir.glob(f'*.{extension}') if f.is_file()]
    else:
        files = [f.name for f in out_dir.iterdir() if f.is_file()]
    return json.dumps({"files": files})

# 外部接口 -----------------------------------------------------------

__all__ = [
    "run_code_interpreter",
    "write_markdown",
    "get_fred_series",
    "list_output_files",
    "read_file",
] 