import arxiv
import ast
import concurrent
import json
import os
import pandas as pd
import tiktoken
from csv import writer
from openai import OpenAI
from PyPDF2 import PdfReader
from scipy import spatial
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm
from termcolor import colored
from dotenv import load_dotenv


load_dotenv('./.env')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL =os.getenv("OPENAI_BASE_URL")
GPT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-ada-002"

paper_dir_filepath ='data/papers/arxiv_library.csv'


def create_folder():
    directory = 'data/papers'
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully")
    else :
        print(f"Directory '{directory}' already exists")


    if not os.path.exists(paper_dir_filepath):
        df =pd.DataFrame(list())
        df.to_csv(paper_dir_filepath)

@retry(wait=wait_random_exponential(min=1,max=40),stop=stop_after_attempt(3))
def embedding_request(text):
    response =client.embeddings.create(input=text,model=EMBEDDING_MODEL)
    return response

@retry(wait=wait_random_exponential(min=1,max=40),stop=stop_after_attempt(3))
def get_articles(query,library=paper_dir_filepath,top_k=3):
    client =arxiv.Client()
    search =arxiv.Search(
        query=query,
        max_results=top_k
    ) #表示查询条件
    result_list=[]
    
    data_dir =os.path.join(os.curdir,'data','papers')

    for result in client.results(search):
        result_dict={}
        result_dict.update({'title':result.title})
        result_dict.update({'summary':result.summary})

        result_dict.update({"article_url": [x.href for x in result.links][0]})
        result_dict.update({"pdf_url": [x.href for x in result.links][1]})
        result_list.append(result_dict)

        response = embedding_request(text=result.title)
        file_reference =[
            result.title,
            result.download_pdf(data_dir),
            response.data[0].embedding
        ]

        with open(library,'a') as f_object:
            writer_objecte =writer(f_object)
            writer_objecte.writerow(file_reference)
            f_object.close()

        
    return result_list

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages,tools=None,model=GPT_MODEL):
    try:
        response =client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools

        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

class Conversation:
    def __init__(self):
        self.conversation_history = []

    def add_message(self, role, content):
        message = {"role": role, "content": content}
        self.conversation_history.append(message)

    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        for message in self.conversation_history:
            print(
                colored(
                    f"{message['role']}: {message['content']}\n\n",
                    role_to_color[message["role"]],
                )
            )

def strings_ranked_by_relatedness(
        query:str,
        df:pd.DataFrame,  #保存之前的csv
        relatedness_fn =lambda x ,y:1-spatial.distance.cosine(x,y), #判断相似性
        top_n=100,
) ->list[str]:
    query_embedding_response =embedding_request(query)
    query_embedding =query_embedding_response.data[0].embedding
    strings_and_relatednesses =[
        (row['filepath'],relatedness_fn(query_embedding,row['embedding']))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x:x[1],reverse=True) #按照从达到小排列
    strings,relatednesses =zip(*strings_and_relatednesses) #返回pdf文件列表

    return strings[:top_n]

def read_pdf(filepath):
    # creating a pdf reader object
    reader = PdfReader(filepath)
    pdf_text = ""
    page_number = 0
    for page in reader.pages:
        page_number += 1
        pdf_text += page.extract_text() + f"\nPage Number: {page_number}"
    return pdf_text


def create_chunks(text,n,tokenizer):
    tokens =tokenizer.encode(text)
    i=0
    while i<len(tokens):
        j = min(i+int (1.5*n),len(tokens))
        while j>i+int(0.5*n):
            chunk =tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j-=1
        if j==i+int(0.5*n):
            j = min (i+n,len(tokens))
        yield tokens[i:j]
        i=j

def extract_chunk(content,template_prompt):
    prompt = template_prompt+content
    response =client.chat.completions.create(
        model= GPT_MODEL,messages=[{"role":"user","content":prompt}],temperature=0
    )
    return response.choices[0].message.content

def summarize_text(query):
    '''
    阅读csv文件，查询相似度最大的，然后将其分片总结，最后返回给用户
    '''

    summary_prompt ='''Summarize this text from an academic.Extract any key points with reasoning.\n\n COntent:'''

    print(paper_dir_filepath)
    library_df =pd.read_csv(paper_dir_filepath).reset_index()
    if len(library_df) == 0 :
        print('No papers searched yet ,downloading first.')
        get_articles(query)
        print('papers downloaded,continuing')
        library_df =pd.read_csv(paper_dir_filepath).reset_index()
    else :
        print('Existing papers found... Articles',len(library_df))

    library_df.columns = ['title','filepath','embedding']
    library_df['embedding'] =  library_df['embedding'].apply(ast.literal_eval)
    strings =strings_ranked_by_relatedness(query,library_df,top_n=1)
    print("Chunking text from paper")

    pdf_text = read_pdf(strings[0])

    tokenizer =tiktoken.get_encoding('cl100k_base')
    results=''

    chunks =create_chunks(pdf_text,1500,tokenizer)
    text_chunks =[tokenizer.decode(chunk) for chunk in chunks]
    print("Summarizing each chunk of text")

    with concurrent.futures.ThreadPoolExecutor(  #多线程，对每个text——chunk执行extract_chunk，使用tqdm显示进度
        max_workers =len(text_chunks)
    ) as executor:
        futures =[
            executor.submit(extract_chunk,chunk,summary_prompt)  #提交任务到线程池
            for chunk in text_chunks
        ]
        with tqdm(total= len(text_chunks)) as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)
        for future in futures:
            data =future.result() 
            results+=data  #汇总所有的结果

    print("Summarizing into overall summary")
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "user",
                "content": f"""Write a summary collated from this collection of key points extracted from an academic paper.
                        The summary should highlight the core argument, conclusions and evidence, and answer the user's query.
                        User query: {query}
                        The summary should be structured in bulleted lists following the headings Core Argument, Evidence, and Conclusions.
                        Key points:\n{results}\nSummary:\n""",
            }
        ],
        temperature=0,
    )
    return response

def chat_completion_with_function_execution(messages, functions=[None]):
    """This function makes a ChatCompletion API call with the option of adding functions"""
    response = chat_completion_request(messages, functions)
    full_message = response.choices[0]
    #print(full_message)
    if full_message.finish_reason == "tool_calls":
        print(f"Function generation requested, calling function")
        messages.append(
            {
                "role":"assistant",
                "content":None,
                "tool_calls":full_message.message.tool_calls
            }
        )
        return call_arxiv_function(messages, full_message)
    else:
        print(f"Function not required, responding to user")
        return response

def call_arxiv_function(messages, full_message):
    """Function calling function which executes function calls when the model believes it is necessary.
    Currently extended by adding clauses to this if statement."""

    if full_message.message.tool_calls[0].function.name == "get_articles":
        try:
            parsed_output = json.loads(
                full_message.message.tool_calls[0].function.arguments
            )
            print("Getting search results")
            results = get_articles(parsed_output["query"])
        except Exception as e:
            print(parsed_output)
            print(f"Function execution failed")
            print(f"Error message: {e}")
        
        print('full :' ,full_message)
        messages.append(
            {
                "role": "tool",
                #"name": full_message.message.tool_calls[0].function.name,
                "content": str(results),
                "tool_call_id":full_message.message.tool_calls[0].id
            }
        )
        try:
            print("Got search results, summarizing content")
            response = chat_completion_request(messages)
            return response
        except Exception as e:
            print(type(e))
            raise Exception("Function chat request failed")

    elif (
        full_message.message.tool_calls[0].function.name == "read_article_and_summarize"
    ):
        parsed_output = json.loads(
            full_message.message.tool_calls[0].function.arguments
        )
        print("Finding and reading paper")
        summary = summarize_text(parsed_output["query"])
        return summary

    else:
        raise Exception("Function does not exist and cannot be called")


if __name__=='__main__':

    client = OpenAI()
    create_folder()
    #get_articles('agent')
    #chat_test_response = summarize_text("agent LLM")
    #print(chat_test_response.choices[0].message.content)

    arxiv_functions = [
        {
            "type":"function",
            "function":{
                "name": "get_articles",
                "description": """Use this function to get academic papers from arXiv to answer user questions.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": f"""
                                    User query in JSON. Responses should be summarized and should include the article URL reference
                                    """,
                        }
                    },
                    "required": ["query"],
                },
            }
        },
        {
            "type":"function",
            "function":{
                "name": "read_article_and_summarize",
                "description": """Use this function to read whole papers and provide a summary for users.
                You should NEVER call this function before get_articles has been called in the conversation.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": f"""
                                    Description of the article in plain text based on the user's query
                                    """,
                        }
                    },
                    "required": ["query"],
                },
            }
        }
    ]

    paper_system_message = """You are arXivGPT, a helpful assistant pulls academic papers to answer user questions.
    You summarize the papers clearly so the customer can decide which to read to answer their question.
    You always provide the article_url and title so the user can understand the name of the paper and click through to access it.
    Begin!"""
    paper_conversation = Conversation()
    paper_conversation.add_message("system", paper_system_message)

    paper_conversation.add_message("user", "Hi, how does agent learning work?")
    chat_response = chat_completion_with_function_execution(
    paper_conversation.conversation_history, functions=arxiv_functions
    )
    assistant_message = chat_response.choices[0].message.content
    paper_conversation.add_message("assistant", assistant_message)


    print(assistant_message)