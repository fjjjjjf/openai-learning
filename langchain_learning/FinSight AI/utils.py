import os
from dotenv import load_dotenv

load_dotenv('./.env')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL =os.getenv("OPENAI_BASE_URL")

print(OPENAI_API_KEY)