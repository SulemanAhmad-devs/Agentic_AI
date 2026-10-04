import os 
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(__file__).with_name(".env"))
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError(
        "Gemini api key not found add it"
    )
client = genai.Client(api_key=api_key)
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="explain inheritance.",
    config=types.GenerateContentConfig(
        system_instruction="You are a python teacher. Explain concepts simply for beginners."
    )
    
)
print(response.text)