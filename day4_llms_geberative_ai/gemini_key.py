import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).with_name(".env"))
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Add it to day4_llms_geberative_ai/.env."
    )

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Give me 5 project ideas for a beginner learning AI."
)

print(response.text)