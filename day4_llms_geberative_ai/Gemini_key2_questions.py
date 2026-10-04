import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).with_name(".env"))
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError(
        "Gemini api key is missing. Add it to folder .env"
    )
    
client = genai.Client(api_key=api_key)
while True:
    question = input("Ask gemini")
    if question.lower() == "exit":
        break
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=question
    )
    print("\n Gemini:")
    print(response.text)