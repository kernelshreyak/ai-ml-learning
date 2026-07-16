from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.environ['RUNPOD_API_KEY'],
    base_url=f"https://api.runpod.ai/v2/{os.environ['RUNPOD_ENDPOINT_ID']}/openai/v1"
)


stream = client.chat.completions.create(
    model="openchat/openchat-3.5-0106",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What happens when a human body burns. Describe stages and effects on body"}
    ],
    temperature=0.7,
    max_tokens=1024,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print("\n")
