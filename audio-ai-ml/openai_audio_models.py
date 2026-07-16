import asyncio

from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer
from dotenv import load_dotenv

load_dotenv()
openai = AsyncOpenAI()


input = """
He says in anger: "Ha ha ha! This is so not going to go well for you"
"""

instructions = """Voice Affect: Angry like a psychopath but not overly shouting.\n\n
Tone: Deeply serious and mysterious, maintaining an undercurrent of unease throughout.\n\n
."""


async def main() -> None:
    async with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="onyx",
        input=input,
        instructions=instructions,
        response_format="pcm",
    ) as response:
        await LocalAudioPlayer().play(response)



if __name__ == "__main__":
    asyncio.run(main())
