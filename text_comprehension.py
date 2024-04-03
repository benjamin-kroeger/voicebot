import asyncio
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import openai
import pyaudio
from openai import OpenAI

logger = logging.getLogger(__name__)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def call_gpt_stream(model: Literal['gpt-4', 'gpt-3.5-turbo-0125'], message: str = None, queue: asyncio.Queue = None):
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": message}, ],
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            queue.put_nowait(chunk.choices[0].delta.content)
            logger.info(f'GPT: {chunk.choices[0].delta.content}')
    queue.put_nowait(None)  # Signal end of stream


def stream_response_to_speaker(sentence_input: str, player_stream) -> None:
    logger.info(f"TTS: {sentence_input}")
    with openai.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            response_format="pcm",  # similar to WAV, but without a header chunk at the start.
            input=sentence_input,
    ) as response:
        for chunk in response.iter_bytes(chunk_size=1024):
            player_stream.write(chunk)


async def main():
    player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=16) as executor:
        word_queue = asyncio.Queue()
        speech_queue = asyncio.Queue()
        # Start streaming GPT responses

        logger.info('Starting speech task')
        gpt_task = loop.run_in_executor(executor, call_gpt_stream, 'gpt-3.5-turbo-0125', "How do I brush my teeth", word_queue)

        full_response = ''
        sentence = ''
        multiplier = 1
        while True:
            word = await word_queue.get()

            if word == '.' and len(sentence) > 20 * multiplier and re.match(r'\w', sentence[-1]):
                stream_response_to_speaker(sentence, player_stream)
                multiplier += 2
                sentence = ''

            if word is None:
                stream_response_to_speaker(sentence, player_stream)
                break

            sentence += word

        logger.info(full_response)
        await gpt_task  # Wait for GPT stream task to complete
        await speech_queue.put(None)  # Signal end of speech queue


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', )
    asyncio.run(main())
