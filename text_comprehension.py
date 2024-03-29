import logging
import os
import asyncio
import timeit
from typing import Literal, Iterator
from openai import OpenAI
from io import BytesIO
import simpleaudio as sa
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


async def call_gpt_stream(model: Literal['gpt-4', 'gpt-3.5-turbo-0125'], message: str = None, queue: asyncio.Queue = None):
    def stream_messages():
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": message}, ],
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                queue.put_nowait(chunk.choices[0].delta.content)
        queue.put_nowait(None)  # Signal end of stream

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        await loop.run_in_executor(executor, stream_messages)


def create_speech_async(sentence: str):
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=sentence,
        response_format='wav'
    )
    return BytesIO(response.content)


def play_audio(file_name):
    logger.debug('Trying to play audio')
    wave_obj = sa.WaveObject.from_wave_file(file_name)
    play_obj = wave_obj.play()
    play_obj.wait_done()


async def main():
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=16) as executor:
        queue = asyncio.Queue()
        logger.info('starting gpt readint')
        # Start streaming GPT responses
        gpt_task = loop.create_task(call_gpt_stream(model='gpt-3.5-turbo-0125', message="How do I brush my teeth, in one sentence", queue=queue))
        logger.info('finished')
        full_response = ''
        sentence = ''
        while True:
            word = await queue.get()
            if word is None:
                break
            logger.info(word)
            sentence += word
            full_response += word
            if word == '.':
                task = loop.run_in_executor(executor, create_speech_async, sentence)
                audio_bytes = await task
                play_audio(audio_bytes)
                sentence = ''

        logger.info(full_response)
        await gpt_task  # Wait for GPT stream task to complete


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',)
    asyncio.run(main())
