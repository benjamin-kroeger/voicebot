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


def call_gpt_stream(model: Literal['gpt-4', 'gpt-3.5-turbo-0125'], message: str = None, queue: asyncio.Queue = None):
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": message}, ],
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            queue.put_nowait(chunk.choices[0].delta.content)
    queue.put_nowait(None)  # Signal end of stream


def create_speech_async(sentence: str, queue: asyncio.Queue = None):
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=sentence,
        response_format='wav'
    )
    queue.put_nowait((sentence,BytesIO(response.content)))


async def check_speech_queue(speech_queue: asyncio.Queue):
    made_output = False
    while True:
        res = await speech_queue.get()
        if res is None and made_output:
            break
        if res is None:
            continue
        sentence,audio = res
        logger.info(f'Finished: {sentence}')
        wave_obj = sa.WaveObject.from_wave_file(audio)
        play_obj = wave_obj.play()
        play_obj.wait_done()
        made_output = True


def play_audio(file_name):
    logger.debug('Trying to play audio')
    wave_obj = sa.WaveObject.from_wave_file(file_name)
    play_obj = wave_obj.play()
    play_obj.wait_done()


async def main():
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=16) as executor:
        word_queue = asyncio.Queue()
        speech_queue = asyncio.Queue()
        # Start streaming GPT responses

        # Start task to check speech queue
        speech_task = asyncio.create_task(check_speech_queue(speech_queue))

        logger.info('Starting speech task')
        gpt_task = loop.run_in_executor(executor, call_gpt_stream, 'gpt-3.5-turbo-0125', "How do I brush my teeth", word_queue)


        full_response = ''
        sentence = ''
        while True:
            word = await word_queue.get()
            if word is None:
                break
            logger.info(word)
            sentence += word
            full_response += word
            if word == '.':
                logger.info(f'Starting text2speech with {sentence}')
                loop.run_in_executor(executor, create_speech_async, sentence, speech_queue)
                sentence = ''

        logger.info(full_response)
        await gpt_task  # Wait for GPT stream task to complete
        await speech_queue.put(None)  # Signal end of speech queue
        await speech_task


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', )
    asyncio.run(main())
