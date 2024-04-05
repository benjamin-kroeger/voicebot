import asyncio
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import openai
from openai import OpenAI

from audio_stream_warpper import AudioStreamWrapper

logger = logging.getLogger(__name__)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def call_gpt_stream(model: Literal['gpt-4', 'gpt-3.5-turbo-0125'], message: str = None, queue: asyncio.Queue = None,
                    loop: asyncio.AbstractEventLoop = None):
    """
    Function to use openais completion endpoint

    :param model: The LLM that shall be used for the completion
    :param message: The first user message
    :param queue: A asyncio.Queue to store the streamed words comming from openai
    :param loop: The loop in which to run the api call
    :return: None, puts the resluts into the queue as they come available
    """
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": message}, ],
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            # put words into queue
            loop.call_soon_threadsafe(queue.put_nowait, chunk.choices[0].delta.content)
            logger.info(f'GPT: {chunk.choices[0].delta.content}')
    loop.call_soon_threadsafe(queue.put_nowait, None)  # Signal end of stream


def stream_response_to_speaker(sentence_input: str, player_stream) -> None:
    """
    Call openai TTS and stream the response into the output player stream

    :param sentence_input: The sentence that shall be turned into speech
    :param player_stream: The player stream object that outputs the streamed repose
    :return:
    """
    logger.info(f"TTS: {sentence_input}")
    # make a request to openai
    with openai.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            response_format="pcm",  # similar to WAV, but without a header chunk at the start.
            input=sentence_input,
    ) as response:
        for chunk in response.iter_bytes(chunk_size=1024):
            player_stream.write(chunk)


def convert_speech_to_text(path_to_mp3: str) -> str:
    """
    Convert the provided mp3 to text

    :param path_to_mp3: The mp3
    :return: The
    """
    audio_file = open(path_to_mp3, "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )
    return transcription.text


async def main():
    logger.info("Starting Speech to text")
    query_text = convert_speech_to_text(r'/home/benjaminkroeger/PycharmProjects/my_openai/teeth.mp3')

    # create audio wrapper
    with AudioStreamWrapper() as player_stream:
        # create async loop in which to run subprocesses
        loop = asyncio.get_running_loop()
        # Create executor for concurrent execution
        with ThreadPoolExecutor(max_workers=16) as executor:
            word_queue = asyncio.Queue()
            logger.info('Starting speech task')
            # start the gpt completion task and consume the results in the queue
            gpt_task = loop.run_in_executor(executor, call_gpt_stream, 'gpt-3.5-turbo-0125', query_text, word_queue, loop)

            full_response = ''
            sentence = ''
            multiplier = 1
            # Keep checking if there is a new word in the queue
            while True:
                # Wait for a new response to be available
                word = await word_queue.get()

                if word is None:
                    stream_response_to_speaker(sentence, player_stream)
                    break
                word = word.rstrip('\n')
                sentence += word

                # if there is a sentence available turn it into speech
                # At the beginning this should happen as soon as possible, after the first sentence
                if word == '.' and len(sentence) > 20 * multiplier and re.match(r'\w', sentence[-2]):
                    stream_response_to_speaker(sentence, player_stream)
                    multiplier += 2
                    sentence = ''

        logger.info(full_response)
        await gpt_task


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', )
    asyncio.run(main())
