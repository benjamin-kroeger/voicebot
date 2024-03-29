import logging
import os
import asyncio
from typing import Literal, Iterator
from openai import OpenAI
from io import BytesIO
import simpleaudio as sa
from concurrent.futures import ThreadPoolExecutor,as_completed

logger = logging.getLogger(__name__)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def call_gpt_stream(model: Literal['gpt-4', 'gpt-3.5-turbo-0125'], message: str = None) -> Iterator[str]:
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": message}, ],
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


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


def main():
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        full_response = ''
        sentence = ''
        for word in call_gpt_stream(model='gpt-3.5-turbo-0125', message="How do I brush my teeth"):
            logger.info(word)
            sentence += word
            full_response += word
            if word == '.':
                futures.append(executor.submit(create_speech_async, sentence))
                sentence = ''

        logger.info(full_response)
        for audio in futures:
            play_audio(audio.result())


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',)
    main()
