import time

import openai


def stream_to_speakers() -> None:
    import pyaudio

    player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)


    with openai.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            response_format="pcm",  # similar to WAV, but without a header chunk at the start.
            input="""I see skies of blue and clouds of white 
                The bright blessed days, the dark sacred nights 
                And I think to myself 
                What a wonderful world""",
    ) as response:
        for chunk in response.iter_bytes(chunk_size=1024):
            player_stream.write(chunk)

#
# from openai import OpenAI
#
# client = OpenAI()
#
# with client.audio.speech.with_streaming_response.create(
#     model="tts-1",
#     voice="alloy",
#     input="""I see skies of blue and clouds of white
#              The bright blessed days, the dark sacred nights
#              And I think to myself
#              What a wonderful world""",
# ) as response:
#     # This doesn't seem to be *actually* streaming, it just creates the file
#     # and then doesn't update it until the whole generation is finished
#     response.stream_to_file("speech.mp3")