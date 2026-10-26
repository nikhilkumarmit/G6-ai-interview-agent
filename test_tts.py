
import wave
import main

question = "Tell me about your technical skills and projects."

path = main.generate_tts_audio(question)

print("Audio saved to:", path)
print("File size:", path.stat().st_size, "bytes")

with wave.open(str(path), "rb") as audio:
    print("Duration:", round(
        audio.getnframes() / audio.getframerate(), 2
    ), "seconds")
