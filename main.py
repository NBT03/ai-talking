import requests
from gtts import gTTS
import os
from speech2text import recognize_from_microphone as i
from chat_memory import load_memory, save_memory

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma3"
# MODEL = "gemma3:270m"
# SESSION_ID = "Vietnamese"
SESSION_ID = "Japanese"
AUDIO_FILE = "tts.mp3"

system_prompt = {
    "role": "system",
    "content": (
            """あなたは日本語で会話する友だちAIです。
返事は必ず日本語だけでしてください。
ベトナム語や英語は使わないでください。

話し方は友だちのように、やさしくて自然にしてください。
先生のように説明しすぎないでください。

日本語N5レベルのかんたんな言葉だけを使ってください。
むずかしい漢字や文法は使わないでください。
文はみじかく、わかりやすくしてください。

会話がつづくように、ときどきやさしい質問をしてください。
でも、質問は1回に1つだけにしてください。

相手は日本語を勉強している人です。
まちがいがあっても、やさしく自然に直してください。"""
    # """
    #     Bạn là một người bạn đồng hành để có thể nói chuyện và
    #     chia sẻ với tôi về cuộc sống, kiến thức xã hội, và cả điều trị tâm lý.
    # """
    )
}

def speak(text, lang="vi"):
    tts = gTTS(text=text, lang=lang)
    tts.save(AUDIO_FILE)
    # tld="com.vn"
    # phát mp3 bằng mpg123
    # os.system(f"mpg123 -q {AUDIO_FILE}")
    os.system(
    f"bash -c \"ffplay -nodisp -autoexit -af 'atempo=1.2' {AUDIO_FILE}\""
)
    os.remove(AUDIO_FILE)

def chat_loop():
    # Load memory cũ
    messages = [system_prompt]
    messages += load_memory(SESSION_ID)

    while True:
        text = i()

        messages.append({"role": "user", "content": text})

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": messages,
                "stream": False
            }
        )

        assistant_reply = response.json()["message"]["content"]
        print(f"AI: {assistant_reply}\n")

        # Text → Speech
        speak(assistant_reply, lang="ja")

        messages.append({"role": "assistant", "content": assistant_reply})

        # Lưu memory sau mỗi lượt
        save_memory(SESSION_ID, messages)

if __name__ == "__main__":
    chat_loop()
