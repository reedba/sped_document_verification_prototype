import gradio as gr
import tempfile
import os
from dotenv import load_dotenv
import openai
from elevenlabs import set_api_key, save, voices, generate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.chat_message_histories import FileChatMessageHistory

# === Load environment variables ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
huggingface_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
set_api_key(eleven_labs_api_key)

# Ensure 'history' folder exists
os.makedirs("history", exist_ok=True)

# === System prompt ===
system_prompt = """
You are a helpful programming coach who specializes in LeetCode problems.
The user will speak in pseudo code or describe their approach.
Do not solve the problem. Instead, ask clarifying questions and guide them
step-by-step, as if you are their rubber duck.
Focus on structure, edge cases, and algorithm design — but never give the full solution.
"""

# === LangChain modern setup ===
chat = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
    huggingfacehub_api_token=huggingface_api_key,
    temperature=0.5,
    max_new_tokens=512
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
    ("ai", "")
])

# === Runnable with history ===
chain = prompt | chat
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: FileChatMessageHistory(f"history/{session_id}.json"),
    input_messages_key="input",
    history_messages_key="history"
)

# === Whisper STT ===
def whisper_transcribe(audio_path):
    with open(audio_path, "rb") as f:
        transcript = openai.Audio.transcribe("whisper-1", f)
        return transcript["text"]

# === ElevenLabs TTS ===
def speak_with_voice(text, voice_name="Rachel"):
    try:
        available_voices = voices()
        voice_to_use = next((v for v in available_voices if v.name.lower() == voice_name.lower()), available_voices[0])
        audio = generate(text=text, voice=voice_to_use.name)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        save(audio, temp_file.name)
        return temp_file.name
    except Exception as e:
        print(f"Voice error: {e}")
        return None

# === Main handler ===
def leetcode_coach(audio_file, voice_choice, session_id):
    try:
        user_text = whisper_transcribe(audio_file)
    except Exception as e:
        return f"Speech recognition error: {str(e)}", None

    response = chain_with_history.invoke({"input": user_text}, config={"configurable": {"session_id": session_id}})
    ai_response = response.content if hasattr(response, "content") else str(response)

    if voice_choice != "Text Only":
        voice_file = speak_with_voice(ai_response, voice_choice)
        return ai_response, voice_file

    return ai_response, None

# === Voice dropdown ===
try:
    available_voices = [voice.name for voice in voices()]
    voice_choices = ["Text Only"] + available_voices
except Exception:
    voice_choices = ["Text Only", "Rachel", "Adam", "Antoni"]

# === Gradio UI ===
iface = gr.Interface(
    fn=leetcode_coach,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="Speak your LeetCode logic"),
        gr.Dropdown(choices=voice_choices, value="Text Only", label="Voice Output"),
        gr.Textbox(label="Session ID", placeholder="Enter your session ID (e.g. user123)")
    ],
    outputs=[
        gr.Textbox(label="Coach Response"),
        gr.Audio(label="AI Voice")
    ],
    title="🧠 LeetCode Rubber Duck Coach",
    description="Talk through your logic. This AI will guide your thinking and can respond with voice."
)

iface.launch()