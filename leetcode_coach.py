import gradio as gr
import tempfile
import os
from dotenv import load_dotenv
import openai
from elevenlabs import set_api_key, save, voices, generate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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

# === LangChain with Mistral model ===
chat = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
    huggingfacehub_api_token=huggingface_api_key,
    temperature=0.5,
    max_new_tokens=512
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

    # Load and trim history
    history_path = f"history/{session_id}.json"
    chat_history = FileChatMessageHistory(history_path)
    recent_messages = chat_history.messages[-10:] if len(chat_history.messages) > 10 else chat_history.messages

    # Compose prompt manually
    messages = [SystemMessage(content=system_prompt)] + recent_messages + [HumanMessage(content=user_text)]

    try:
        ai_response_raw = chat.invoke(messages)
        ai_response = ai_response_raw if isinstance(ai_response_raw, str) else ai_response_raw.content.strip()
    except Exception as e:
        return f"Model error: {str(e)}", None

    # Post-process to remove common prefixes like "Mini:"
    if ai_response.lower().startswith("mini"):
        ai_response = ai_response[4:].lstrip(":").strip()

    # Remove UTF-16 surrogate characters (invalid in UTF-8) and force UTF-8 safe output
    try:
        cleaned = ''.join(c for c in ai_response if not (0xD800 <= ord(c) <= 0xDFFF))
        ai_response = cleaned.encode("utf-8", "ignore").decode("utf-8", "ignore")
    except Exception as e:
        print(f"Normalization error: {e}")
        ai_response = "[Error decoding AI response.]"

    # Save this exchange to history
    chat_history.add_user_message(user_text)
    chat_history.add_ai_message(ai_response)

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

iface.launch(share=True)
