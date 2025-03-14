import streamlit as st
import base64
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

if "audio" not in st.session_state:
    st.session_state['audio'] = None

def text_to_speech(text: str):
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
    )
    st.success("Speech generated successfully!")
    return response.content


st.title("Text-to-Speech Converter")
text_input = st.text_area("Enter your text here:")
if st.button("Convert to Speech"):
    st.session_state['audio'] = None
    st.markdown("Generating audio...")
    if text_input:
        audio_content = text_to_speech(text_input)
        st.session_state['audio'] = audio_content
else:
    st.warning("Please enter text to convert.")


if st.session_state['audio'] is not None:
    audio_content = st.session_state['audio']
    audio_base64 = base64.b64encode(audio_content).decode('utf-8')
    audio_html = f'''
        <audio controls>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
    '''
    st.markdown(audio_html, unsafe_allow_html=True)