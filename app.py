import streamlit as st

from src.ollama_chain import OllamaChain, OllamaRAGChain
from src.llama_cpp_chains import LlamaChain
from src.pdf_handler import extract_pdf
from src.vqa import answer_visual_question
from src.audio_processor import AudioProcessor
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from dotenv import load_dotenv
import os

load_dotenv()

audio_processor = AudioProcessor()

@st.cache_resource
def load_chain(_chat_memory):
    if hasattr(st.session_state, 'pdf_chat') and st.session_state.pdf_chat:
        return OllamaRAGChain(_chat_memory)
    else:
        return OllamaChain(_chat_memory)


def file_uploader_change():
    if st.session_state.uploaded_file:
        if not st.session_state.pdf_chat:
            clear_cache()
            st.session_state.pdf_chat = True

        st.session_state.knowledge_change = True

    else:
        clear_cache()
        st.session_state.pdf_chat = False


def toggle_pdf_chat_change():
    clear_cache()
    if st.session_state.pdf_chat and st.session_state.uploaded_file:
        st.session_state.knowledge_change = True


def clear_input_field():
    # store the question
    st.session_state.user_question = st.session_state.user_input
    # clear the variable
    st.session_state.user_input = ""


def set_send_input():
    st.session_state.send_input = True
    clear_input_field()


def clear_cache():
    st.cache_resource.clear()


def initial_session_state():
    st.session_state.send_input = False
    st.session_state.knowledge_change = False
    st.session_state.pdf_chat = False


def main():
    # Initialize session state first
    if 'send_input' not in st.session_state:
        initial_session_state()
    
    # Initialize
    # Title
    st.title('Multi-Media Chat Bot')
    chat_container = st.container()

    # sidebar
    # st.sidebar.title('Chat Session')

    # file upload
    st.sidebar.toggle('PDF Chat', value=False, key='pdf_chat', on_change=toggle_pdf_chat_change)
    uploaded_pdf = st.sidebar.file_uploader('Upload your pdf files',
                                            type='pdf',
                                            accept_multiple_files=True,
                                            key='uploaded_file',
                                            on_change=file_uploader_change)

    # Image upload
    uploaded_image = st.sidebar.file_uploader('Upload Images', type=['jpg', 'jpeg', 'png'], key='uploaded_image')

    # Audio upload
    uploaded_audio = st.sidebar.file_uploader('Upload Audio', type=['wav', 'mp3'], key='uploaded_audio')

    # Input objects
    user_input = st.text_input('Chat here', key='user_input', on_change=set_send_input)
    send_button = st.button('Send', key='send_button')

    # ----------------------------------------------------------------------------------------------------

    chat_history = StreamlitChatMessageHistory(key='history')

    with chat_container:
        for msg in chat_history.messages:
            st.chat_message(msg.type).write(msg.content)

    llm_chain = load_chain(chat_history)
    if st.session_state.knowledge_change:
        with st.spinner('Updating knowledge base'):
            llm_chain.update_chain(uploaded_pdf)
            st.session_state.knowledge_change = False

    # we use "or" operation here because user can press 'Enter' instead of 'Send' button
    if (send_button or st.session_state.send_input) and st.session_state.user_question != "":
        with chat_container:
            st.chat_message('user').write(st.session_state.user_question)
            
            if uploaded_image:
                # Ensure cache directory exists
                cache_dir = './.cache/temp_files'
                os.makedirs(cache_dir, exist_ok=True)
                image_path = os.path.join(cache_dir, uploaded_image.name)
                with open(image_path, 'wb') as f:
                    f.write(uploaded_image.getvalue())
                llm_response = answer_visual_question(image_path, st.session_state.user_question)
            elif uploaded_audio:
                # Ensure cache directory exists
                cache_dir = './.cache/temp_files'
                os.makedirs(cache_dir, exist_ok=True)
                
                # Sanitize filename and ensure proper extension
                import re
                safe_filename = re.sub(r'[^\w\-_\.]', '_', uploaded_audio.name)
                if not safe_filename.lower().endswith(('.mp3', '.wav')):
                    safe_filename += '.mp3'  # Default to mp3 if no extension
                
                audio_path = os.path.join(cache_dir, safe_filename)
                
                # Save the audio file
                with open(audio_path, 'wb') as f:
                    f.write(uploaded_audio.getvalue())
                
                # Verify file was saved
                if os.path.exists(audio_path):
                    st.write(f"Processing audio file: {audio_path}")  # Debug statement
                    
                    # Store the original question
                    original_question = st.session_state.user_question
                    
                    # Transcribe the audio
                    transcribed_text = audio_processor.audio_to_text(audio_path)
                    st.write(f"Converted audio to text: {transcribed_text}")  # Debug statement
                    
                    # Create a combined prompt that includes both the user's question and the audio content
                    if "what" in original_question.lower() and "audio" in original_question.lower():
                        # User is asking about the audio content
                        combined_prompt = f"Audio transcription: {transcribed_text}\n\nUser question: {original_question}"
                    else:
                        # User is asking a question that should be answered using the audio content as context
                        combined_prompt = f"Based on this audio transcription: {transcribed_text}\n\nAnswer this question: {original_question}"
                    
                    llm_response = llm_chain.run(user_input=combined_prompt)
                else:
                    st.error(f"Failed to save audio file to {audio_path}")
                    llm_response = "Error: Could not save audio file for processing."
            else:
                # Use the appropriate chain (RAG chain if PDF chat is enabled, regular chain otherwise)
                llm_response = llm_chain.run(user_input=st.session_state.user_question)
            
            st.session_state.user_question = ""
            st.chat_message('ai').write(llm_response)

            # Convert response to speech and play it
            audio_file = audio_processor.text_to_speech(llm_response)
            audio_bytes = open(audio_file, 'rb').read()
            st.audio(audio_bytes, format='audio/mp3')


if __name__ == '__main__':
    main()