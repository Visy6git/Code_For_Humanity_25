from groq import Groq
from dotenv import load_dotenv
import pyperclip
from PIL import ImageGrab,Image
from streamlit_lottie import st_lottie 
import speech_recognition as sr
import google.generativeai as genai
import cv2
import os
import datetime
import wave
import uuid
import shelve
from st_audiorec import st_audiorec
import time
import streamlit as st
from llama_index.core import (
    StorageContext,
    ServiceContext,
    load_index_from_storage
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq as Groq1
import wave
import datetime
from st_audiorec import st_audiorec
import requests
import json
import base64
from st_audiorec import st_audiorec
from streamlit_navigation_bar import st_navbar

from pydub import AudioSegment

AudioSegment.converter = r"path\to\ffmpeg.exe"

load_dotenv()


grok_key=os.getenv('GROQ_API_KEY')
genai_key=os.getenv('GENAI_API_KEY')
bhashini_key=os.getenv('BHASHINI_API_KEY')
genai.configure(api_key=genai_key)
groq_client=Groq(api_key= grok_key)

web_cam=cv2.VideoCapture(0)

AUDIO_FOLDER = "audio_input"
if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)
  

pages = ["Text Assistant 💻", "Voice Assistant 🤖"]
styles = {
    "nav": {
        "background-color": "rgb(123, 209, 146)",
    },
    "div": {
        "max-width": "32rem",
    },
    "span": {
        "border-radius": "0.5rem",
        "color": "rgb(49, 51, 63)",
        "margin": "0 0.125rem",
        "padding": "0.4375rem 0.625rem",
    },
    "active": {
        "background-color": "rgba(255, 255, 255, 0.25)",
    },
    "hover": {
        "background-color": "rgba(255, 255, 255, 0.35)",
    },
}

page = st_navbar(pages, styles=styles)

text_languages = {
    "Hindi": "hi",
    "Assamese": "as",
    "Gujarati": "gu",
    "Marathi": "mr",
    "Odia": "or",
    "Punjabi": "pa",
    "Bengali": "bn",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Tamil": "ta",
    "Telugu": "te",
    "English": "en",
    "Bodo": "brx",
    "Manipuri": "mni"
}


if "groq_model" not in st.session_state:
    st.session_state["groq_model"] = "llama3-70b-8192"

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# Load chat history from shelve file
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])

# Save chat history to shelve file
def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages

if 'messages' not in st.session_state:
    st.session_state['messages'] = load_chat_history()  # Load history into session state if not already

st.title("Sarathi: Your multilingual AI Assistant")



with st.sidebar:
    selected_language = st.selectbox("Select a language", options=list(text_languages.keys()))
    st.write(f"Selected language: {selected_language}")
    target_lang = text_languages[selected_language]
    voice_gender = st.radio("Select Voice", ["male", "female"])
    st.write("Chatbot responds in same language")
    same_lang = st.radio("Converse in the same language", ["yes", "no"])
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        save_chat_history([])

sys_msg=(
    'You are a empathetic , kind ,helpful multi-modal AI voice assistant for government schemes.Your user may or may not have attached a photo for context'
    '(either a screenshot or webcam capture).Any photo has already been processed into highly detailed '
    'text prompt that will be attached to thier transcribed voice prompt.You will also be given relavent context '
    ' for the answer from the database .Generate the most useful and'
    'factual response possible,carefully considering all previous generated text in your response before'
    'adding new tokens to the response.Do not expect or request images,just use the context if added'
    'Use all the context of this conversation to generate a relavent response to the conversation.Make'
    'your responses clear and concise,avoiding any verbosisity keep it readable and understandable.'
)

convo=[{'role':'system','content':sys_msg}]

generation_config={
    'temperature':0.7,
    'top_p':1,
    'top_k':1,
    'max_output_tokens':2048,
}
safety_settings=[
    {
        'category':'HARM_CATEGORY_HARASSMENT',
        'threshold':'BLOCK_NONE'
    },
    {
        'category':'HARM_CATEGORY_HATE_SPEECH',
        'threshold':'BLOCK_NONE'
    },
    {
        'category':'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        'threshold':'BLOCK_NONE'
    },
    {
        'category':'HARM_CATEGORY_DANGEROUS_CONTENT',
        'threshold':'BLOCK_NONE'
    },

]

model=genai.GenerativeModel('gemini-1.5-flash-latest',
                            generation_config=generation_config,
                            safety_settings=safety_settings,)

r=sr.Recognizer()
source=sr.Microphone()

def groq_prompt(prompt,img_context,vb):
    conversation_history = "\n".join(
        [f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state['messages']]
    )
    if img_context:
        prompt=(f'USER PROMPT:{prompt}\n\n RELATED CONTEXT:{vb} \n\nIMAGE CONTEXT: {img_context}\n\nConversation History:\n{conversation_history}')
    else:
        prompt=(f'USER PROMPT:{prompt}\n\n RELATED CONTEXT:{vb} \n\nConversation History:\n{conversation_history}')
    convo.append({'role':'user','content':prompt})
    chat_completion=groq_client.chat.completions.create(messages=convo,model='llama3-70b-8192')
    response=chat_completion.choices[0].message
    convo.append(response)
    return response.content

def save_audio(file_name, audio_data):
    # Save the audio file
    with wave.open(file_name, 'wb') as wav_file:
        wav_file.setnchannels(2)  # mono audio
        wav_file.setsampwidth(2)  # sample width in bytes
        wav_file.setframerate(44100)  # frame rate
        wav_file.writeframes(audio_data)


def audiorec_demo_app():
    wav_audio_data = st_audiorec()  # Record audio
    if wav_audio_data is not None:
        # Display audio playback
        col_playback, col_space = st.columns([0.58, 0.42])
        # Save the audio file
        unique_filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".wav"
        audio_file_path = os.path.join(AUDIO_FOLDER, unique_filename)
        save_audio(audio_file_path, wav_audio_data)
        with open(audio_file_path, "rb") as audio_file:
            audio_data = audio_file.read()
        base64_string = base64.b64encode(audio_data).decode('utf-8')
        target_lang = "en"
        translated=bhashini_stt(target_lang,base64_string)
        print(translated)
        return translated


def function_call(prompt):
    sys_msg=(
        'You are a AI function calling model.You will determine whether extracting the users clipboard content,'
        'taking a screenshot,capturing the webcam or calling no functions is best for a voice assistant to respond '
        'to the user prompt.The webcam can be assumed to be a normal laptop webcam facing the user .You will '
        'respond with only one selection from this list;["extract clipboard","take screenshot","capture webcam","None"] \n'
        'Do no respond with anything but the most logical selection from that list with no explanations.Format the '
        'function call name excatly as it appears in the list above.'
    )

    function_convo=[{'role':'system','content':sys_msg},{'role':'user','content':prompt}]
    chat_completion=groq_client.chat.completions.create(messages=function_convo,model='llama3-70b-8192')
    response=chat_completion.choices[0].message
    return response.content

def take_screenshot():
    path = 'screenshot.jpg'
    screenshot=ImageGrab.grab()

    rgb_im=screenshot.convert('RGB')
    rgb_im.save(path,quality=15)
    with st.expander(f"Captured image"):
        st.image("screenshot.jpg", caption="I see your screen!",use_column_width="auto")

def web_cam_capture():
    if not web_cam.isOpened():
        print("Error: Could not open webcam")
        exit()
    path='webcam.jpg'
    ret,frame=web_cam.read()
    cv2.imwrite(path,frame)
    with st.expander(f"Captured image"):
        st.image("webcam.jpg", caption="I see you!")

def get_clipboard():
    clipboard_content=pyperclip.paste()
    if isinstance(clipboard_content,str):
        return clipboard_content
    else:
        print('No clipboard text to copy')
        return None
 
def scheme_context(question):
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = Groq1(model="llama3-70b-8192", api_key=grok_key)
    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)
    storage_context = StorageContext.from_defaults(persist_dir="./storage2_mini")
    index = load_index_from_storage(storage_context, service_context=service_context)
    query_engine = index.as_query_engine(service_context=service_context)
    resp = query_engine.query(question)
    return resp.response

def vision_prompt(prompt,photo_path):
    img=Image.open(photo_path)
    prompt=(
        'You are a vision analysis AI model that provides semantic meaning from images to provide context'
        'to send to another AI that will create a response to the user.Do not respond as the AI assistant'
        'to the user.Instead take the user prompt and try to extract all meaning from the photo'
        'relavent to the user prompt .Then generate as much objective data about the image for the next AI'
        f'assistant who will respond to the user prompt. \nUser Prompt: {prompt}'
    )
    response = model.generate_content([prompt,img])
    return response.text

def translation(prompt, source_lang,target_lang):
    service_id_trans = "ai4bharat/indictrans-v2-all-gpu--t4"
    
    if same_lang=="no":
        return prompt
    if source_lang == "en":
        return prompt
    
    # Headers as provided
    headers = {
        "Postman-Token": "<calculated when request is sent>", 
        "Content-Type": "application/json",
        "Content-Length": "<calculated when request is sent>", 
        "Host": "<calculated when request is sent>",
        "User-Agent": "PostmanRuntime/7.40.0",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "User-Agent": "Python", 
        "Authorization": bhashini_key
    }
    
    # Correct body structure with sourceLanguage and targetLanguage properly set
    body = {
        "pipelineTasks": [
            {
                "taskType": "translation",
                "config": {
                    "language": {
                        "sourceLanguage": source_lang,  # Source language (Malayalam in this case)
                        "targetLanguage": target_lang   # Target language (English in this case)
                    },
                    "serviceId": service_id_trans
                }
            }
        ],
        "inputData": {
            "input": [
                {
                    "source": prompt
                }
            ]
        }
    }
    
    # Make the POST request
    response = requests.post("https://dhruva-api.bhashini.gov.in/services/inference/pipeline", headers=headers, json=body)
    
    # Get response data
    response_data = response.json()
    
    # Extracting only the English translation (target)
    translation_text = response_data['pipelineResponse'][0]['output'][0]['target']
    with st.expander(f"Language converting to english from {source_lang}"):
        st.write(translation_text)
    return translation_text


def bhashini_tts(target_lang,query,gender):
    if target_lang in ["hi", "as", "gu", "mr", "or", "pa", "bn"]:
        service_id_tts = "ai4bharat/indic-tts-coqui-indo_aryan-gpu--t4"
    elif target_lang in ["kn", "ml", "ta", "te"]:
        service_id_tts = "ai4bharat/indic-tts-coqui-dravidian-gpu--t4"
    elif target_lang in ["en", "brx", "mni"]:
        service_id_tts = "ai4bharat/indic-tts-coqui-misc-gpu--t4"

    headers = {
     "Postman-Token": "<calculated when request is sent>", 
     "Content-Type": "application/json",
     "Content-Length": "<calculated when request is sent>", 
     "Host": "<calculated when request is sent>",
     "User-Agent": "PostmanRuntime/7.40.0",
     "Accept": "*/*",
     "Accept-Encoding": "gzip, deflate, br",
     "Connection": "keep-alive",
     "Accept":"*/*",
     "User-Agent": "Python", 
     "Authorization":bhashini_key
     }
    body = {
    "pipelineTasks": [
        {
            "taskType": "translation",
            "config": {
                "language": {
                    "sourceLanguage": "en",
                    "targetLanguage": target_lang
                },
                "serviceId": "ai4bharat/indictrans-v2-all-gpu--t4"
            }
        },
        {
            "taskType": "tts",
            "config": {
                "language": {
                    "sourceLanguage": target_lang
                },
                "serviceId": service_id_tts,
                "gender": gender,
                "samplingRate": 8000
            }
        }
    ],
    "inputData": {
        "input": [
            {
                "source": query
            }
        ]
    }
}
    response1 = requests.post("https://dhruva-api.bhashini.gov.in/services/inference/pipeline", headers=headers,json=body)
    response_data = response1.json()
    target_text = response_data["pipelineResponse"][0]["output"][0]["target"]
    with st.expander(f"Text in: {selected_language}"):
        st.write(target_text)
    D = response_data["pipelineResponse"][1]["audio"][0]["audioContent"]
    audio_data = base64.b64decode(D)
    audio_folder = "audio_files"
    os.makedirs(audio_folder, exist_ok=True)
    filename = f"{audio_folder}/output_{uuid.uuid4()}.wav"
    with open(filename, "wb") as audio_file:
        audio_file.write(audio_data)
    st.markdown(
    """
    <style>
    audio {
        width: 150px; /* Adjust this value to make it smaller or larger */
        height: 30px; /* Adjust the height as needed */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the audio file
    st.audio(filename, format="audio/wav")

def bhashini_stt(target_language,audio):
    if target_language in ["hi"]:
        service_id_sst= "ai4bharat/conformer-hi-gpu--t4"
    elif target_language in ["kn", "ml", "ta", "te"]:
        service_id_sst = "ai4bharat/conformer-multilingual-dravidian-gpu--t4"
    elif target_language in ["en"]:
        service_id_sst = "ai4bharat/whisper-medium-en--gpu--t4"
    elif target_language in ["bn", "gu", "mr", "or", "pa", "sa","ur"]:
        service_id_sst = "ai4bharat/conformer-multilingual-indo_aryan-gpu--t4"
    headers = {
     "Postman-Token": "<calculated when request is sent>", 
     "Content-Type": "application/json",
     "Content-Length": "<calculated when request is sent>", 
     "Host": "<calculated when request is sent>",
     "User-Agent": "PostmanRuntime/7.40.0",
     "Accept": "*/*",
     "Accept-Encoding": "gzip, deflate, br",
     "Connection": "keep-alive",
     "Accept":"*/*",
     "User-Agent": "Python", 
     "Authorization":bhashini_key
     }
    body = {
    "pipelineTasks": [
        {
            "taskType": "asr",
            "config": {
                "language": {
                    "sourceLanguage": target_language
                },
                "serviceId": service_id_sst,
                "audioFormat": "flac",
                "samplingRate": 16000
            }
        },
        { 
            "taskType": "translation",
            "config": {
                "language": {
                    "sourceLanguage": target_language,
                    "targetLanguage": "en"
                },
                "serviceId": "ai4bharat/indictrans-v2-all-gpu--t4"
            }
        }
    ],
    "inputData": {
        "audio": [
            {
                "audioContent": audio
            }
        ]
    }
}
    response = requests.post("https://dhruva-api.bhashini.gov.in/services/inference/pipeline", headers=headers,json=body)
    response_data = response.json()
    source_text = response_data["pipelineResponse"][0]["output"][0]["source"]
    st.expander(source_text)
    target_text = response_data["pipelineResponse"][1]["output"][0]["target"]
    return target_text

def wav_to_text(audio_file_path):
    with open(audio_file_path, "rb") as audio_file:
        audio_data = audio_file.read()
    base64_string = base64.b64encode(audio_data).decode('utf-8')
    target_lang = "en"
    translated=bhashini_stt(target_lang,base64_string)
    return translated

def callback():
    clean_prompt=audiorec_demo_app()
    st.session_state['messages'].append({"role": "user", "content": clean_prompt})
    if clean_prompt:
        call=function_call(clean_prompt)
        rag_answer=scheme_context(clean_prompt)
        if 'take screenshot' in call:
            print('Taking screenshot...')
            take_screenshot()
            visual_context=vision_prompt(prompt=clean_prompt,photo_path='screenshot.jpg')

        elif 'capture webcam' in call:
            print('Capturing webcam...')
            web_cam_capture()
            visual_context=vision_prompt(prompt=clean_prompt,photo_path='webcam.jpg')
    
        elif 'extract clipboard' in call:
            print('Extracting clipboard...')
            paste=get_clipboard()
            clean_prompt = f'{clean_prompt}\n\n CLIPBOARD CONTENT: {paste}'
            visual_context = None
        
        elif ' context' in call:
            print('Need more context...')
            pdf_context=pdf_context(clean_prompt)
            clean_prompt = f'{clean_prompt}\n\n PDF CONTEXT: {pdf_context}'
            visual_context = None
        else:
            visual_context=None
        response = groq_prompt(prompt=clean_prompt,img_context=visual_context,vb=rag_answer)
        bhashini_tts(target_lang,response,voice_gender)
        st.session_state['messages'].append({"role": "assistant", "content": response})
        save_chat_history(st.session_state['messages'])



if page == "Text Assistant 💻":
# Display chat messages
    for message in st.session_state['messages']:
        avatar = USER_AVATAR if message['role'] == 'user' else BOT_AVATAR
        with st.chat_message(message['role'], avatar=avatar):
            st.markdown(message['content'])


    # Main chat interface
    if prompt := st.chat_input("How can I help?"):
        st.session_state['messages'].append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)
    
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            message_placeholder = st.empty()
            clean_prompt=translation(prompt, target_lang,"en")
            if clean_prompt:
                call=function_call(clean_prompt)
                rag_answer=scheme_context(clean_prompt)
                if 'take screenshot' in call:
                    print('Taking screenshot...')
                    take_screenshot()
                    visual_context=vision_prompt(prompt=clean_prompt,photo_path='screenshot.jpg')

                elif 'capture webcam' in call:
                    print('Capturing webcam...')
                    web_cam_capture()
                    visual_context=vision_prompt(prompt=clean_prompt,photo_path='webcam.jpg')
        
                elif 'extract clipboard' in call:
                    print('Extracting clipboard...')
                    paste=get_clipboard()
                    clean_prompt = f'{clean_prompt}\n\n CLIPBOARD CONTENT: {paste}'
                    visual_context = None
            
                elif ' context' in call:
                    print('Need more context...')
                    pdf_context=scheme_context(clean_prompt)
                    clean_prompt = f'{clean_prompt}\n\n PDF CONTEXT: {pdf_context}'
                    visual_context = None
                else:
                    visual_context=None
                main_response = groq_prompt(prompt=clean_prompt,img_context=visual_context,vb=rag_answer)
                lang_for_tra="en"
                translated_text=translation(main_response,lang_for_tra,target_lang)
                st.markdown(translated_text)
                bhashini_tts(target_lang,main_response,voice_gender)
                st.session_state['messages'].append({"role": "assistant", "content": main_response})
                # Save chat history after interaction
                save_chat_history(st.session_state['messages'])

else:
    st.sidebar.write("Voice Assistant 🤖")
    callback()
    
