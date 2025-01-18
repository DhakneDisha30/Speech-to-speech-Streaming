import streamlit as st
import os
import ffmpeg
import speech_recognition as sr
from gtts import gTTS
from langdetect import detect
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def extract_audio(input_video_path, output_audio_path):
    """Extract audio from video and save it as a WAV file."""
    ffmpeg.input(input_video_path).output(
        output_audio_path, acodec="pcm_s16le", ac=1, ar="16000"
    ).run()

def transcribe_audio(audio_path):
    """Transcribe audio using SpeechRecognition."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio)

def translate_text(input_text, target_language):
    """Translate text using Google Generative AI."""
    summary_prompt = """
    You are a translator. Translate the following input text:
    {input}
    into the desired language: {convertLanguage}.
    """
    prompt_template = PromptTemplate(
        input_variables=["input", "convertLanguage"], template=summary_prompt
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=GOOGLE_API_KEY
    )
    chain = prompt_template | llm | StrOutputParser()
    return chain.invoke({"input": input_text, "convertLanguage": target_language})

def text_to_speech(text, output_audio_path):
    """Convert text to speech and save it as an MP3 file."""
    detected_language = detect(text)
    tts = gTTS(text=text, lang=detected_language)
    tts.save(output_audio_path)

def pad_or_trim_audio(audio_segment, target_duration_ms):
    """Pad or trim an audio segment to match the target duration."""
    duration_difference = target_duration_ms - len(audio_segment)
    if duration_difference > 0:
        # Add silence to the end
        return audio_segment + AudioSegment.silent(duration=duration_difference)
    else:
        # Trim audio
        return audio_segment[:target_duration_ms]

def synchronize_audio_with_video(video_path, audio_path, output_path):
    """Synchronize translated audio with the video."""
    # Load video and translated audio
    video_clip = VideoFileClip(video_path)
    translated_audio = AudioSegment.from_file(audio_path)
    
    # Match the duration of the translated audio to the video
    video_duration_ms = int(video_clip.duration * 1000)  # Video duration in milliseconds
    translated_audio = pad_or_trim_audio(translated_audio, video_duration_ms)

    # Export adjusted audio
    adjusted_audio_path = "adjusted_audio.mp3"
    translated_audio.export(adjusted_audio_path, format="mp3")
    
    # Replace the video audio with the adjusted translated audio
    adjusted_audio_clip = AudioFileClip(adjusted_audio_path).set_duration(video_clip.duration)
    final_video = video_clip.set_audio(adjusted_audio_clip)

    # Write the output video file
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
    
    # Clean up temporary files
    os.remove(adjusted_audio_path)

# Updated Streamlit App
st.title("Speech-to-Speech Translation")

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mkv", "avi"])
target_language = st.text_input("Enter the target language (e.g., 'fr' for French, 'hi' for Hindi)")

if st.button("Translate"):
    if uploaded_video and target_language:
        input_video_path = "uploaded_video.mp4"
        output_audio_path = "extracted_audio.wav"
        translated_audio_path = "translated_audio.mp3"
        output_video_path = "final_video.mp4"

        # Save uploaded video
        with open(input_video_path, "wb") as f:
            f.write(uploaded_video.read())

        # Step 1: Extract Audio
        st.write("Extracting audio from video...")
        extract_audio(input_video_path, output_audio_path)

        # Step 2: Transcribe Audio
        st.write("Transcribing audio...")
        transcribed_text = transcribe_audio(output_audio_path)

        # Step 3: Translate Text
        st.write("Translating text...")
        translated_text = translate_text(transcribed_text, target_language)

        # Step 4: Text-to-Speech
        st.write("Converting text to speech...")
        text_to_speech(translated_text, translated_audio_path)

        # Step 5: Synchronize Audio and Video
        st.write("Synchronizing audio with video...")
        synchronize_audio_with_video(input_video_path, translated_audio_path, output_video_path)

        st.success("Translation and synchronization complete!")
        st.video(output_video_path)
