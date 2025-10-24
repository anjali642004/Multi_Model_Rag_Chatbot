import speech_recognition as sr
from gtts import gTTS
import tempfile
import os
import subprocess

class AudioProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def audio_to_text(self, audio_file):
        """Process an uploaded audio file and convert it to text."""
        temp_wav = None
        try:
            # Normalize the file path and check if file exists
            audio_file = os.path.normpath(audio_file)
            if not os.path.exists(audio_file):
                return f"Error: Audio file not found at {audio_file}"
            
            print(f"Processing audio file: {audio_file}")  # Debug statement
            
            # Convert audio to WAV format using FFmpeg for better speech recognition
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_wav.close()  # Close the file handle so FFmpeg can write to it
            
            # Use FFmpeg to convert audio to WAV format (16kHz, mono)
            ffmpeg_cmd = [
                'ffmpeg', '-i', audio_file, 
                '-ar', '16000',  # Sample rate
                '-ac', '1',      # Mono channel
                '-y',            # Overwrite output file
                temp_wav.name
            ]
            
            print(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")  # Debug statement
            
            # Run FFmpeg conversion
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}")  # Debug statement
                return f"Error converting audio with FFmpeg: {result.stderr}"
            
            # Check if the converted WAV file exists and has content
            if not os.path.exists(temp_wav.name) or os.path.getsize(temp_wav.name) == 0:
                return "Error: FFmpeg conversion failed - no output file created"
            
            print(f"FFmpeg conversion successful, WAV file size: {os.path.getsize(temp_wav.name)} bytes")  # Debug statement
            
            # Use speech recognition on the converted WAV file
            with sr.AudioFile(temp_wav.name) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data)
                print(f"Speech recognition successful: {text}")  # Debug statement
                return text
                
        except sr.UnknownValueError:
            return "Could not understand the audio. Please try speaking more clearly."
        except sr.RequestError as e:
            return f"Error with speech recognition service: {e}"
        except Exception as e:
            print(f"Error processing audio file: {e}")  # Debug statement
            return f"Error processing audio file: {e}"
        finally:
            # Clean up temporary WAV file
            if temp_wav and os.path.exists(temp_wav.name):
                try:
                    os.unlink(temp_wav.name)
                    print(f"Cleaned up temporary file: {temp_wav.name}")  # Debug statement
                except Exception as e:
                    print(f"Error cleaning up temporary file: {e}")  # Debug statement

    def text_to_speech(self, text):
        """Convert text to speech using gTTS and save as a .mp3 file."""
        tts = gTTS(text)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts.save(temp_audio.name)
            return temp_audio.name