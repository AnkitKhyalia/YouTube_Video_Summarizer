from pytube import YouTube
import subprocess
import torch
from googletrans import Translator
import librosa
from huggingsound import SpeechRecognitionModel
from pydub import AudioSegment
from transformers import pipeline
import soundfile as sf
from tkinter import *

choice = 1
VIDEO_URL = "https://www.youtube.com/watch?v=hWLf6JFbZoo"
OUT = ''




def fun():
    global OUT
    yt = YouTube(VIDEO_URL)
    yt.streams.filter(only_audio=True, file_extension='mp4').first().download(filename='ytaudio.mp4')

    # ! ffmpeg -i ytaudio.mp4 -acodec pcm_s16le -ar 16000 ytaudio.wav
    # Set the input and output file names
    input_file = './ytaudio.mp4'
    output_file = './ytaudio.wav'

    # Set the audio codec and sampling rate
    audio_codec = 'pcm_s16le'
    sampling_rate = 16000

    # Construct the ffmpeg command
    command = f'ffmpeg -i {input_file} -acodec {audio_codec} -ar {sampling_rate} {output_file}'

    # Run the ffmpeg command as a subprocess
    subprocess.run(command.split())

    if choice == 0:
        audio = AudioSegment.from_file("./ytaudio.wav")
        audio = audio.set_frame_rate(16000)
        audio.export("./ytaudio_16k.wav", format="wav")
        audio = './ytaudio_16k.wav'
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        transcribe = pipeline(task="automatic-speech-recognition", model="vasista22/whisper-hindi-small", chunk_length_s=30,
                              device=device)
        transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language="hi",
                                                                                                 task="transcribe")
        full_transcript = transcribe(audio)["text"]
        # full_transcript

        translator = Translator()
        text = full_transcript

        result = translator.translate(text, src='hi', dest='en')

        # print(result)
        summarization = pipeline('summarization')
        summarized_text = summarization(result.text)
        summarized_text_str = str(summarized_text[0]['summary_text'])
        OUT = summarized_text[0]['summary_text']

        # Translate summarized text from English to Hindi
        final = translator.translate(summarized_text_str, src='en', dest='hi')
        print(final.text)

    if choice == 1:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SpeechRecognitionModel(
            "jonatasgrosman/wav2vec2-large-xlsr-53-english", device=device)
        input_file = 'ytaudio.wav'
        print(librosa.get_samplerate(input_file))

        # Stream over 30 seconds chunks rather than load the full file
        stream = librosa.stream(
            input_file,
            block_length=30,
            frame_length=16000,
            hop_length=16000
        )

        for i, speech in enumerate(stream):
            sf.write(f'{i}.wav', speech, 16000)

        audio_path = []
        for a in range(i + 1):
            audio_path.append(f'./{a}.wav')
            audio_path
        transcriptions = model.transcribe(audio_path)
        full_transcript = ' '

        for item in transcriptions:
            full_transcript += ''.join(item['transcription'])
            # print(full_transcript)

        # full_transcript
        summarization = pipeline('summarization')
        summarized_text = summarization(full_transcript)
        OUT = summarized_text[0]['summary_text']
        print(summarized_text[0]['summary_text'])

    root1 = Tk(baseName="Output")
    root1.title("YouTube Video Summarizer Output")
    root1.configure(background='#009688')
    root1.geometry("1200x900+500+300")
    root1.resizable(0, 0)
    l1 = OUT[:min(len(OUT), 100)]
    l2 = OUT[100:min(len(OUT), 200)]
    l3 = OUT[200:min(len(OUT), 300)]
    l4 = OUT[300:]
    Label(root1, text=l1, font="bold 10", bg="#009688", padx=20, pady=10).grid(row=0, column=0)
    Label(root1, text=l2, font="bold 10", bg="#009688", padx=20, pady=10).grid(row=1, column=0)
    Label(root1, text=l3, font="bold 10", bg="#009688", padx=20, pady=10).grid(row=2, column=0)
    Label(root1, text=l4, font="bold 10", bg="#009688", padx=20, pady=10).grid(row=3, column=0)
    root1.mainloop()



# GUI BLOCK
root = Tk(baseName="Video Summarizer")
root.title("YouTube Video Summarizer")
root.configure(background='#009688')
root.geometry("600x400+400+200")
root.resizable(0, 0)

# Main Title Label
title = Label(root, text="Video Summarizer", font="bold 26", bg="#009688", padx=140, pady=10).grid(row=0, column=0)
# URL Label
url_label = Label(root, text="URL:", font="bold", bg='#009688', justify="right", bd=1)
url_label.place(height=50, x=100, y=70)

# Model Label
model_label = Label(root, text="Model:", font="bold", bg='#009688', justify="right", bd=1)
model_label.place(height=50, x=90, y=135)

# Entry --> String
get_url = Entry(root, width=40)
get_url.place(width=300, height=30, x=150, y=80)

# DropDown
options = ["English", "Hindi"]

# Declaring Variable and choosing default one
default_option = StringVar(root)
default_option.set(options[0])
drop = OptionMenu(root, default_option, *options)
drop.place(width=200, x=150, y=145)

# Button Clear --> Reset all settings to default
def on_clear():
    default_option.set(options[0])
    get_url.delete(0, END)

clear = Button(root, text="Clear", command=on_clear)
clear.place(width=50, x=240, y=350)

# Function on Submit
def on_submit():
    global VIDEO_URL, choice
    VIDEO_URL = get_url.get()
    choice = default_option.get()
    if choice == 'English':
        choice = 1
    else:
        choice = 0

    print(VIDEO_URL, choice)
    fun()



# Button -->Submit
submit = Button(root, text="Submit", command=on_submit)
submit.place(width=50, x=300, y=350)
root.mainloop()

# Button Open Folder to view Saved files


