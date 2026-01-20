# utils.py
import os
import re
import yt_dlp
import whisper
import torch
import gc
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

def ensure_dirs():
    os.makedirs("downloads", exist_ok=True)
    os.makedirs("transcripts", exist_ok=True)


def download_audio(video_url, video_id):
    audio_path = f"downloads/podcast_{video_id}.opus"
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'opus',
            'preferredquality': '96',       #оптимально для речи
        }],
        'outtmpl': audio_path.replace('.opus', ''),  #убираем расширение для yt-dlp
        'quiet': False,
        'retries': 5,
        'fragment_retries': 10,
        'sleep_interval': 1,
        'ignoreerrors': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    return audio_path

def format_time(s):
    m, s = divmod(int(s), 60)
    return f"{m:02d}:{s:02d}"


def clean_text(text):
    if not text:
        return ""
    cleaned = re.sub(r'[^а-яА-Яa-zA-Z0-9\s.,!?;:"\'()\-—–]', ' ', text)
    return re.sub(r'\s+', ' ', cleaned).strip()

import warnings
warnings.filterwarnings("ignore", message=".*weights.*not initialized.*")
warnings.filterwarnings("ignore", message=".*Failed to determine 'entailment' label id.*")


def detect_topic(full_text):
    #тема подкаста определение
    candidate_labels = [
        "бизнес", "личные отношения", "психология", "искусство", "музыка",
        "технологии", "образование", "здоровье", "путешествия", "история",
        "философия", "политика", "спорт", "наука", "медицина", "религия"
    ]

    try:
        classifier = pipeline(
            "zero-shot-classification",
            model="cointegrated/rubert-tiny2",
            device=0 if torch.cuda.is_available() else -1
        )
        result = classifier(full_text[:500], candidate_labels, multi_label=False)
        return result['labels'][0]
    except Exception as e:
        print(f"  → Ошибка при определении темы: {e}")
        return "не определена"


def transcribe_audio(audio_path, video_id):
    print(f"  -> Загружаем модель...")
    model = whisper.load_model("large-v3-turbo").to(device)

    print(f"  -> Распознаём речь...")
    result = model.transcribe(audio_path, language="ru", verbose=False)

    #текст для определения темы
    full_text = " ".join([seg['text'] for seg in result['segments']])

    #Определяем тему
    topic = detect_topic(full_text)

    #Формируем строки: сначала расшифровка, потом тема
    lines = []

    #Добавляем каждый сегмент с таймкодом
    for seg in result['segments']:
        start = format_time(seg['start'])
        end = format_time(seg['end'])
        text = seg['text'].strip()
        lines.append(f"[{start} - {end}] {text}")

    #dобавляем пустую строку и тему в конце
    lines.append("")
    lines.append(f"Тема подкаста: {topic}")

    #.txt
    txt_path = f"transcripts/{video_id}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    #Очищаем память
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return txt_path