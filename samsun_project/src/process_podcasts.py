import os
import re
import time
import logging
from datetime import datetime
import numpy as np
from collections import Counter
from pymongo import MongoClient
from hdfs import InsecureClient
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === Конфигурация ===
HDFS_URL = os.getenv("HDFS_URL", "http://namenode:9870")
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://mongodb:27017")
LOCAL_DATA_DIR = "/app/data/raw/transcripts"
HDFS_RAW_PATH = "/podcasts/raw"


def wait_for_hdfs(max_wait=120):
    """Ожидание готовности HDFS"""
    logger.info(f"⏳ Ожидание HDFS (макс. {max_wait} сек)...")

    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            client = InsecureClient(HDFS_URL, user='root', timeout=10)
            status = client.status('/')
            logger.info("✅ HDFS готов!")
            return client
        except Exception as e:
            logger.debug(f"Ожидание HDFS... ({e})")
            time.sleep(5)




def upload_local_files_to_hdfs():
    """Загружает локальные файлы в HDFS"""
    client = wait_for_hdfs()

    # Создаем директорию в HDFS
    try:
        client.makedirs(HDFS_RAW_PATH)
        logger.info(f"✅ Создана директория HDFS: {HDFS_RAW_PATH}")
    except:
        logger.info(f"Директория уже существует: {HDFS_RAW_PATH}")

    # Загружаем файлы
    if not os.path.exists(LOCAL_DATA_DIR):
        logger.error(f" Локальная директория не найдена: {LOCAL_DATA_DIR}")
        return

    files = [f for f in os.listdir(LOCAL_DATA_DIR) if f.endswith('.txt')]
    logger.info(f" Найдено локальных файлов: {len(files)}")

    for filename in files:
        local_path = os.path.join(LOCAL_DATA_DIR, filename)
        hdfs_path = f"{HDFS_RAW_PATH}/{filename}"

        try:
            with open(local_path, 'rb') as f:
                client.write(hdfs_path, f, overwrite=True)
            logger.info(f"✅ Загружен в HDFS: {filename}")
        except Exception as e:
            logger.error(f"Ошибка загрузки {filename}: {e}")



def time_to_seconds(time_str):
    try:
        m, s = map(int, time_str.split(':'))
        return m * 60 + s
    except:
        return 0


def parse_transcript(text):
    pattern = r'\[(\d{2}:\d{2}) - (\d{2}:\d{2})\]\s*(.+?)(?=\[\d{2}:\d{2}|$)'
    dialogues = []

    if not text or not text.strip():
        return dialogues

    for match in re.finditer(pattern, text, re.MULTILINE | re.DOTALL):
        try:
            start_time, end_time, speaker_text = match.groups()
            duration = time_to_seconds(end_time) - time_to_seconds(start_time)

            if duration <= 0:
                continue

            dialogues.append({
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'text': speaker_text.strip(),
                'words_count': len(speaker_text.split())
            })
        except Exception as e:
            logger.debug(f"Ошибка парсинга диалога: {e}")
            continue

    return dialogues


def simple_sentiment(text):
    """Простой анализ настроения"""
    if not text:
        return 0

    text_lower = text.lower()
    positive = len(re.findall(r'хорош|круто|класс|отлично|спасибо|супер|прекрасн|люблю|замечательн', text_lower))
    negative = len(re.findall(r'плох|ужасн|отвратительн|ненавижу|раздражает|скучн', text_lower))
    return positive - negative


def extract_keywords(texts, top_n=5):
    if not texts:
        return []

    stop_words = {'и', 'в', 'на', 'с', 'по', 'для', 'не', 'что', 'это', 'как', 'а', 'но', 'да', 'ты', 'я', 'ещё', 'там',
                  'вот', 'так', 'же', 'ну', 'или'}
    all_words = []

    for text in texts:
        words = re.findall(r'\b\w{3,}\b', text.lower())
        all_words.extend([w for w in words if w not in stop_words and len(w) > 2])

    word_counts = Counter(all_words)
    return [word for word, count in word_counts.most_common(top_n)]


def analyze_speakers(dialogues):
    if not dialogues:
        return {"Спикер1": 0, "Спикер2": 0}

    speakers = Counter()
    for dialog in dialogues:
        # Простая эвристика для определения спикера
        first_words = dialog['text'][:30].lower()
        if any(w in first_words for w in ['я', 'мне', 'мой', 'мое']):
            speaker = "Спикер1"
        elif any(w in first_words for w in ['ты', 'вы', 'ваш']):
            speaker = "Спикер2"
        else:
            speaker = "Спикер1"  # default

        speakers[speaker] += dialog['words_count']

    return dict(speakers)


def process_episode(filename, text):
    """Обработка одного эпизода"""
    dialogues = parse_transcript(text)

    if not dialogues:
        logger.warning(f" Нет диалогов в файле: {filename}")
        return None

    # Основные метрики
    texts = [d['text'] for d in dialogues]
    total_duration = sum(d['duration'] for d in dialogues)
    total_words = sum(d['words_count'] for d in dialogues)

    words_per_minute = total_words / (total_duration / 60) if total_duration > 0 else 0

    # ML-анализ
    keywords = extract_keywords(texts, top_n=3)
    speakers_words = analyze_speakers(dialogues)
    sentiments = [simple_sentiment(d['text']) for d in dialogues]
    avg_sentiment = np.mean(sentiments) if sentiments else 0

    # Формируем запись
    record = {
        "episode_id": os.path.splitext(filename)[0],
        "filename": filename,
        "dialogues_count": len(dialogues),
        "total_duration_sec": int(total_duration),
        "total_duration_min": round(total_duration / 60, 1),
        "total_words": total_words,
        "words_per_minute": round(words_per_minute, 1),
        "avg_dialogue_duration": round(np.mean([d['duration'] for d in dialogues]), 1),
        "avg_sentiment": round(avg_sentiment, 2),
        "keywords": keywords,
        "topics": ", ".join(keywords[:3]),  # Для удобства отображения
        "speaker1_words": speakers_words.get("Спикер1", 0),
        "speaker2_words": speakers_words.get("Спикер2", 0),
        "speaker_balance": round(speakers_words.get("Спикер1", 0) / max(total_words, 1), 2),
        "processed_at": datetime.now().isoformat(),
        "raw_dialogues": dialogues[:10]  # Сохраняем первые 10 диалогов для отладки
    }

    return record


def process_episodes_from_hdfs():
    """Загрузка и обработка файлов из HDFS"""
    records = []

    try:
        client = wait_for_hdfs()

        # Получаем список файлов из HDFS
        try:
            files = client.list(HDFS_RAW_PATH)
            logger.info(f" Найдено файлов в HDFS: {len(files)}")
        except Exception as e:
            logger.error(f" Ошибка чтения HDFS: {e}")
            # Пробуем использовать локальные файлы как fallback
            return process_episodes_local()

        # Обрабатываем каждый файл
        for filename in files:
            if not filename.endswith('.txt'):
                continue

            logger.info(f"🔍 Обработка: {filename}")

            try:
                # Чтение из HDFS
                with client.read(f"{HDFS_RAW_PATH}/{filename}") as reader:
                    content = reader.read()
                    text = content.decode('utf-8')

                record = process_episode(filename, text)
                if record:
                    records.append(record)
                    logger.info(f"✅ {filename}: {record['dialogues_count']} диалогов, "
                                f"{record['words_per_minute']:.1f} слов/мин")

            except UnicodeDecodeError:
                # Пробуем другую кодировку
                try:
                    with client.read(f"{HDFS_RAW_PATH}/{filename}") as reader:
                        content = reader.read()
                        text = content.decode('cp1251', errors='replace')
                    record = process_episode(filename, text)
                    if record:
                        records.append(record)
                except Exception as e:
                    logger.error(f"❌ Ошибка кодировки {filename}: {e}")
            except Exception as e:
                logger.error(f"❌ Ошибка обработки {filename}: {e}")

    except Exception as e:
        logger.error(f"❌ Критическая ошибка HDFS: {e}")
        records = process_episodes_local()  # Fallback на локальные файлы

    return records


def process_episodes_local():
    """Fallback: обработка локальных файлов"""
    records = []

    if not os.path.exists(LOCAL_DATA_DIR):
        logger.error(f"❌ Локальная директория не найдена: {LOCAL_DATA_DIR}")
        return records

    files = [f for f in os.listdir(LOCAL_DATA_DIR) if f.endswith('.txt')]
    logger.info(f"🔄 Использую локальные файлы: {len(files)}")

    for filename in files:
        filepath = os.path.join(LOCAL_DATA_DIR, filename)

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            try:
                with open(filepath, 'r', encoding='cp1251') as f:
                    text = f.read()
            except Exception as e:
                logger.error(f"❌ Ошибка чтения {filename}: {e}")
                continue

        record = process_episode(filename, text)
        if record:
            records.append(record)

    return records


def save_to_mongodb(records):
    """Сохранение в MongoDB"""
    if not records:
        logger.warning("⚠️ Нет данных для сохранения в MongoDB")
        return

    try:
        client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')  # Проверка подключения

        db = client["podcasts_db"]
        coll = db["podcasts_ml"]

        # Очищаем коллекцию
        coll.delete_many({})

        # Вставляем новые данные
        result = coll.insert_many(records)

        logger.info(f"✅ MongoDB: сохранено {len(result.inserted_ids)} записей")
        client.close()

    except Exception as e:
        logger.error(f"❌ Ошибка MongoDB: {e}")
        # Создаем fallback - сохраняем в JSON
        import json
        with open('/app/data/fallback_data.json', 'w') as f:
            json.dump(records, f, indent=2, default=str)
        logger.info("📁 Данные сохранены в fallback JSON файл")


def save_to_parquet(records):
    """Сохранение в Parquet для Streamlit"""
    if not records:
        logger.warning("⚠️ Нет данных для Parquet")
        return

    try:
        df = pd.DataFrame(records)

        # Удаляем временные колонки
        columns_to_drop = ['_id', 'processed_at', 'raw_dialogues']
        for col in columns_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Создаем директорию
        parquet_dir = "/app/data/parquet"
        os.makedirs(parquet_dir, exist_ok=True)

        # Сохраняем
        parquet_path = os.path.join(parquet_dir, "podcasts_ml.parquet")
        df.to_parquet(parquet_path, index=False)

        logger.info(f"✅ Parquet сохранен: {parquet_path} ({len(df)} записей)")
        logger.info(f"📊 Колонки: {list(df.columns)}")

    except Exception as e:
        logger.error(f"❌ Ошибка сохранения Parquet: {e}")


def main():
    """Основной ETL процесс"""
    logger.info("=" * 50)
    logger.info("🚀 ЗАПУСК ML-ETL ПРОЦЕССА")
    logger.info("=" * 50)

    start_time = time.time()

    # 1. Загружаем файлы в HDFS
    logger.info("📤 Шаг 1: Загрузка файлов в HDFS...")
    upload_local_files_to_hdfs()

    # 2. Обрабатываем данные из HDFS
    logger.info("🔧 Шаг 2: Обработка данных из HDFS...")
    records = process_episodes_from_hdfs()

    if not records:
        logger.error("❌ Нет обработанных данных! Проверьте файлы в data/raw/transcripts/")
        return

    # 3. Сохраняем в MongoDB
    logger.info("💾 Шаг 3: Сохранение в MongoDB...")
    save_to_mongodb(records)

    # 4. Сохраняем в Parquet
    logger.info("💾 Шаг 4: Сохранение в Parquet...")
    save_to_parquet(records)

    # 5. Завершение
    total_time = time.time() - start_time
    logger.info("=" * 50)
    logger.info(f"🎉 ETL УСПЕШНО ЗАВЕРШЕН!")
    logger.info(f"⏱️  Время выполнения: {total_time:.1f} секунд")
    logger.info(f"📊 Обработано эпизодов: {len(records)}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()