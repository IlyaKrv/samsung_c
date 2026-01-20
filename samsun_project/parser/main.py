import os
import yt_dlp
from config import SOURCES
from utils import ensure_dirs, download_audio, transcribe_audio


def get_videos_from_playlist(playlist_url):
    #получает спсиок видео из плейлиста
    print(f"\nПарсим плейлист: {playlist_url}")
    ydl_opts = {'extract_flat': True, 'quiet': False}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
        return info.get('entries', [])


def get_video_info(video_url):
    #получение методанных одного видео youtube or rutube
    print(f"\nПарсим видео: {video_url}")
    ydl_opts = {'quiet': False}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        return {
            'id': info.get('id'),
            'title': info.get('title', 'Без названия'),
            'duration': info.get('duration', 0),
            'url': video_url
        }


def is_playlist_url(url):
    #определение является ли ссылка плейлистом
    return (
        "/playlist" in url or
        "list=" in url or
        "/plst/" in url
    )


def main():
    ensure_dirs()
    all_videos = []

    for url in SOURCES:
        try:
            if is_playlist_url(url):
                videos = get_videos_from_playlist(url)
                print(f"  -> Найдено видео в плейлисте: {len(videos)}")
                all_videos.extend(videos)
            else:
                video_info = get_video_info(url)
                all_videos.append(video_info)
                print(f"  -> Добавлено отдельное видео: {video_info['title'][:50]}...")
        except Exception as e:
            print(f"  -> Ошибка при обработке {url}: {e}")
            continue

    print(f"\nВсего видео для обработки: {len(all_videos)}")

    for i, video in enumerate(all_videos, 1):
        video_id = video['id']
        title = video.get('title', 'Без названия')
        print(f"\n[{i}/{len(all_videos)}] Обрабатываю: {title[:50]}...")

        txt_path = f"transcripts/{video_id}.txt"
        if os.path.exists(txt_path):
            print(f"  -> Уже существует. Пропускаем.")
            continue

        try:
            #Формируем ссылку (для Rutube и YouTube он разный)
            video_url = video.get('url') or f"https://rutube.ru/video/{video_id}/"
            audio_path = download_audio(video_url, video_id)
            output_path = transcribe_audio(audio_path, video_id)
            print(f"  → Готово: {output_path}")

            #Удаляем аудио
            os.remove(audio_path)

        except Exception as e:
            print(f"  -> Ошибка: {e}")
            continue


if __name__ == "__main__":
    main()