import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys
from pymongo import MongoClient

# Добавляем путь для импорта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Настройка страницы
st.set_page_config(
    layout="wide",
    page_title="🎙️ Podcast ML Analytics",
    page_icon="🎙️"
)

st.title("Аналитика Подкастов")

@st.cache_data(ttl=300)  # Кешируем на 5 минут
def load_data():
    """Загрузка данных из Parquet или MongoDB"""

    # Пробуем загрузить из Parquet
    parquet_path = '/app/data/parquet/podcasts_ml.parquet'
    if os.path.exists(parquet_path):
        try:
            df = pd.read_parquet(parquet_path)
            st.sidebar.success(f"Данные из Parquet: {len(df)} записей")
            return df
        except Exception as e:
            st.sidebar.warning(f"Ошибка Parquet: {e}")

    # Fallback: загрузка из MongoDB
    try:
        client = MongoClient("mongodb://mongodb:27017/", serverSelectionTimeoutMS=5000)
        db = client["podcasts_db"]
        coll = db["podcasts_ml"]

        # Получаем данные
        data = list(coll.find({}, {'_id': 0, 'raw_dialogues': 0}))

        if data:
            df = pd.DataFrame(data)
            st.sidebar.success(f" Данные из MongoDB: {len(df)} записей")
            return df
        else:
            st.sidebar.warning(" В MongoDB нет данных")

    except Exception as e:
        st.sidebar.error(f" Ошибка MongoDB: {e}")

    # Если ничего не загрузилось
    return pd.DataFrame()


# Загружаем данные
df = load_data()

# Преобразование типов данных
numeric_cols = ['words_per_minute', 'avg_sentiment', 'dialogues_count',
                'total_duration_min', 'speaker_balance', 'total_words']

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Удаляем строки с NaN в ключевых полях
df_clean = df.dropna(subset=['words_per_minute', 'avg_sentiment'], how='any').copy()



# ================== SIDEBAR ФИЛЬТРЫ ==================
st.sidebar.header("🔍 Фильтры данных")

# Фильтр по эпизодам
if 'episode_id' in df_clean.columns:
    episodes = sorted(df_clean['episode_id'].unique())
    selected_episodes = st.sidebar.multiselect(
        "Выберите эпизоды:",
        episodes,
        default=episodes[:3] if len(episodes) > 3 else episodes
    )
    df_filtered = df_clean[df_clean['episode_id'].isin(selected_episodes)].copy()
else:
    df_filtered = df_clean.copy()

# Динамические диапазоны для слайдеров
if not df_filtered.empty:
    # Sentiment range
    sent_min = float(df_filtered['avg_sentiment'].min())
    sent_max = float(df_filtered['avg_sentiment'].max())
    if sent_min == sent_max:
        sent_range = [sent_min - 0.5, sent_max + 0.5]
    else:
        sent_range = [sent_min, sent_max]

    # WPM range
    wpm_min = float(df_filtered['words_per_minute'].min())
    wpm_max = float(df_filtered['words_per_minute'].max())
    if wpm_min == wpm_max:
        wpm_range = [max(0, wpm_min - 10), wpm_max + 10]
    else:
        wpm_range = [wpm_min, wpm_max]

    # Слайдеры
    sentiment_range = st.sidebar.slider(
        "Диапазон эмоциональности",
        sent_range[0],
        sent_range[1],
        (sent_range[0], sent_range[1]),
        step=0.1
    )

    wpm_range = st.sidebar.slider(
        "Темп речи (слов/мин)",
        int(wpm_range[0]),
        int(wpm_range[1]) + 1,
        (int(wpm_range[0]), int(wpm_range[1])),
        step=1
    )

    # Применяем фильтры
    df_filtered = df_filtered[
        (df_filtered['avg_sentiment'] >= sentiment_range[0]) &
        (df_filtered['avg_sentiment'] <= sentiment_range[1]) &
        (df_filtered['words_per_minute'] >= wpm_range[0]) &
        (df_filtered['words_per_minute'] <= wpm_range[1])
        ].copy()

# ================== ОСНОВНАЯ ПАНЕЛЬ ==================
st.header("📊 Ключевые метрики")


# KPI метрики
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Эпизодов", len(df_filtered))

with col2:
    avg_wpm = df_filtered['words_per_minute'].mean()
    st.metric("Средний темп речи", f"{avg_wpm:.1f} слов/мин")

with col3:
    avg_sent = df_filtered['avg_sentiment'].mean()
    sentiment_label = " Позитивный" if avg_sent > 0 else " Нейтральный" if avg_sent == 0 else " Негативный"
    st.metric("Средняя эмоциональность", f"{avg_sent:+.2f}", sentiment_label)

with col4:
    total_words = int(df_filtered['total_words'].sum()) if 'total_words' in df_filtered.columns else "N/A"
    st.metric("Всего слов", total_words)

# ================== ГРАФИКИ ==================
st.header("📈 Визуализация данных")

tab1, tab2, tab3 = st.tabs(["📊 Распределения", "🔗 Корреляции", "🎤 Спикеры"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        try:
            fig1 = px.histogram(
                df_filtered,
                x='words_per_minute',
                nbins=20,
                title="Распределение темпа речи",
                labels={'words_per_minute': 'Слов в минуту'},
                color_discrete_sequence=['#636EFA']
            )
            st.plotly_chart(fig1, width='stretch')
        except Exception as e:
            st.error(f"Ошибка построения гистограммы: {e}")

    with col2:
        try:
            fig2 = px.scatter(
                df_filtered,
                x='total_duration_min',
                y='avg_sentiment',
                size='dialogues_count',
                color='words_per_minute',
                hover_data=['episode_id'],
                title="Зависимость эмоциональности от длительности",
                labels={
                    'total_duration_min': 'Длительность (мин)',
                    'avg_sentiment': 'Эмоциональность',
                    'words_per_minute': 'Темп речи'
                }
            )
            st.plotly_chart(fig2, width='stretch')
        except Exception as e:
            st.error(f"Ошибка построения scatter plot: {e}")

with tab2:
    # Корреляционная матрица
    corr_cols = ['words_per_minute', 'avg_sentiment', 'dialogues_count',
                 'total_duration_min', 'speaker_balance']
    available_cols = [col for col in corr_cols if col in df_filtered.columns]

    if len(available_cols) >= 2:
        try:
            corr_matrix = df_filtered[available_cols].corr()
            fig_corr = px.imshow(
                corr_matrix,
                text_auto='.2f',
                color_continuous_scale='RdBu',
                title="Корреляция между метриками",
                aspect="auto"
            )
            st.plotly_chart(fig_corr, width='stretch')
        except Exception as e:
            st.error(f"Ошибка построения матрицы: {e}")
    else:
        st.info("Недостаточно данных для корреляционного анализа")

with tab3:
    # Анализ спикеров
    if 'speaker_balance' in df_filtered.columns:
        col1, col2 = st.columns(2)

        with col1:
            avg_balance = df_filtered['speaker_balance'].mean()
            speaker_data = pd.DataFrame({
                'Спикер': ['Спикер 1', 'Спикер 2'],
                'Доля слов': [avg_balance, 1 - avg_balance]
            })

            fig_pie = px.pie(
                speaker_data,
                values='Доля слов',
                names='Спикер',
                title="Среднее распределение слов между спикерами",
                color_discrete_sequence=['#FF6B6B', '#4ECDC4']
            )
            st.plotly_chart(fig_pie, width='stretch')

        with col2:
            if 'speaker1_words' in df_filtered.columns and 'speaker2_words' in df_filtered.columns:
                df_speakers = df_filtered[['episode_id', 'speaker1_words', 'speaker2_words']].melt(
                    id_vars=['episode_id'],
                    var_name='Спикер',
                    value_name='Количество слов'
                )

                fig_bar = px.bar(
                    df_speakers,
                    x='episode_id',
                    y='Количество слов',
                    color='Спикер',
                    barmode='group',
                    title="Распределение слов по эпизодам",
                    color_discrete_map={'speaker1_words': '#FF6B6B', 'speaker2_words': '#4ECDC4'}
                )
                st.plotly_chart(fig_bar, width='stretch')

# ================== ТАБЛИЦА ДАННЫХ ==================
st.header("Детальные данные")

# Выбираем колонки для отображения
display_cols = ['episode_id', 'words_per_minute', 'avg_sentiment', 'topics',
                'dialogues_count', 'total_duration_min', 'speaker_balance']

available_cols = [col for col in display_cols if col in df_filtered.columns]


# Статус системы
st.sidebar.header(" Статус системы")
st.sidebar.metric("Записей в данных", len(df_filtered))
st.sidebar.metric("Колонок", len(df_filtered.columns))
st.sidebar.info(f"Обновлено: {pd.Timestamp.now().strftime('%H:%M:%S')}")

