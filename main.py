import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import io
from pathlib import Path
import torch

# Настройка страницы
st.set_page_config(
    page_title="Обнаружение пешеходов на аэроснимках",
    page_icon="🚶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок приложения
st.title("🚶 Обнаружение пешеходов на аэроснимках")
st.markdown("### Автоматическое обнаружение пешеходов с помощью YOLOv11l")
st.markdown("---")

# Путь к модели в GitHub (относительный путь)
MODEL_PATH = "model/best.pt"  # Папка model в корне репозитория

# Проверка GPU (в облаке Streamlit GPU нет, но оставим для совместимости)
def check_gpu():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return True, gpu_name, gpu_memory
    return False, None, None

@st.cache_resource
def load_model():
    """Загрузка модели YOLO из GitHub репозитория"""
    try:
        # Проверяем существование файла модели
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Модель не найдена по пути: {MODEL_PATH}")
            st.info("""
            **Убедитесь, что:**
            1. Файл модели `best.pt` находится в папке `model/`
            2. Файл добавлен в GitHub репозиторий
            3. Название файла точно `best.pt`
            """)
            
            # Показываем содержимое папки model для отладки
            if os.path.exists("model"):
                files = os.listdir("model")
                st.write(f"Файлы в папке model: {files}")
            return None
        
        # Загружаем модель
        with st.spinner("🔄 Загрузка модели YOLOv11l..."):
            model = YOLO(MODEL_PATH)
        
        # В облаке Streamlit всегда используем CPU
        model.to('cpu')
        st.success("✅ Модель успешно загружена!")
        return model
        
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {e}")
        return None

# Инициализация session_state для хранения результатов
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = {}

# Функция отрисовки рамок
def draw_boxes_with_confidence(image, boxes, conf_threshold=0.25, show_conf=True, thickness=2, color=(0, 255, 0)):
    """Отрисовка bounding boxes на изображении"""
    img_copy = image.copy()
    
    if boxes is None or len(boxes) == 0:
        return img_copy
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        
        if conf < conf_threshold:
            continue
        
        # Рисуем рамку
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
        
        # Рисуем уверенность
        if show_conf:
            label = f"{conf:.2f}"
            font_scale = 0.6
            font_thickness = 1
            
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(img_copy, (x1, y1 - text_h - 6), (x1 + text_w + 6, y1), color, -1)
            cv2.putText(img_copy, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
    
    return img_copy

# Функция обработки одного изображения
def process_single_image(uploaded_file, model, confidence, iou, max_det, show_conf, box_thickness, color):
    """Обработка одного загруженного изображения"""
    try:
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            temp_path = tmp_file.name
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            cv2.imwrite(temp_path, img_bgr)
        
        # Детекция
        results = model(temp_path, conf=confidence, iou=iou, max_det=max_det)
        
        # Отрисовка
        result_img = draw_boxes_with_confidence(
            img_bgr, 
            results[0].boxes, 
            conf_threshold=confidence,
            show_conf=show_conf,
            thickness=box_thickness,
            color=color
        )
        
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        boxes_count = len(results[0].boxes) if results[0].boxes is not None else 0
        
        # Собираем информацию о найденных объектах
        confidences = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                confidences.append(float(box.conf[0]))
        
        # Очистка
        os.unlink(temp_path)
        
        return result_img_rgb, boxes_count, confidences
        
    except Exception as e:
        st.error(f"Ошибка обработки изображения: {e}")
        return None, 0, []

# Боковая панель с настройками
with st.sidebar:
    st.header("⚙️ Настройки детекции")
    
    confidence = st.slider(
        "Порог уверенности (conf)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.25,
        step=0.05,
        help="Чем ниже значение, тем больше объектов будет найдено, но выше риск ложных срабатываний"
    )
    
    iou = st.slider(
        "Порог NMS (iou)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.3,
        step=0.05,
        help="Пересечение рамок. Меньшее значение = меньше дубликатов"
    )
    
    max_det = st.number_input(
        "Максимум детекций на изображение",
        min_value=1,
        max_value=100,
        value=15,
        step=5,
        help="Максимальное количество пешеходов, которое может найти модель"
    )
    
    st.markdown("---")
    st.header("🎨 Визуализация")
    
    show_conf = st.checkbox("Показывать уверенность", value=True)
    box_thickness = st.slider("Толщина рамки", 1, 5, 2)
    
    # Выбор цвета рамки
    box_color = st.color_picker("Цвет рамки", "#00FF00")
    box_color_rgb = tuple(int(box_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    box_color_bgr = (box_color_rgb[2], box_color_rgb[1], box_color_rgb[0])
    
    st.markdown("---")
    st.header("💻 Системная информация")
    
    # Информация об оборудовании
    gpu_available, gpu_name, gpu_memory = check_gpu()
    if gpu_available:
        st.success(f"✅ GPU: {gpu_name[:30]}")
        st.info(f"Память: {gpu_memory:.1f} GB")
    else:
        st.info("🖥️ Используется CPU (облачная среда)")
        st.caption("Streamlit Cloud использует только CPU")
    
    st.markdown("---")
    st.header("📖 Инструкция")
    st.markdown("""
    **Как использовать:**
    1. Загрузите изображения через вкладку "📤 Загрузка"
    2. Настройте параметры детекции (при необходимости)
    3. Нажмите "🚀 Обработать все изображения"
    4. Скачайте результаты с рамками
    
    **Советы:**
    - Если пешеходы не находятся → уменьшите порог уверенности до 0.15
    - При ложных срабатываниях → увеличьте порог до 0.4-0.5
    - Поддерживаются форматы: JPG, PNG, BMP, TIFF
    """)
    
    st.markdown("---")
    st.header("ℹ️ О приложении")
    st.info(f"""
    **Модель:** YOLOv11l  
    **Задача:** Обнаружение пешеходов  
    **Тип данных:** Аэроснимки  
    **Платформа:** Streamlit Cloud
    
    📌 Изображения с пешеходами автоматически получают префикс "Пешеход_"
    """)

# Загрузка модели
with st.spinner("🚀 Загрузка модели YOLOv11l..."):
    model = load_model()

if model is None:
    st.stop()

# Основные вкладки
tab1, tab2, tab3 = st.tabs(["📤 Загрузка изображений", "📊 Статистика", "❓ Помощь"])

# Вкладка 1: Загрузка изображений
with tab1:
    st.subheader("📤 Загрузите изображения для обработки")
    
    uploaded_files = st.file_uploader(
        "Выберите одно или несколько изображений",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        accept_multiple_files=True,
        help="Можно выбрать несколько файлов одновременно"
    )
    
    if uploaded_files:
        # Кнопка массовой обработки
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Обработать все изображения", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Очищаем предыдущие результаты
                st.session_state.processed_images = {}
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"🔄 Обработка: {uploaded_file.name} ({idx+1}/{len(uploaded_files)})")
                    
                    # Обрабатываем изображение
                    result_img, boxes_count, confidences = process_single_image(
                        uploaded_file, model, confidence, iou, max_det,
                        show_conf, box_thickness, box_color_bgr
                    )
                    
                    if result_img is not None:
                        # Сохраняем результат
                        st.session_state.processed_images[uploaded_file.name] = {
                            'result': result_img,
                            'boxes_count': boxes_count,
                            'confidences': confidences,
                            'original': Image.open(uploaded_file)
                        }
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("✅ Обработка завершена!")
                st.rerun()
        
        # Отображение результатов
        if st.session_state.processed_images:
            st.markdown("---")
            st.subheader("📊 Результаты обработки")
            
            for filename, data in st.session_state.processed_images.items():
                with st.expander(f"📷 {filename} - найдено {data['boxes_count']} пешеходов", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🖼️ Оригинал:**")
                        st.image(data['original'], use_container_width=True)
                    
                    with col2:
                        st.markdown("**🎯 Результат детекции:**")
                        st.image(data['result'], use_container_width=True)
                        
                        # Статистика по изображению
                        st.markdown(f"**👥 Найдено пешеходов:** {data['boxes_count']}")
                        
                        if data['confidences']:
                            avg_conf = np.mean(data['confidences'])
                            st.markdown(f"**📊 Средняя уверенность:** {avg_conf:.3f}")
                        
                        # Кнопка скачивания
                        result_pil = Image.fromarray(data['result'])
                        buf = io.BytesIO()
                        result_pil.save(buf, format="JPEG", quality=95)
                        buf.seek(0)
                        
                        # Добавляем префикс "Пешеход_" если есть находки
                        name, ext = os.path.splitext(filename)
                        if data['boxes_count'] > 0:
                            download_name = f"Пешеход_{name}{ext}"
                        else:
                            download_name = filename
                        
                        st.download_button(
                            label="💾 Скачать результат",
                            data=buf,
                            file_name=download_name,
                            mime="image/jpeg",
                            key=f"download_{filename}"
                        )
    
    else:
        st.info("👈 Начните с загрузки изображений в левой панели")

# Вкладка 2: Статистика
with tab2:
    st.header("📊 Статистика обработки")
    
    if st.session_state.processed_images:
        # Собираем статистику
        stats_data = []
        for filename, data in st.session_state.processed_images.items():
            stats_data.append({
                "Файл": filename,
                "Найдено пешеходов": data['boxes_count'],
                "Средняя уверенность": f"{np.mean(data['confidences']):.3f}" if data['confidences'] else "—"
            })
        
        df = pd.DataFrame(stats_data)
        
        # Общая статистика
        col1, col2, col3 = st.columns(3)
        with col1:
            total_people = df["Найдено пешеходов"].sum()
            st.metric("Всего обнаружено пешеходов", total_people)
        
        with col2:
            images_with_people = len(df[df["Найдено пешеходов"] > 0])
            st.metric("Изображения с пешеходами", images_with_people)
        
        with col3:
            avg_people = df["Найдено пешеходов"].mean()
            st.metric("Среднее на изображение", f"{avg_people:.1f}")
        
        st.markdown("---")
        
        # Таблица
        st.subheader("📋 Детальная таблица")
        st.dataframe(df, use_container_width=True)
        
        # График
        st.subheader("📈 Визуализация")
        st.bar_chart(df.set_index("Файл")["Найдено пешеходов"])
        
        # Фильтр
        st.subheader("🔍 Фильтр")
        show_only_with_people = st.checkbox("Показать только изображения с пешеходами")
        
        if show_only_with_people:
            filtered = {k: v for k, v in st.session_state.processed_images.items() if v['boxes_count'] > 0}
            st.write(f"Найдено изображений с пешеходами: **{len(filtered)}**")
            
            for filename, data in filtered.items():
                st.markdown(f"- **{filename}**: {data['boxes_count']} пешеходов")
    else:
        st.info("📭 Нет данных для отображения. Обработайте изображения в первой вкладке.")

# Вкладка 3: Помощь
with tab3:
    st.header("❓ Часто задаваемые вопросы")
    
    with st.expander("🤔 Почему модель не находит пешеходов?"):
        st.markdown("""
        **Возможные причины:**
        1. **Слишком высокий порог уверенности** - попробуйте уменьшить до 0.15-0.2
        2. **Качество изображения** - модель обучена на аэроснимках, попробуйте другое фото
        3. **Масштаб** - пешеходы должны быть достаточно крупными на снимке
        
        **Решение:** Настройте параметры в боковой панели
        """)
    
    with st.expander("⚡ Как ускорить работу?"):
        st.markdown("""
        **Советы:**
        - Обрабатывайте не более 5-10 изображений за раз
        - Используйте изображения меньшего размера
        - Увеличьте порог уверенности - это уменьшит количество ложных срабатываний и ускорит работу
        - Streamlit Cloud использует CPU, поэтому скорость ограничена
        """)
    
    with st.expander("📦 Какие форматы поддерживаются?"):
        st.markdown("""
        **Поддерживаемые форматы:**
        - JPG / JPEG
        - PNG
        - BMP
        - TIFF
        
        Максимальный размер файла: ~200 МБ (ограничение Streamlit)
        """)
    
    with st.expander("🔧 Как обновить модель?"):
        st.markdown("""
        **Для обновления модели:**
        1. Замените файл `model/best.pt` в репозитории GitHub
        2. Streamlit Cloud автоматически перезагрузит приложение при следующем запуске
        3. Или нажмите "Redeploy" в настройках приложения
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 14px;'>"
    "🚶 Обнаружение пешеходов на аэроснимках | YOLOv11l | Работает в Streamlit Cloud"
    "</div>",
    unsafe_allow_html=True
)