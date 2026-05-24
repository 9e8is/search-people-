import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from ultralytics import YOLO
from PIL import Image
import io
import torch

# Настройка страницы
st.set_page_config(
    page_title="Обнаружение пешеходов на аэроснимках",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок приложения
st.title("Обнаружение пешеходов на аэроснимках")
st.markdown("---")

# Путь к модели в GitHub (относительный путь)
MODEL_PATH = "model/best.pt"

# Стандартные параметры детекции (фиксированные)
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 300

# Проверка GPU
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
        if not os.path.exists(MODEL_PATH):
            st.error(f"Модель не найдена по пути: {MODEL_PATH}")
            st.info()
            
            if os.path.exists("model"):
                files = os.listdir("model")
                st.write(f"Файлы в папке model: {files}")
            return None
        
        with st.spinner("Загрузка модели YOLOv11l..."):
            model = YOLO(MODEL_PATH)
        
        model.to('cpu')
        st.success("Модель успешно загружена")
        return model
        
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None

# Инициализация session_state для хранения результатов
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = {}

# Функция отрисовки рамок
def draw_boxes_with_confidence(image, boxes, show_conf=True, thickness=2, color=(0, 255, 0)):
    img_copy = image.copy()
    
    if boxes is None or len(boxes) == 0:
        return img_copy
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
        
        if show_conf:
            label = f"{conf:.2f}"
            font_scale = 0.6
            font_thickness = 1
            
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(img_copy, (x1, y1 - text_h - 6), (x1 + text_w + 6, y1), color, -1)
            cv2.putText(img_copy, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
    
    return img_copy

# Функция обработки одного изображения
def process_single_image(uploaded_file, model, show_conf, box_thickness, color):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            temp_path = tmp_file.name
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            cv2.imwrite(temp_path, img_bgr)
        
        # Используем стандартные параметры
        results = model(temp_path, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, max_det=MAX_DETECTIONS)
        
        result_img = draw_boxes_with_confidence(
            img_bgr, 
            results[0].boxes, 
            show_conf=show_conf,
            thickness=box_thickness,
            color=color
        )
        
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        boxes_count = len(results[0].boxes) if results[0].boxes is not None else 0
        
        os.unlink(temp_path)
        
        return result_img_rgb, boxes_count
        
    except Exception as e:
        st.error(f"Ошибка обработки изображения: {e}")
        return None, 0

# Боковая панель с настройками
with st.sidebar:
    st.header("Настройки визуализации")
    
    show_conf = st.checkbox("Показывать уверенность", value=True)
    box_thickness = st.slider("Толщина рамки", 1, 5, 2)
    
    # Выбор цвета рамки из нескольких вариантов
    color_options = {
        "Зеленый": (0, 255, 0),
        "Красный": (0, 0, 255),
        "Синий": (255, 0, 0),
        "Желтый": (0, 255, 255),
        "Оранжевый": (0, 165, 255),
        "Розовый": (255, 192, 203),
        "Голубой": (255, 255, 0),
        "Фиолетовый": (128, 0, 128)
    }
    
    selected_color = st.selectbox("Цвет рамки", list(color_options.keys()), index=0)
    box_color_bgr = color_options[selected_color]
    
    st.markdown("---")
    st.header("Системная информация")
    
    gpu_available, gpu_name, gpu_memory = check_gpu()
    if gpu_available:
        st.success(f"GPU: {gpu_name[:30]}")
        st.info(f"Память: {gpu_memory:.1f} GB")
    else:
        st.info("Используется CPU (облачная среда)")
    
    st.markdown("---")
    st.header("Инструкция")
    st.markdown("""
    Как использовать:
    1. Загрузите изображения
    2. Нажмите Обработать все изображения
    3. Скачайте результаты
    
    Советы:
    - Если пешеходы не находятся, попробуйте другие изображения
    - Поддерживаются форматы: JPG, PNG, BMP, TIFF
    """)
    
    st.markdown("---")
    st.info(f"""
    Модель: YOLOv11l
    Задача: Обнаружение пешеходов
    Тип данных: Аэроснимки
    Параметры детекции: conf=0.25, iou=0.45
    """)

# Загрузка модели
with st.spinner("Загрузка модели YOLOv11l..."):
    model = load_model()

if model is None:
    st.stop()

# Основная часть
st.subheader("Загрузите изображения для обработки")

uploaded_files = st.file_uploader(
    "Выберите одно или несколько изображений",
    type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
    accept_multiple_files=True
)

if uploaded_files:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Обработать все изображения", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            st.session_state.processed_images = {}
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Обработка: {uploaded_file.name} ({idx+1}/{len(uploaded_files)})")
                
                result_img, boxes_count = process_single_image(
                    uploaded_file, model, show_conf, box_thickness, box_color_bgr
                )
                
                if result_img is not None:
                    st.session_state.processed_images[uploaded_file.name] = {
                        'result': result_img,
                        'boxes_count': boxes_count
                    }
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("Обработка завершена")
            st.rerun()
    
    if st.session_state.processed_images:
        st.markdown("---")
        st.subheader("Результаты обработки")
        
        for filename, data in st.session_state.processed_images.items():
            with st.expander(f"{filename} - найдено {data['boxes_count']} пешеходов", expanded=True):
                st.image(data['result'], use_container_width=True)
                
                st.markdown(f"**Найдено пешеходов:** {data['boxes_count']}")
                
                result_pil = Image.fromarray(data['result'])
                buf = io.BytesIO()
                result_pil.save(buf, format="JPEG", quality=95)
                buf.seek(0)
                
                name, ext = os.path.splitext(filename)
                if data['boxes_count'] > 0:
                    download_name = f"Пешеход_{name}{ext}"
                else:
                    download_name = filename
                
                st.download_button(
                    label="Скачать результат",
                    data=buf,
                    file_name=download_name,
                    mime="image/jpeg",
                    key=f"download_{filename}"
                )

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 14px;'>"
    "Обнаружение пешеходов на аэроснимках | YOLOv11l | Streamlit Cloud"
    "</div>",
    unsafe_allow_html=True
)