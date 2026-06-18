import streamlit as st
from ultralytics import YOLOWorld
from PIL import Image
import json
import os
import numpy as np

# 1. Seite konfigurieren
st.set_page_config(page_title="Fisch-Erkennung KI", page_icon="🐟", layout="centered")

st.title("🐟 Spezialisierte Fisch-Erkennung")
st.write("Dank YOLO-World erkennt diese KI Fische anhand von Texteingaben – ganz ohne eigenes Training!")

# 2. Liste der Fische, die gesucht werden sollen
fisch_arten = ["pike fish", "perch fish", "zander fish", "carp fish", "trout fish"]

# 3. Modell & Zusatzdaten laden
@st.cache_resource
def load_yolo_world():
    model = YOLOWorld("yolov8l-world.pt")
    model.set_classes(fisch_arten)
    return model

@st.cache_data
def load_fish_data():
    if os.path.exists("fische_daten.json"):
        with open("fische_daten.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

model = load_yolo_world()
fish_info_db = load_fish_data()

# 4. Bild-Upload
uploaded_file = st.file_uploader("Wähle ein Bild deines Fangs...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild öffnen und sicherstellen, dass es im RGB-Modus ist
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Dein Fang")
        
        # KI-Erkennung ausführen
        results = model(image, conf=0.20)
        
        # Boxen zeichnen lassen
        res_plotted = results[0].plot()
        
        # FIX: Wir konvertieren das Array absolut sicher in ein PIL-Bild.
        # YOLO plot liefert standardmäßig BGR als NumPy Array. 
        # Wir drehen die Kanäle um ([:, :, ::-1]) und machen ein PIL Image daraus.
        if isinstance(res_plotted, np.ndarray):
            res_rgb_array = res_plotted[:, :, ::-1]  # BGR zu RGB
            final_image = Image.fromarray(res_rgb_array)
        else:
            final_image = image # Fallback, falls Plot fehlschlägt
        
        # Bild in Streamlit anzeigen (ohne anfällige Parameter)
        st.image(final_image, caption="KI Analyse")

    with col2:
        st.subheader("Analyse-Ergebnisse")
        
        detected_classes = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0]) * 100
                detected_classes.append((class_name, confidence))
        
        if not detected_classes:
            st.warning("Es wurde leider kein Fisch erkannt. Versuche ein schärferes Foto.")
        else:
            unique_classes = list(set([c[0] for c in detected_classes]))
            
            for f_class in unique_classes:
                max_conf = max([c[1] for c in detected_classes if c[0] == f_class])
                st.success(f"Gefunden: **{f_class.upper()}** ({max_conf:.1f}%)")
                
                mapping = {
                    "pike fish": "hecht",
                    "perch fish": "barsch",
                    "zander fish": "zander",
                    "carp fish": "karpfen",
                    "trout fish": "forelle"
                }
                
                db_key = mapping.get(f_class, f_class)
                
                if db_key in fish_info_db:
                    info = fish_info_db[db_key]
                    st.markdown(f"### 📋 Infos zu: {info['name']}")
                    st.info(f"⏳ **Schonzeit:** {info['schonzeit']}")
                    st.warning(f"📏 **Mindestmaß:** {info['mindestmass']}")
                    st.write(f"💡 **Angler-Tipp:** {info['tipp']}")
                else:
                    st.info(f"Keine Schonzeit-Infos für '{db_key}' in fische_daten.json.")
