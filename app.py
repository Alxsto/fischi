import streamlit as st
from ultralytics import YOLOWorld
from PIL import Image
import json
import os
import numpy as np

# 1. Seite konfigurieren
st.set_page_config(page_title="Fisch-Erkennung KI", page_icon="🐟", layout="centered")

st.title("🐟 Spezialisierte Fisch-Erkennung")
st.write("Diese KI erkennt Süß- und Salzwasserfische anhand moderner Text-Bild-Abgleiche – ganz ohne eigenes Training!")

# 2. Erweiterte Liste der Fische (Süßwasser & Nord-/Ostsee)
fisch_arten = [
    "pike fish", "perch fish", "zander fish", "carp fish", "trout fish",
    "cod fish", "herring fish", "mackerel fish", "plaice fish", "flounder fish",
    "sea trout fish", "garfish"
]

# 3. Modell & Zusatzdaten laden
@st.cache_resource
def load_yolo_world():
    # Wir nutzen das 'l' (large) Modell für höhere Genauigkeit bei feinen Details
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
        # conf=0.25 filtert zu ungenaue Raten-Versuche heraus
        results = model(image, conf=0.25)
        
        # Boxen zeichnen lassen
        res_plotted = results[0].plot()
        
        # Absolute sichere Konvertierung in ein PIL-Bild für Streamlit
        if isinstance(res_plotted, np.ndarray):
            res_rgb_array = res_plotted[:, :, ::-1]  # BGR zu RGB Kanäle drehen
            final_image = Image.fromarray(res_rgb_array)
        else:
            final_image = image
        
        # Bild in Streamlit anzeigen
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
            st.warning("Es wurde leider kein Fisch erkannt. Versuche ein schärferes Foto aus einem besseren Winkel.")
        else:
            unique_classes = list(set([c[0] for c in detected_classes]))
            
            for f_class in unique_classes:
                max_conf = max([c[1] for c in detected_classes if c[0] == f_class])
                st.success(f"Gefunden: **{f_class.upper()}** ({max_conf:.1f}%)")
                
                # Neues, erweitertes Mapping der englischen Begriffe auf deutsche JSON-Schlüssel
                mapping = {
                    "pike fish": "hecht",
                    "perch fish": "barsch",
                    "zander fish": "zander",
                    "carp fish": "karpfen",
                    "trout fish": "forelle",
                    "cod fish": "dorsch",
                    "herring fish": "hering",
                    "mackerel fish": "makrele",
                    "plaice fish": "scholle",
                    "flounder fish": "flunder",
                    "sea trout fish": "meerforelle",
                    "garfish": "hornhecht"
                }
                
                db_key = mapping.get(f_class, f_class)
                
                if db_key in fish_info_db:
                    info = fish_info_db[db_key]
                    st.markdown(f"### 📋 Infos zu: {info['name']}")
                    st.info(f"⏳ **Schonzeit:** {info['schonzeit']}")
                    st.warning(f"📏 **Mindestmaß:** {info['mindestmass']}")
                    st.write(f"💡 **Angler-Tipp:** {info['tipp']}")
                else:
                    st.info(f"Keine Schonzeit-Infos für '{db_key}' in fische_daten.json hinterlegt.")
