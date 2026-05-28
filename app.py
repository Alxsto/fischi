import streamlit as st
from ultralytics import YOLOWorld
from PIL import Image
import json
import os

# 1. Seite konfigurieren
st.set_page_config(page_title="Fisch-Erkennung KI", page_icon="🐟", layout="centered")

st.title("🐟 Spezialisierte Fisch-Erkennung")
st.write("Dank YOLO-World erkennt diese KI Fische anhand von Texteingaben – ganz ohne eigenes Training!")

# 2. Liste der Fische, die gesucht werden sollen
# Tipp: Nutze die englischen oder wissenschaftlichen Namen, da die KI darauf weltklasse trainiert ist.
fisch_arten = ["pike fish", "perch fish", "zander fish", "carp fish", "trout fish"]

# 3. Modell & Zusatzdaten laden
@st.cache_resource
def load_yolo_world():
    # Wir laden das 'l' (large) Modell für hohe Genauigkeit bei feinen Details
    model = YOLOWorld("yolov8l-world.pt")
    # Hier sagen wir der KI, nach welchen Klassen sie suchen soll
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
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Dein Fang")
        # KI-Erkennung ausführen
        results = model(image, conf=0.20) # 20% Sicherheit reicht zum Matchen
        
        # Boxen auf das Bild zeichnen
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="KI Analyse", use_container_width=True)

    with col2:
        st.subheader("Analyse-Ergebnisse")
        
        detected_classes = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0]) * 100
            detected_classes.append((class_name, confidence))
        
        if not detected_classes:
            st.warning("Es wurde leider keine der angegebenen Fischarten erkannt. Versuche ein schärferes Foto.")
        else:
            unique_classes = list(set([c[0] for c in detected_classes]))
            
            for f_class in unique_classes:
                max_conf = max([c[1] for c in detected_classes if c[0] == f_class])
                st.success(f"Gefunden: **{f_class.upper()}** ({max_conf:.1f}%)")
                
                # Wir mappen den englischen KI-Suchbegriff auf unsere deutschen JSON-Daten
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
                    st.info(f"Keine Schonzeit-Infos für '{db_key}' in der Datenbank.")
