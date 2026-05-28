import streamlit as st
from ultralytics import YOLO
from PIL import Image
import json
import os

# 1. Seite konfigurieren
st.set_page_config(page_title="Fisch-Erkennung KI", page_icon="🐟", layout="centered")

st.title("🐟 Spezialisierte Fisch-Erkennung")
st.write("Lade ein Foto deines Fangs hoch, um die Art und wichtige Infos (Schonzeiten, Mindestmaße) zu erhalten.")

# 2. Daten & Modell laden
@st.cache_resource
def load_model():
    # Wenn du ein eigenes Modell trainiert hast, ersetze 'yolov8n.pt' durch 'best.pt'
    # 'yolov8n.pt' ist das Standard-Modell (erkennt "fish" als allgemeine Klasse)
    return YOLO("yolov8n.pt")

@st.cache_data
def load_fish_data():
    if os.path.exists("fische_daten.json"):
        with open("fische_daten.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

model = load_model()
fish_info_db = load_fish_data()

# 3. Bild-Upload
uploaded_file = st.file_uploader("Wähle ein Bild...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild öffnen
    image = Image.open(uploaded_file)
    
    # Layout aufteilen: Links Original/Erkennung, Rechts Infos
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Dein Fang")
        # KI-Vorhersage ausführen
        # conf=0.25 bedeutet: Zeige nur Dinge, bei denen sich die KI zu 25% sicher ist
        results = model(image, conf=0.25)
        
        # Ergebnis-Bild mit Boxen rendern
        res_plotted = results[0].plot()
        # OpenCV nutzt BGR, Pillow braucht RGB -> Konvertierung falls nötig (YOLO plot liefert RGB)
        st.image(res_plotted, caption="KI Erkennung", use_container_width=True)

    with col2:
        st.subheader("Analyse-Ergebnisse")
        
        # Erkannte Klassen heraussuchen
        detected_classes = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id].lower() # Name der Klasse (z.B. "fish" oder später "hecht")
            confidence = float(box.conf[0]) * 100
            detected_classes.append((class_name, confidence))
        
        if not detected_classes:
            st.warning("Es wurde leider kein Fisch erkannt. Versuche es mit einem schärferen Bild.")
        else:
            # Doppelte Erkennungen herausfiltern
            unique_classes = list(set([c[0] for c in detected_classes]))
            
            for f_class in unique_classes:
                # Höchste Confidence für diese Klasse finden
                max_conf = max([c[1] for c in detected_classes if c[0] == f_class])
                st.success(f"Gefunden: **{f_class.upper()}** (Sicherheit: {max_conf:.1f}%)")
                
                # Zusatzinfos aus JSON laden
                # Falls du das Standard-Modell nutzt, matcht "fish" eventuell nicht. 
                # Hier ist ein Fallback für das Standard 'yolov8n.pt' (Klasse 56 ist 'fish')
                db_key = "hecht" if f_class == "fish" else f_class # Temporärer Platzhalter für den Test
                
                if db_key in fish_info_db:
                    info = fish_info_db[db_key]
                    st.markdown(f"### 📋 Infos zu: {info['name']}")
                    st.info(f"⏳ **Schonzeit:** {info['schonzeit']}")
                    st.warning(f"📏 **Mindestmaß:** {info['mindestmass']}")
                    st.write(f"💡 **Angler-Tipp:** {info['tipp']}")
                else:
                    st.info("Keine Schonzeit-Informationen für diese Fischart in der Datenbank hinterlegt.")
