import streamlit as st
from ultralytics import YOLOWorld
from PIL import Image
import json
import os
import numpy as np

# 1. Seite konfigurieren
st.set_page_config(page_title="Fisch-Erkennung KI", page_icon="🐟", layout="centered")

st.title("🐟 Spezialisierte Fisch-Erkennung")
st.write("Diese KI erkennt Süß- und Salzwasserfische anhand moderner Text-Bild-Abgleiche – ganz ohne extra JSON-Datei!")

# 2. Erweiterte Liste der Fische (Süßwasser & Nord-/Ostsee)
fisch_arten = [
    "pike fish", "perch fish", "zander fish", "carp fish", "trout fish",
    "cod fish", "herring fish", "mackerel fish", "plaice fish", "flounder fish",
    "sea trout fish", "garfish"
]

# 3. Fisch-Datenbank direkt im Code integriert (Verhindert Fehler mit fehlenden Dateien)
fish_info_db = {
    "hecht": {
        "name": "Hecht (Esox lucius) - Süßwasser",
        "schonzeit": "15. Februar bis 30. April (je nach Bundesland)",
        "mindestmass": "45 - 60 cm",
        "tipp": "Vorsicht beim Kiemengriff – scharfe Zähne!"
    },
    "barsch": {
        "name": "Flussbarsch (Perca fluviatilis) - Süßwasser",
        "schonzeit": "Meist keine Schonzeit",
        "mindestmass": "Oft kein Mindestmaß (regional ca. 20 cm)",
        "tipp": "Stachlige Rückenflosse! Perfekt als Speisefisch."
    },
    "zander": {
        "name": "Zander (Sander lucioperca) - Süßwasser",
        "schonzeit": "01. April bis 31. Mai",
        "mindestmass": "45 - 50 cm",
        "tipp": "Lichtscheu. Beißt am besten in der Dämmerung oder nachts."
    },
    "karpfen": {
        "name": "Spiegel-/Schuppenkarpfen (Cyprinus carpio) - Süßwasser",
        "schonzeit": "Meist keine (bzw. regional unterschiedlich)",
        "mindestmass": "35 - 40 cm",
        "tipp": "Karpfen kämpfen stark! Achte auf eine gut eingestellte Rollenbremse."
    },
    "forelle": {
        "name": "Bachforelle (Salmo trutta fario) - Süßwasser",
        "schonzeit": "20. Oktober bis 15. März",
        "mindestmass": "25 - 30 cm",
        "tipp": "Sehr schreckhaft. Pirsche dich vorsichtig an das Ufer heran."
    },
    "dorsch": {
        "name": "Dorsch / Kabeljau (Gadus morhua) - Nord- und Ostsee",
        "schonzeit": "Ganzjähriges Fangverbot für Freizeitangler (Baglimit beachten!)",
        "mindestmass": "38 cm",
        "tipp": "Aktuell herrscht in der Ostsee ein striktes Fangverbot (Baglimit 0). Beifänge extrem schonend zurücksetzen!"
    },
    "hering": {
        "name": "Hering (Clupea harengus) - Nord- und Ostsee",
        "schonzeit": "Keine direkte Schonzeit, aber Fangquoten beachten",
        "mindestmass": "Kein gesetzliches Mindestmaß",
        "tipp": "Zieht im Frühjahr in riesigen Schwärmen in Küstennähe. Nutze Heringspaternoster ohne zusätzlichen Köder."
    },
    "makrele": {
        "name": "Makrele (Scomber scombrus) - Nordsee / westl. Ostsee",
        "schonzeit": "Keine Schonzeit",
        "mindestmass": "30 cm (Nordsee)",
        "tipp": "Ein pfeilschneller Raubfisch. Kommt im Hochsommer nah an die Küsten. Macht am leichten Gerät viel Spaß."
    },
    "scholle": {
        "name": "Scholle (Pleuronectes platessa) - Nord- und Ostsee",
        "schonzeit": "01. Februar bis 30. April (gilt oft nur für weibliche Schollen)",
        "mindestmass": "25 cm",
        "tipp": "Plattfisch. Beißt hervorragend auf Wattwürmer oder Seeringelwürmer an der Brandungsrute."
    },
    "flunder": {
        "name": "Flunder (Platichthys flesus) - Nord- und Ostsee",
        "schonzeit": "01. Februar bis 30. April (regional unterschiedlich für Weibchen)",
        "mindestmass": "25 cm",
        "tipp": "Verträgt auch Brackwasser und zieht oft weit in Flussmündungen hinein. Schmeckt geräuchert fantastisch."
    },
    "meerforelle": {
        "name": "Meerforelle (Salmo trutta trutta) - Küste (Nord- und Ostsee)",
        "schonzeit": "01. Oktober bis 31. Dezember (gilt an der Küste meist nur für braune Fische im Laichkleid)",
        "mindestmass": "40 - 45 cm",
        "tipp": "Der 'Fisch der 1000 Würfe'. Wathose anziehen und Blinker oder Küstenfliegen weit hinauswerfen."
    },
    "hornhecht": {
        "name": "Hornhecht (Belone belone) - Nord- und Ostsee",
        "schonzeit": "Keine Schonzeit",
        "mindestmass": "40 cm (Schleswig-Holstein) / 45 cm (Mecklenburg-Vorpommern)",
        "tipp": "Kommt im Mai zum Laichen an die Küsten ('Wenn der Raps blüht'). Hat auffällige, ungiftige grüne Gräten!"
    }
}

# 4. Modell laden
@st.cache_resource
def load_yolo_world():
    model = YOLOWorld("yolov8l-world.pt")
    model.set_classes(fisch_arten)
    return model

model = load_yolo_world()

# 5. Bild-Upload
uploaded_file = st.file_uploader("Wähle ein Bild deines Fangs...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Dein Fang")
        
        # KI-Erkennung ausführen
        results = model(image, conf=0.25)
        
        # Boxen zeichnen lassen
        res_plotted = results[0].plot()
        
        # Konvertierung in ein PIL-Bild für Streamlit
        if isinstance(res_plotted, np.ndarray):
            res_rgb_array = res_plotted[:, :, ::-1]  # BGR zu RGB
            final_image = Image.fromarray(res_rgb_array)
        else:
            final_image = image
        
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
                
                # Exaktes Mapping auf unsere Dictionary-Schlüssel oben
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
                    st.info(f"Keine Schonzeit-Infos für '{db_key}' im Code hinterlegt.")
