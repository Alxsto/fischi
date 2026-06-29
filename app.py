import streamlit as st
from ultralytics import YOLOWorld
from PIL import Image
import numpy as np

# 1. Seite konfigurieren
st.set_page_config(
    page_title="FishID Pro – Intelligente Fischerkennung", 
    page_icon="🐟", 
    layout="wide"
)

# Custom CSS für das Design
st.markdown("""
    <style>
        .main {
            background-color: #f4f6f9;
        }
        h1 {
            color: #0A3641 !important;
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 700;
        }
        .fisch-card {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            margin-bottom: 20px;
            border-left: 5px solid #0A3641;
        }
        .meta-label {
            font-weight: bold;
            color: #4A5568;
        }
    </style>
""", unsafe_allow_html=True)

# 2. Header-Bereich
st.title("🐟 FishID Pro")
st.subheader("Spezialisierte Erkennung & Schonzeiten für Angler")

with st.expander("ℹ️ Wie funktioniert FishID Pro?", expanded=False):
    st.write("""
        1. Mache ein möglichst scharfes Foto deines Fangs (am besten flach von der Seite).
        2. Lade das Bild hier hoch.
        3. Unsere KI gleicht das Bild in Echtzeit mit den Merkmalen von Süß- und Salzwasserfischen ab und liefert dir die geltenden Schonzeiten und Mindestmaße.
    """)

# 3. KI Konfiguration
fisch_arten = [
    "pike fish", "perch fish", "zander fish", "carp fish", "trout fish",
    "cod fish", "herring fish", "mackerel fish", "plaice fish", "flounder fish",
    "sea trout fish", "garfish"
]

fish_info_db = {
    "hecht": {
        "name": "Hecht (Esox lucius)",
        "typ": "Süßwasser Raubfisch",
        "schonzeit": "15. Februar bis 30. April (Abweichungen je nach Bundesland)",
        "mindestmass": "45 - 60 cm",
        "tipp": "Vorsicht beim Kiemengriff – Hechte besitzen bis zu 700 messerscharfe Zähne!"
    },
    "barsch": {
        "name": "Flussbarsch (Perca fluviatilis)",
        "typ": "Süßwasser Raubfisch",
        "schonzeit": "Meist ganzjährig frei",
        "mindestmass": "In der Regel kein gesetzliches Mindestmaß (Küchenmaß ca. 20-25 cm)",
        "tipp": "Achtung vor der stachligen Rückenflosse und den Dornen an den Kiemendeckeln."
    },
    "zander": {
        "name": "Zander (Sander lucioperca)",
        "typ": "Süßwasser Raubfisch",
        "schonzeit": "01. April bis 31. Mai (Laichschonzeit)",
        "mindestmass": "45 - 50 cm",
        "tipp": "Zander sind extrem lichtscheu. Beste Fangchancen hast du in der Dämmerung, nachts oder bei trübem Wasser."
    },
    "karpfen": {
        "name": "Spiegel- / Schuppenkarpfen (Cyprinus carpio)",
        "typ": "Süßwasser Friedfisch",
        "schonzeit": "Meist keine Schonzeit (regional stark unterschiedlich)",
        "mindestmass": "35 - 40 cm",
        "tipp": "Karpfen haben eine enorme Ausdauer im Drill. Stelle deine Rollenbremse präzise ein."
    },
    "forelle": {
        "name": "Bachforelle (Salmo trutta fario)",
        "typ": "Süßwasser Salmonide",
        "schonzeit": "20. Oktober bis 15. März (Herbstlaicher)",
        "mindestmass": "25 - 30 cm",
        "tipp": "Forellen jagen auf Sicht und sind extrem schreckhaft. Meide auffällige Kleidung und pirsche dich vorsichtig an."
    },
    "dorsch": {
        "name": "Dorsch / Kabeljau (Gadus morhua)",
        "typ": "Salzwasser (Nord- und Ostsee)",
        "schonzeit": "Ganzjähriges Fangverbot für Freizeitangler (Strenges EU-Baglimit beachten!)",
        "mindestmass": "38 cm",
        "tipp": "Derzeit gilt in vielen Bereichen ein striktes Entnahmeverbot. Beifänge müssen unverzüglich und maximal schonend zurückgesetzt werden."
    },
    "hering": {
        "name": "Atlantischer Hering (Clupea harengus)",
        "typ": "Salzwasser (Nord- und Ostsee)",
        "schonzeit": "Keine Schonzeit (Jedoch Quoten- und Zonenregelungen beachten)",
        "mindestmass": "Kein gesetzliches Mindestmaß (Ostsee regional teils 11 cm)",
        "tipp": "Zieht im Frühjahr zum Laichen in riesigen Schwärmen an die Küsten. Heringspaternoster ohne zusätzlichen Köder verwenden."
    },
    "makrele": {
        "name": "Makrele (Scomber scombrus)",
        "typ": "Salzwasser (Nordsee / westl. Ostsee)",
        "schonzeit": "Keine Schonzeit",
        "mindestmass": "30 cm (nur in der Nordsee gesetzlich)",
        "tipp": "Pfeilschnelle Oberflächenjäger. Kommen im Hochsommer in Küstennähe und liefern spektakuläre Drills am leichten Gerät."
    },
    "scholle": {
        "name": "Scholle (Pleuronectes platessa)",
        "typ": "Salzwasser (Plattfisch)",
        "schonzeit": "01. Februar bis 30. April (Schutz gilt primär für weibliche Schollen / 'Laichschollen')",
        "mindestmass": "25 cm",
        "tipp": "Nutze Perlen oder kleine Spinnerblättchen am Vorfach, um die Neugier der Schollen am Grund zu wecken."
    },
    "flunder": {
        "name": "Flunder (Platichthys flesus)",
        "typ": "Salzwasser / Brackwasser",
        "schonzeit": "01. Februar bis 30. April (Gilt regional nur für Weibchen in der Ostsee)",
        "mindestmass": "25 cm",
        "tipp": "Flundern vertragen Süßwasser und ziehen weit in Flussmündungen hinein. Ein absoluter Top-Fisch für den Räucherofen."
    },
    "meerforelle": {
        "name": "Meerforelle (Salmo trutta trutta)",
        "typ": "Küsten-Salmonide",
        "schonzeit": "01. Oktober bis 31. Dezember (Betrifft meist nur gefärbte Fische im Laichkleid, blanke Fische oft frei)",
        "mindestmass": "40 - 45 cm",
        "tipp": "Auch bekannt als der 'Fisch der 1000 Würfe'. Strecke machen, Wassertrübung im Auge behalten und küstennahe Krautkanten anwerfen."
    },
    "hornhecht": {
        "name": "Hornhecht (Belone belone)",
        "typ": "Salzwasser (Saisonal)",
        "schonzeit": "Keine Schonzeit",
        "mindestmass": "40 cm (Schleswig-Holstein) / 45 cm (Mecklenburg-Vorpommern)",
        "tipp": "Taucht im Mai ('Wenn der Raps blüht') massenhaft an den Küsten auf. Die markanten grünen Gräten sind völlig ungiftig!"
    }
}

@st.cache_resource
def load_yolo_world():
    model = YOLOWorld("yolov8l-world.pt")
    model.set_classes(fisch_arten)
    return model

model = load_yolo_world()

# 4. Hauptbereich: Upload-Zone
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

st.markdown("---")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1.2, 1.0], gap="large")
    
    with col1:
        st.markdown("### 📸 Foto-Analyse")
        with st.spinner("KI analysiert das Bild..."):
            results = model(image, conf=0.25)
            res_plotted = results[0].plot()
            
            if isinstance(res_plotted, np.ndarray):
                res_rgb_array = res_plotted[:, :, ::-1]
                final_image = Image.fromarray(res_rgb_array)
            else:
                final_image = image
            
            st.image(final_image, caption="Analysiertes Fangfoto mit KI-Erkennungsrahmen", use_container_width=True)

    with col2:
        st.markdown("### 📊 Analyse-Ergebnisse")
        
        detected_classes = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0]) * 100
                detected_classes.append((class_name, confidence))
        
        if not detected_classes:
            st.error("❌ **Kein Fisch erkannt.**")
            st.info("💡 **Tipp:** Achte darauf, dass der Fisch gut belichtet ist und flach von der Seite fotografiert wurde.")
        else:
            unique_classes = list(set([c[0] for c in detected_classes]))
            
            for f_class in unique_classes:
                max_conf = max([c[1] for c in detected_classes if c[0] == f_class])
                
                mapping = {
                    "pike fish": "hecht", "perch fish": "barsch", "zander fish": "zander",
                    "carp fish": "karpfen", "trout fish": "forelle", "cod fish": "dorsch",
                    "herring fish": "hering", "mackerel fish": "makrele", "plaice fish": "scholle",
                    "flounder fish": "flunder", "sea trout fish": "meerforelle", "garfish": "hornhecht"
                }
                db_key = mapping.get(f_class, f_class)
                
                st.markdown(f"#### Erkannte Art: **{db_key.upper()}**")
                st.progress(int(max_conf))
                st.caption(f"KI-Sicherheit: {max_conf:.1f}%")
                
                if db_key in fish_info_db:
                    info = fish_info_db[db_key]
                    
                    st.markdown(f"""
                    <div class="fisch-card">
                        <h3 style='margin-top:0; color:#0A3641;'>{info['name']}</h3>
                        <p><span class="meta-label">🐟 Kategorie:</span> {info['typ']}</p>
                        <p style='color: #b91c1c;'><span class="meta-label" style='color: #b91c1c;'>⏳ Schonzeit:</span> {info['schonzeit']}</p>
                        <p style='color: #047857;'><span class="meta-label" style='color: #047857;'>📏 Mindestmaß:</span> {info['mindestmass']}</p>
                        <hr style='border: 0; border-top: 1px solid #e2e8f0; margin: 10px 0;'>
                        <p style='font-style: italic; color: #4A5568;'><span class="meta-label">💡 Angler-Praxistipp:</span> {info['tipp']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning(f"Keine detaillierten Schonzeit-Regelungen für '{db_key}' im System hinterlegt.")
