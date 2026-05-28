@st.cache_resource
def load_model():
    # Lädt deine heruntergeladene Datei aus dem Projektordner
    return YOLO("fisch_modell.pt") 


    with col2:
        st.subheader("Analyse-Ergebnisse")
        
        detected_classes = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id].lower() # Z.B. "hecht", "barsch", "carp"
            confidence = float(box.conf[0]) * 100
            detected_classes.append((class_name, confidence))
        
        if not detected_classes:
            st.warning("Es wurde leider kein Fisch erkannt.")
        else:
            unique_classes = list(set([c[0] for c in detected_classes]))
            
            for f_class in unique_classes:
                max_conf = max([c[1] for c in detected_classes if c[0] == f_class])
                st.success(f"Gefunden: **{f_class.upper()}** ({max_conf:.1f}%)")
                
                # 2. ÄNDERUNG: Wir nutzen jetzt direkt den echten Klassennamen der KI!
                # Wichtig: Die Schlüssel in deiner fische_daten.json müssen genau so heißen,
                # wie die Klassen, die das Roboflow-Modell ausgibt.
                db_key = f_class 
                
                if db_key in fish_info_db:
                    info = fish_info_db[db_key]
                    st.markdown(f"### 📋 Infos zu: {info['name']}")
                    st.info(f"⏳ **Schonzeit:** {info['schonzeit']}")
                    st.warning(f"📏 **Mindestmaß:** {info['mindestmass']}")
                    st.write(f"💡 **Angler-Tipp:** {info['tipp']}")
                else:
                    st.info(f"Keine Infos zu '{f_class}' in fische_daten.json gefunden. Füge diesen Schlüssel hinzu, um Schonzeiten anzuzeigen!")
