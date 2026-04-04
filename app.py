import streamlit as st
import requests, pytz, joblib, math, numpy as np, pandas as pd, json
from datetime import datetime

# --- 1. CONFIGURATION & DICTIONNAIRES ---
# Données socio-économiques par arrondissement (Paris)
DATA_ARRDT = {
    1: {"REV_M": 2916, "VEH": 0.32}, 2: {"REV_M": 2666, "VEH": 0.28}, 3: {"REV_M": 2833, "VEH": 0.25},
    4: {"REV_M": 2750, "VEH": 0.26}, 5: {"REV_M": 3166, "VEH": 0.35}, 6: {"REV_M": 3750, "VEH": 0.38},
    7: {"REV_M": 4166, "VEH": 0.42}, 8: {"REV_M": 4000, "VEH": 0.40}, 9: {"REV_M": 3000, "VEH": 0.30},
    10: {"REV_M": 2333, "VEH": 0.22}, 11: {"REV_M": 2583, "VEH": 0.24}, 12: {"REV_M": 2500, "VEH": 0.33},
    13: {"REV_M": 2250, "VEH": 0.35}, 14: {"REV_M": 2583, "VEH": 0.36}, 15: {"REV_M": 2916, "VEH": 0.38},
    16: {"REV_M": 4583, "VEH": 0.45}, 17: {"REV_M": 3083, "VEH": 0.37}, 18: {"REV_M": 2083, "VEH": 0.20},
    19: {"REV_M": 1833, "VEH": 0.22}, 20: {"REV_M": 1916, "VEH": 0.23}
}

# Traduction pour le modèle entraîné en Français
JOURS_FR = {
    "Monday": "Lundi", "Tuesday": "Mardi", "Wednesday": "Mercredi",
    "Thursday": "Jeudi", "Friday": "Vendredi", "Saturday": "Samedi", "Sunday": "Dimanche"
}

# --- 2. FONCTIONS DE RÉCUPÉRATION DYNAMIQUES ---

def obtenir_places_total(type_louis, nom_louis, arrdt_gps):
    """Calcule la feature 'NBR PLACES' (Somme de la colonne 'placal')"""
    try:
        nom_cherche = nom_louis.upper().strip()
        url = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/stationnement-sur-voie-publique-emprises/records"
        params = {"where": f"suggest(nomvoie, '{nom_cherche}') AND arrond = {arrdt_gps}", "limit": 100}
        res = requests.get(url, params=params, timeout=10).json()
        somme_p, voies = 0, []
        if res and 'results' in res:
            for item in res['results']:
                t_api = str(item.get('typevoie','')).upper()
                n_api = str(item.get('nomvoie','')).upper()
                reg = str(item.get('regpri','')).upper()
                p = item.get('placal', 0)
                
                match = False
                tl = type_louis.upper()
                if tl == "BOULEVARD" and ("BD" in t_api or "BOULEVARD" in t_api): match = True
                elif tl == "AVENUE" and ("AV" in t_api or "AVENUE" in t_api): match = True
                elif tl in t_api: match = True
                
                if nom_cherche in n_api and match:
                    if any(r in reg for r in ["PAYANT ROTATIF", "PAYANT MIXTE", "GRATUIT"]):
                        somme_p += p
                        voies.append(f"{t_api} {n_api}")
        return somme_p, list(set(voies))
    except: return 15, ["Erreur API"]

def get_weather(lat, lon):
    """Récupère météo et température via Open-Meteo"""
    try:
        r = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code").json()
        c = r['current']['weather_code']
        m = "Beau" if c in [0,1] else ("Nuageux" if c in [2,3,45,48] else "Pluie")
        return m, r['current']['temperature_2m']
    except: return "Beau", 18.0

@st.cache_resource
def load_assets():
    """Charge le modèle et le préprocesseur"""
    try: return joblib.load("modele_lightgbmDA.pkl"), joblib.load("preprocessorDA.pkl")
    except: return None, None

model, prepro = load_assets()

# --- 3. INTERFACE STREAMLIT ---
st.set_page_config(page_title="SLOT V65.1", layout="wide")
st.title("🅿️ SLOT - Assistant Terrain")

col_in1, col_in2, col_in3 = st.columns([1, 2, 3])
num_v = col_in1.text_input("N°")
type_v = col_in2.selectbox("Type", ["Rue", "Boulevard", "Avenue", "Place", "Quai", "Impasse"])
nom_v = col_in3.text_input("Nom de la rue (ex: Rivoli)")

if st.button("🚀 ANALYSER LA RUE"):
    if not nom_v:
        st.warning("Précise le nom de la rue !")
    else:
        with st.spinner("Collecte des données..."):
            # A. GPS Dynamique
            geo = requests.get(f"https://api-adresse.data.gouv.fr/search/?q={num_v}+{type_v}+{nom_v}+Paris&limit=1").json()
            
            if 'features' in geo and len(geo['features']) > 0:
                f = geo['features'][0]
                lat, lon = f['geometry']['coordinates'][1], f['geometry']['coordinates'][0]
                arrdt = int(f['properties']['postcode'][-2:])
                
                # B. Récupération Données Dynamiques
                total_p, liste_v = obtenir_places_total(type_v, nom_v, arrdt)
                mto_label, temp_val = get_weather(lat, lon)
                socio = DATA_ARRDT.get(arrdt, {"REV_M": 2500, "VEH": 0.30})
                
                # C. Calculs Temporels
                tz = pytz.timezone('Europe/Paris')
                now = datetime.now(tz)
                minutes_total = now.hour * 60 + now.minute
                
                # --- D. PANEL DE CONTRÔLE (Pour Louis) ---
                st.subheader("📊 Données lues en temps réel")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Météo", mto_label)
                c2.metric("Température", f"{temp_val}°C")
                c3.metric("Trafic", "0.0 (Fluide)")
                c4.metric("Places (Total)", total_p)
                
                with st.expander("🔍 Détails techniques"):
                    st.write(f"Arrondissement : {arrdt}e")
                    st.write(f"Segments trouvés : {', '.join(liste_v)}")

                # --- E. PRÉPARATION DES 15 FEATURES (Ordre strict) ---
                X_dict = {
                    'DATE': now.strftime("%d/%m/%Y"),           # 1
                    'JOUR': JOURS_FR.get(now.strftime("%A")),   # 2
                    'HEURE': now.strftime("%H:%M"),             # 3
                    'RUE': nom_v.upper(),                       # 4
                    'VILLE': "Paris",                           # 5
                    'TRAFIC': 0.0,                              # 6
                    '% PARKING OC': 0.5,                        # 7
                    'NBR PLACES': total_p,                      # 8
                    'REVENUS / H': socio["REV_M"],              # 9
                    'VEHICULES / H': socio["VEH"],              # 10
                    'MTO': mto_label,                           # 11
                    'TEMPERATURE': temp_val,                    # 12
                    'HEURE_MINUTES': minutes_total,             # 13
                    'HEURE_SIN': np.sin(2 * np.pi * minutes_total / 1440), # 14
                    'HEURE_COS': np.cos(2 * np.pi * minutes_total / 1440)  # 15
                }
                
                X_df = pd.DataFrame([X_dict])
                
                # Verrouillage de l'ordre exact demandé
                colonnes_15 = [
                    'DATE', 'JOUR', 'HEURE', 'RUE', 'VILLE', 'TRAFIC', '% PARKING OC',
                    'NBR PLACES', 'REVENUS / H', 'VEHICULES / H', 'MTO', 'TEMPERATURE',
                    'HEURE_MINUTES', 'HEURE_SIN', 'HEURE_COS'
                ]
                X_df = X_df[colonnes_15]

                # F. PRÉDICTION
                try:
                    X_trans = prepro.transform(X_df)
                    occ_pred = model.predict(X_trans)[0]
                    libres = max(0, math.floor(total_p * (1 - occ_pred)))
                    
                    st.divider()
                    st.success(f"🤖 **Résultat IA :** Environ **{libres} places libres** sur {total_p}.")
                    st.progress(occ_pred if 0 <= occ_pred <= 1 else 1.0, text=f"Occupation : {round(occ_pred*100)}%")
                except Exception as e:
                    st.error(f"Erreur modèle : {e}")
                    st.dataframe(X_df) # Pour débugger les colonnes si besoin
            else:
                st.error("Adresse introuvable.")
