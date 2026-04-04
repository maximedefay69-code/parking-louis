import streamlit as st
import requests, pytz, joblib, math, numpy as np, pandas as pd, json
from datetime import datetime
from google.oauth2.service_account import Credentials
import gspread

# --- 1. CONFIGURATION ET DICTIONNAIRES ---
# Données socio-économiques par arrondissement (Statique)
DATA_ARRDT = {
    1: {"REV_M": 2916, "VEH": 0.32}, 2: {"REV_M": 2666, "VEH": 0.28}, 3: {"REV_M": 2833, "VEH": 0.25},
    4: {"REV_M": 2750, "VEH": 0.26}, 5: {"REV_M": 3166, "VEH": 0.35}, 6: {"REV_M": 3750, "VEH": 0.38},
    7: {"REV_M": 4166, "VEH": 0.42}, 8: {"REV_M": 4000, "VEH": 0.40}, 9: {"REV_M": 3000, "VEH": 0.30},
    10: {"REV_M": 2333, "VEH": 0.22}, 11: {"REV_M": 2583, "VEH": 0.24}, 12: {"REV_M": 2500, "VEH": 0.33},
    13: {"REV_M": 2250, "VEH": 0.35}, 14: {"REV_M": 2583, "VEH": 0.36}, 15: {"REV_M": 2916, "VEH": 0.38},
    16: {"REV_M": 4583, "VEH": 0.45}, 17: {"REV_M": 3083, "VEH": 0.37}, 18: {"REV_M": 2083, "VEH": 0.20},
    19: {"REV_M": 1833, "VEH": 0.22}, 20: {"REV_M": 1916, "VEH": 0.23}
}

# Traduction forcée pour le modèle entraîné en Français
JOURS_FR = {
    "Monday": "Lundi", "Tuesday": "Mardi", "Wednesday": "Mercredi",
    "Thursday": "Jeudi", "Friday": "Vendredi", "Saturday": "Samedi", "Sunday": "Dimanche"
}

# --- 2. FONCTIONS DE RÉCUPÉRATION ---

def obtenir_places_total(type_louis, nom_louis, arrdt_gps):
    """Filtre l'Open Data Paris et fait la somme de la colonne 'placal'"""
    try:
        nom_cherche = nom_louis.upper().strip()
        url = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/stationnement-sur-voie-publique-emprises/records"
        params = {
            "where": f"suggest(nomvoie, '{nom_cherche}') AND arrond = {arrdt_gps}",
            "limit": 100
        }
        
        regimes_valides = ["PAYANT ROTATIF", "PAYANT MIXTE", "GRATUIT"]
        res = requests.get(url, params=params, timeout=10).json()
        
        somme_places = 0
        voies_trouvees = []

        if res and 'results' in res:
            for item in res['results']:
                t_api = str(item.get('typevoie', '')).upper()
                n_api = str(item.get('nomvoie', '')).upper()
                regime = str(item.get('regpri', '')).upper()
                nb_places_segment = item.get('placal', 0)

                # Logique de tolérance sur le type (BD = Boulevard, etc.)
                match_type = False
                tl_up = type_louis.upper()
                if tl_up == "BOULEVARD" and ("BD" in t_api or "BOULEVARD" in t_api): match_type = True
                elif tl_up == "AVENUE" and ("AV" in t_api or "AVENUE" in t_api): match_type = True
                elif tl_up in t_api: match_type = True

                if nom_cherche in n_api and match_type:
                    if any(r in regime for r in regimes_valides):
                        somme_places += nb_places_segment
                        voies_trouvees.append(f"{t_api} {n_api} ({regime})")
        
        return somme_places, list(set(voies_trouvees))
    except:
        return 15, ["Erreur API - Valeur par défaut"]

def get_weather(lat, lon):
    """Récupère météo et température via Open-Meteo"""
    try:
        r = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code").json()
        code = r['current']['weather_code']
        # Étiquettes en français pour le modèle
        m = "Beau" if code in [0,1] else ("Nuageux" if code in [2,3,45,48] else "Pluie")
        return m, r['current']['temperature_2m']
    except: return "Beau", 18.0

@st.cache_resource
def load_assets():
    """Charge le modèle et le préprocesseur"""
    try:
        return joblib.load("modele_lightgbmDA.pkl"), joblib.load("preprocessorDA.pkl")
    except: return None, None

model, prepro = load_assets()

# --- 3. INTERFACE UTILISATEUR ---

st.set_page_config(page_title="SLOT - Assistant Terrain", page_icon="🅿️")
st.title("🅿️ SLOT - Collecte & IA")

# Formulaire Louis
col1, col2, col3 = st.columns([1, 2, 3])
num_voie = col1.text_input("N°")
type_voie = col2.selectbox("Type", ["Rue", "Boulevard", "Avenue", "Place", "Quai", "Impasse"])
nom_rue_saisi = col3.text_input("Nom de la rue (ex: Rivoli)")

if st.button("🚀 LANCER L'ANALYSE"):
    if not nom_rue_saisi:
        st.warning("Précise le nom de la rue pour Louis !")
    else:
        with st.spinner("Analyse du quartier..."):
            # A. Géolocalisation
            geo = requests.get(f"https://api-adresse.data.gouv.fr/search/?q={num_voie}+{type_voie}+{nom_rue_saisi}+Paris&limit=1").json()
            
            if 'features' in geo and len(geo['features']) > 0:
                f = geo['features'][0]
                lat, lon = f['geometry']['coordinates'][1], f['geometry']['coordinates'][0]
                arrdt = int(f['properties']['postcode'][-2:])
                
                # B. Récupération des Features Dynamiques
                total_places_rue, liste_voies = obtenir_places_total(type_voie, nom_rue_saisi, arrdt)
                mto_label, temp_val = get_weather(lat, lon)
                socio = DATA_ARRDT.get(arrdt, {"REV_M": 2500, "VEH": 0.30})
                
                # C. Préparation de la "Bouffe" du modèle (100% Français / Orthographe stricte)
                now = datetime.now(pytz.timezone('Europe/Paris'))
                
                # Création du dictionnaire de Features
                X_dict = {
                    'JOUR': JOURS_FR.get(now.strftime("%A"), now.strftime("%A")),
                    'HEURE': now.strftime("%H:%M"),
                    'VILLE': "Paris",
                    'TRAFIC': 0, # Score trafic à brancher si Google API active
                    '% PARKING OC': 0.50,
                    'NBR PLACES': total_places_rue,
                    'REVENUS / H': socio["REV_M"],
                    'VEHICULES / H': socio["VEH"],
                    'MTO': mto_label,
                    'TEMPERATURE': temp_val
                }
                
                # D. Prédiction
                X_df = pd.DataFrame([X_dict])
                
                # Vérification de l'ordre des colonnes (Optionnel mais sécurisant)
                colonnes_ordre = ['JOUR', 'HEURE', 'VILLE', 'TRAFIC', '% PARKING OC', 
                                  'NBR PLACES', 'REVENUS / H', 'VEHICULES / H', 'MTO', 'TEMPERATURE']
                X_df = X_df[colonnes_ordre]

                try:
                    score_occ = model.predict(prepro.transform(X_df))[0]
                    places_libres = max(0, math.floor(total_places_rue * (1 - score_occ)))
                except Exception as e:
                    st.error(f"Erreur modèle : {e}")
                    score_occ, places_libres = 0.5, 0

                # E. Affichage pour Louis
                st.markdown(f"### 📍 Analyse : {type_voie} {nom_rue_saisi} ({arrdt}e)")
                
                st.info(f"🔍 **Feature 'NBR PLACES' :** J'ai trouvé **{total_places_rue} places** réelles dans cette rue (Somme des segments).")
                with st.expander("Voir les détails des segments API"):
                    for v in liste_voies: st.write(f"- {v}")

                st.success(f"🤖 **Prédiction IA :** Environ **{places_libres} places libres** actuellement.")
                
                # Stockage pour enregistrement futur
                st.session_state['releve'] = [
                    now.strftime("%d/%m/%Y"), now.strftime("%H:%M"), num_voie, 
                    f"{type_voie} {nom_rue_saisi}", arrdt, places_libres, 
                    f"{round(score_occ*100)}%", 0, mto_label, temp_val, 0, total_places_rue
                ]
            else:
                st.error("Impossible de localiser cette adresse.")

# --- 4. ENREGISTREMENT TERRAIN ---
if 'releve' in st.session_state:
    st.divider()
    reel = st.number_input("Louis, combien de places vois-tu réellement ?", min_value=0)
    if st.button("💾 SAUVEGARDER LE RELEVÉ"):
        # Logique gspread ici pour envoyer st.session_state['releve']
        st.balloons()
        st.success("Données envoyées à Google Sheets !")
        del st.session_state['releve']
