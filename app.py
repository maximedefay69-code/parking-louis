import streamlit as st
import requests, pytz, joblib, math, numpy as np, pandas as pd, gspread, json
from datetime import datetime
from google.oauth2.service_account import Credentials

# --- 1. CONFIGURATION & DONNÉES ---
DATA_ARRDT = {
    1: {"REV_M": 2916, "VEH": 0.32}, 2: {"REV_M": 2666, "VEH": 0.28}, 3: {"REV_M": 2833, "VEH": 0.25},
    4: {"REV_M": 2750, "VEH": 0.26}, 5: {"REV_M": 3166, "VEH": 0.35}, 6: {"REV_M": 3750, "VEH": 0.38},
    7: {"REV_M": 4166, "VEH": 0.42}, 8: {"REV_M": 4000, "VEH": 0.40}, 9: {"REV_M": 3000, "VEH": 0.30},
    10: {"REV_M": 2333, "VEH": 0.22}, 11: {"REV_M": 2583, "VEH": 0.24}, 12: {"REV_M": 2500, "VEH": 0.33},
    13: {"REV_M": 2250, "VEH": 0.35}, 14: {"REV_M": 2583, "VEH": 0.36}, 15: {"REV_M": 2916, "VEH": 0.38},
    16: {"REV_M": 4583, "VEH": 0.45}, 17: {"REV_M": 3083, "VEH": 0.37}, 18: {"REV_M": 2083, "VEH": 0.20},
    19: {"REV_M": 1833, "VEH": 0.22}, 20: {"REV_M": 1916, "VEH": 0.23}
}

# --- 2. LA FONCTION DE RECHERCHE "ULTRA-TOLÉRANTE" ---
def obtenir_places_total(type_louis, nom_louis, arrdt_gps):
    try:
        nom_cherche = nom_louis.upper().strip()
        # On interroge l'API sur le nom de la voie et l'arrondissement
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
                # On récupère les données de l'API
                t_api = str(item.get('typevoie', '')).upper()
                n_api = str(item.get('nomvoie', '')).upper()
                regime = str(item.get('regpri', '')).upper()
                nb_places_segment = item.get('placal', 0) # C'est ta colonne "Nombre de places réelles"

                # Logique de correspondance du TYPE (Boulevard -> BD, etc.)
                match_type = False
                t_louis_up = type_louis.upper()
                
                if t_louis_up == "BOULEVARD" and ("BD" in t_api or "BOULEVARD" in t_api): match_type = True
                elif t_louis_up == "AVENUE" and ("AV" in t_api or "AVENUE" in t_api): match_type = True
                elif t_louis_up in t_api: match_type = True # Pour Rue, Place, Quai...

                # Si le nom correspond ET le type correspond ET le régime est valide
                if nom_cherche in n_api and match_type:
                    if any(r in regime for r in regimes_valides):
                        somme_places += nb_places_segment
                        voies_trouvees.append(f"{t_api} {n_api} ({regime})")
        
        # On dédoublonne les noms pour l'affichage
        voies_uniques = list(set(voies_trouvees))
        return somme_places, voies_uniques
    except:
        return 15, ["Erreur API - Valeur par défaut utilisée"]

# --- 3. AUTRES FONCTIONS (INCHANGÉES) ---
def save_to_sheets(row):
    try:
        creds = Credentials.from_service_account_info(json.loads(st.secrets["gcp_service_account"]["json_data"]), 
                scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
        gspread.authorize(creds).open("SLOT_Beta1").sheet1.append_row(row)
        return True
    except: return False

def get_weather(lat, lon):
    try:
        r = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code").json()
        m = "Beau" if r['current']['weather_code'] in [0,1] else ("Nuageux" if r['current']['weather_code'] in [2,3,45,48] else "Pluie")
        return m, r['current']['temperature_2m']
    except: return "Beau", 18.0

@st.cache_resource
def load_assets():
    try: return joblib.load("modele_lightgbmDA.pkl"), joblib.load("preprocessorDA.pkl")
    except: return None, None

model, prepro = load_assets()

# --- 4. INTERFACE ---
st.set_page_config(page_title="SLOT - Collecte", page_icon="🅿️")
st.title("🅿️ SLOT - Assistant Terrain")

col1, col2, col3 = st.columns([1, 2, 3])
num_voie = col1.text_input("N°")
type_voie = col2.selectbox("Type", ["Rue", "Boulevard", "Avenue", "Place", "Quai", "Impasse"])
nom_rue_saisi = col3.text_input("Nom (ex: Rivoli)")

if st.button("🚀 ANALYSER LA RUE"):
    if not nom_rue_saisi:
        st.error("Veuillez saisir un nom de rue.")
    else:
        with st.spinner("Analyse en cours..."):
            # A. GÉOLOCALISATION
            geo = requests.get(f"https://api-adresse.data.gouv.fr/search/?q={num_voie}+{type_voie}+{nom_rue_saisi}+Paris&limit=1").json()
            
            if 'features' in geo and len(geo['features']) > 0:
                f = geo['features'][0]
                lat, lon = f['geometry']['coordinates'][1], f['geometry']['coordinates'][0]
                cp = f['properties']['postcode']
                arrdt = int(cp[-2:])
                
                # B. SOMME DES PLACES OPEN DATA (CHIRURGICAL)
                total_places_api, liste_voies = obtenir_places_total(type_voie, nom_rue_saisi, arrdt)
                
                # C. MÉTÉO & SOCIO
                mto, temp = get_weather(lat, lon)
                socio = DATA_ARRDT.get(arrdt, {"REV_M": 2500, "VEH": 0.30})
                
                # D. PRÉDICTION
                now = datetime.now(pytz.timezone('Europe/Paris'))
                df_ia = pd.DataFrame([{
                    'RUE': nom_rue_saisi.upper(), 'VILLE': 'Paris', 'JOUR': now.strftime("%A"), 
                    'MTO': mto, 'TRAFIC': 0, '% PARKING OC': 0.50, 
                    'NBR PLACES': total_places_api, 'REVENUS / H': socio["REV_M"], 
                    'VEHICULES / H': socio["VEH"], 'TEMPERATURE': temp,
                    'HEURE_SIN': np.sin(2*np.pi*(now.hour*60+now.minute)/1440),
                    'HEURE_COS': np.cos(2*np.pi*(now.hour*60+now.minute)/1440)
                }])
                
                occ = model.predict(prepro.transform(df_ia))[0]
                libres = max(0, math.floor(total_places_api * (1 - occ)))

                # --- AFFICHAGE POUR LOUIS ---
                st.markdown(f"### 📍 Résultats pour le {arrdt}ème")
                
                # La "Feature" demandée : Nombre de places réelles trouvées
                st.info(f"🔍 **Open Data Paris :** J'ai trouvé un total de **{total_places_api} places** (Regimes: Payant/Gratuit).")
                with st.expander("Détails des segments trouvés"):
                    for v in liste_voies: st.write(f"- {v}")

                st.success(f"🤖 **Prédiction SLOT :** Environ **{libres} places libres** actuellement.")
                
                st.session_state['data'] = [now.strftime("%d/%m/%Y"), now.strftime("%H:%M"), num_voie, 
                                           f"{type_voie} {nom_rue_saisi}", arrdt, libres, f"{round(occ*100)}%", 
                                           0, mto, temp, 0, total_places_api]
            else:
                st.error("Rue non reconnue par le GPS.")

if 'data' in st.session_state:
    st.divider()
    reel = st.number_input("Combien de places vois-tu réellement ?", min_value=0)
    if st.button("💾 ENREGISTRER LE RELEVÉ"):
        st.session_state['data'][7] = reel
        if save_to_sheets(st.session_state['data']):
            st.balloons()
            st.success("Relevé enregistré dans SLOT_Beta1 !")
            del st.session_state['data']
