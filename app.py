import streamlit as st
import requests, pytz, joblib, math, numpy as np, pandas as pd, re, gspread, json
from datetime import datetime
from google.oauth2.service_account import Credentials

# --- CONFIGURATION GOOGLE SHEETS (SOLUTION 2) ---
def save_to_google_sheets(data_row):
    try:
        # Autorisations pour Google Sheets et Drive
        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        
        # On récupère la chaîne JSON brute stockée dans tes Secrets Streamlit
        json_info = st.secrets["gcp_service_account"]["json_data"]
        
        # On transforme cette chaîne en dictionnaire Python
        creds_dict = json.loads(json_info)
        
        # Connexion avec les identifiants du robot
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(credentials)
        
        # Ouverture du tableur par son nom exact
        sheet = client.open("SLOT_Beta1").sheet1
        sheet.append_row(data_row)
        return True
    except Exception as e:
        st.error(f"Erreur de connexion au Google Sheets : {e}")
        return False

# --- FONCTION MÉTÉO DYNAMIQUE (API OPEN-METEO) ---
def get_live_weather(lat, lon):
    try:
        # Appel à l'API gratuite (sans clé)
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code"
        res = requests.get(url, timeout=5).json()
        temp = res['current']['temperature_2m']
        code = res['current']['weather_code']
        
        # Traduction des codes météo de l'OMM vers tes 3 options
        if code in [0, 1]: 
            mto = "Beau"
        elif code in [2, 3, 45, 48]: 
            mto = "Nuageux"
        else: 
            mto = "Pluie" # Codes 51+ (bruine, pluie, neige, orage)
            
        return mto, temp
    except Exception:
        # Valeurs par défaut si l'API météo échoue
        return "Beau", 18.0

# --- NETTOYAGE DU NOM DE LA RUE ---
def extraire_nom_propre(adresse):
    s = adresse.upper()
    s = re.sub(r'\d+', '', s) # Enlever les numéros
    mots_a_supprimer = ["RUE DU ", "RUE DES ", "RUE DE LA ", "RUE DE ", "RUE ", "AVENUE ", "BOULEVARD ", "PLACE ", "D'"]
    for mot in mots_a_supprimer: 
        s = s.replace(mot, "")
    return s.strip()

# --- CHARGEMENT DES MODÈLES IA ---
@st.cache_resource
def load_assets():
    try:
        m = joblib.load("modele_lightgbmDA.pkl")
        p = joblib.load("preprocessorDA.pkl")
        return m, p
    except Exception as e:
        st.error(f"Erreur chargement modèle : {e}")
        return None, None

model, prepro = load_assets()

# --- INTERFACE UTILISATEUR ---
st.set_page_config(page_title="IA Parking Paris - Collecte", page_icon="🅿️")

st.title("🅿️ IA Parking Paris")
st.markdown("### Assistant de collecte de données terrain")

# Formulaire d'entrée
col_n, col_r = st.columns([1, 3])
with col_n:
    numero = st.text_input("N°")
with col_r:
    rue_saisie = st.text_input("Nom de la rue", placeholder="ex: Rue de Rivoli")

# --- ACTION 1 : ANALYSE ---
if st.button("1. ANALYSER LA ZONE ET PRÉDIRE"):
    if not rue_saisie:
        st.warning("Veuillez saisir un nom de rue.")
    else:
        with st.spinner('Récupération des données réelles...'):
            # 1. Géolocalisation (API Adresse Gouv)
            geo = requests.get(f"https://api-adresse.data.gouv.fr/search/?q={rue_saisie}+Paris&limit=1").json()
            
            if geo['features']:
                feat = geo['features'][0]
                lat, lon = feat['geometry']['coordinates'][1], feat['geometry']['coordinates'][0]
                arrdt = int(feat['properties']['postcode'][-2:])
                nom_pur = extraire_nom_propre(rue_saisie)
                
                # 2. Météo & Température en temps réel
                mto_live, temp_live = get_live_weather(lat, lon)
                
                # 3. Trafic (via Google Maps API si dispo)
                score_trafic = 0
                API_MAPS = st.secrets.get("GOOGLE_API_KEY")
                if API_MAPS:
                    try:
                        t_url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={lat},{lon}&destinations={lat+0.001},{lon+0.001}&departure_time=now&key={API_MAPS}"
                        t_res = requests.get(t_url).json()
                        if t_res['rows'][0]['elements'][0]['status'] == "OK":
                            e = t_res['rows'][0]['elements'][0]
                            d_normal = e['duration']['value']
                            d_trafic = e['duration_in_traffic']['value']
                            diff = d_trafic - d_normal
                            score_trafic = 100 if diff > 120 else (50 if diff > 60 else 0)
                    except: pass

                # 4. Prédiction avec le modèle
                now = datetime.now(pytz.timezone('Europe/Paris'))
                
                # Création du DataFrame pour le modèle
                X = pd.DataFrame([{
                    'RUE': nom_pur, 'VILLE': 'Paris', 'JOUR': now.strftime("%A"), 
                    'MTO': mto_live, 'TRAFIC': score_trafic, '% PARKING OC': 0.50, 
                    'NBR PLACES': 15, 'REVENUS / H': 2500, 'VEHICULES / H': 0.30,
                    'TEMPERATURE': temp_live,
                    'HEURE_SIN': np.sin(2*np.pi*(now.hour*60+now.minute)/1440),
                    'HEURE_COS': np.cos(2*np.pi*(now.hour*60+now.minute)/1440)
                }])
                
                # Transformation et prédiction
                occ_pred = model.predict(prepro.transform(X))[0]
                pred_places = max(0, math.floor(15 * (1 - occ_pred)))

                # Sauvegarde en mémoire session pour l'enregistrement
                st.session_state['releve'] = {
                    "Date": now.strftime("%d/%m/%Y"),
                    "Heure": now.strftime("%H:%M"),
                    "Numero": numero,
                    "Rue": nom_pur,
                    "Arrdt": arrdt,
                    "Predit_P": pred_places,
                    "Predit_O": f"{round(occ_pred*100)}%",
                    "Mto": mto_live,
                    "Temp": f"{temp_live}°C",
                    "Trafic": score_trafic
                }

                # Affichage des résultats
                st.success(f"**Prédiction : {pred_places} places disponibles** ({round(occ_pred*100)}% d'occupation)")
                st.info(f"📍 {nom_pur} ({arrdt}e) | ☁️ {mto_live} | 🌡️ {temp_live}°C | 🚗 Trafic: {score_trafic}")
            else:
                st.error("Adresse introuvable à Paris.")

# --- ACTION 2 : ENREGISTREMENT ---
if 'releve' in st.session_state:
    st.divider()
    st.subheader("📝 Rapporter le réel (Terrain)")
    reel = st.number_input("Combien de places vois-tu réellement ?", min_value=0, max_value=50, step=1)
    
    if st.button("💾 ENREGISTRER DANS LE GOOGLE SHEETS"):
        data = st.session_state['releve']
        ligne_a_ajouter = [
            data["Date"], data["Heure"], data["Numero"], data["Rue"], data["Arrdt"],
            data["Predit_P"], data["Predit_O"], reel, data["Mto"], data["Temp"], data["Trafic"]
        ]
        
        with st.spinner('Envoi au Sheets...'):
            if save_to_google_sheets(ligne_a_ajouter):
                st.balloons()
                st.success("Données enregistrées ! Merci Louis.")
                # Nettoyage après succès
                del st.session_state['releve']
