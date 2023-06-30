import streamlit as st
from PIL import Image
import pandas as pd
import pickle
import sklearn


# tableau Integration

tableau_embedded_code = """  """

st.session_state.fav_text = "Favourite\nclick predict"
st.session_state.ud_text = "Underdog\nclick predict"

st. set_page_config(layout="wide")

st.markdown("<style> .css-18ni7ap{ visibility: hidden; } div.block-container {padding-top:1rem;} </style>", unsafe_allow_html=True)

df = pd.read_csv("fights_to_analyze.csv")

# prediction

def form(fighter_name, datum):
    fights_df = pickle.load(open("fights_df.pkl", 'rb'))
    vysledek = ''
    skore = 0
    koef = 0.1
    result = ['W' if x == 1 else 'L' for x in
              fights_df['result'][(fights_df['fighter'] == fighter_name) & (fights_df['date'] < datum)]]
    for vyhra in result[:-6:-1]:
        if vyhra == 'W':
            skore += koef
        else:
            skore -= koef
        koef += 0.1
        vysledek += vyhra + ' '
    vysledek = vysledek[:-1]
    return (vysledek, skore)


def head_to_head(_arg1, _arg2):
    svc = pickle.load(open("model.pkl", 'rb'))
    fighters_to_analyze = pickle.load(open("fighters_to_analyze.pkl", 'rb'))
    form_fighter = form(_arg1, "2022-12-12")
    form_opponent = form(_arg2, "2022-12-12")
    h1 = fighters_to_analyze[fighters_to_analyze.fighter == _arg1].copy()
    h1.loc[:, 'form_skore'] = form_fighter[1]
    h2 = fighters_to_analyze[fighters_to_analyze.fighter == _arg2].copy()
    h2.loc[:, 'form_skore'] = form_opponent[1]
    h1.loc[:, "opponent"] = _arg2
    h1 = h1.merge(h2, left_on="opponent", right_on="fighter", how="inner", suffixes=("_fighter", "_opponent"))
    h1 = h1.loc[:,
         ["ground_def_skill_fighter", "ground_att_skill_fighter", "stand_att_skill_fighter", "stand_def_skill_fighter",
          "stamina_fighter", "form_skore_fighter",
          "ground_def_skill_opponent", "ground_att_skill_opponent", "stand_att_skill_opponent",
          "stand_def_skill_opponent", "stamina_opponent", "form_skore_opponent"]]
    probs = svc.predict_proba(h1)
    prob_fighter = probs[0][1]
    prob_opponent = probs[0][0]
    return [prob_fighter, prob_opponent]


# read csv

fights_df = pd.read_csv('fights_df.csv', parse_dates=True)
fighters_df = pd.read_csv('fighters_df.csv')
ufc_fighters = pd.DataFrame(fights_df.drop_duplicates("fighter")["fighter"])

st.markdown("<h2 style='text-align: center; color: white;'>Strike Sense</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: white;'>Current Model Accuracy: 71.66%</h2>", unsafe_allow_html=True)

st.markdown("""   <style>
   .stApp {
   background-image: url('https://static.news.bitcoin.com/wp-content/uploads/2021/07/ufc-partners-with-crypto-com--sources-say-175-million-deal-is-mma-firms-largest-sponsorship.jpg');
   background-size: cover;
   }
   </style>   """, unsafe_allow_html=True)

left = Image.open('fighter_left.png')
right = Image.open('fighter_right.png')

# columns for selection

c1, c2, c3 = st.columns([1, 2, 1])
with c1:
    st.markdown("<h3 style='text-align: center; color: white;'>Favourite Fighter</h3>",
                unsafe_allow_html=True)

    fav_fighter = st.selectbox("Select Fighter", df['fighter_fighter'].unique())


with c3:
    st.markdown("<h3 style='text-align: center; color: white;'>Underdog Fighter</h3>",
                unsafe_allow_html=True)

    ud_fighter = st.selectbox("Select Fighter", df['opponent'].unique(), key="underdog fighter")

#button function
def btn_click():
    prob_list = head_to_head('Conor McGregor', 'Khabib Nurmagomedov')
    st.session_state.fav_text = f"{round(prob_list[0] * 100, 2)}%"
    st.session_state.ud_text = f"{round(prob_list[1] * 100, 2)}%"
    c04.text_area('Info_underdog', placeholder=st.session_state.ud_text, label_visibility='hidden')
    c02.text_area('Info', placeholder=st.session_state.fav_text, label_visibility='hidden')
    # switch_page('predict')


col1, col2, col3 = st.columns((0.8, 1, 2))

# predict button
with col3:
    predict_btn = st.button("Predict", on_click=btn_click)


c01, c02, c03, c04, c05 = st.columns(5)

with c01:
    st.image(left, use_column_width=True)

with c02:
    st.title('#')
    st.title('#')
    st.title('#')


with c04:
    st.title('#')
    st.title('#')
    st.title('#')


with c05:
    st.image(right, use_column_width=True)

st.markdown(tableau_embedded_code, unsafe_allow_html=True)

