import streamlit as st
from PIL import Image
import pandas as pd
import pickle
import sklearn


# tableau Integration

tableau_embedded_code = """ <div class='tableauPlaceholder' id='viz1688148877123' style='position: relative'>
<a href='#'><img alt='Stats (L) ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;St&#47;StrikeSense&#47;StatsL&#47;1_rss.png' style='border: none' /></a>
<object class='tableauViz'  style='display:none;'>
<param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> 
<param name='embed_code_version' value='3' /> <param name='site_root' value='' />
<param name='name' value='StrikeSense&#47;StatsL' /><param name='tabs' value='no' />
<param name='toolbar' value='yes' />
<param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;St&#47;StrikeSense&#47;StatsL&#47;1.png' /> 
<param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' />
<param name='display_overlay' value='yes' />
<param name='display_count' value='yes' />
<param name='language' value='en-US' /></object></div>                

<script type='text/javascript'>                    
var divElement = document.getElementById('viz1688148877123');                    
var vizElement = divElement.getElementsByTagName('object')[0];                    
if ( divElement.offsetWidth > 1000 ) { vizElement.style.width='50%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} 
else if ( divElement.offsetWidth > 400 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} 
else { vizElement.style.width='100%';vizElement.style.height='500px';}                     
var scriptElement = document.createElement('script');                    
scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    
vizElement.parentNode.insertBefore(scriptElement, vizElement);                
</script> """

tableau_embedded_code_2 = """ <div class='tableauPlaceholder' id='viz1688149026839' style='position: relative'><a href='#'><img alt='Stats (R) ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;52&#47;5297ZN55Z&#47;1_rss.png' style='border: none' /></a>
<object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> 
<param name='embed_code_version' value='3' /> <param name='path' value='shared&#47;5297ZN55Z' /> 
<param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;52&#47;5297ZN55Z&#47;1.png' /> 
<param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' />
<param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' />
<param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                
<script type='text/javascript'>                    
var divElement = document.getElementById('viz1688149026839');                    
var vizElement = divElement.getElementsByTagName('object')[0];                    
if ( divElement.offsetWidth > 1000 ) 
{ vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} 
else if ( divElement.offsetWidth > 400 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} 
else { vizElement.style.width='100%';vizElement.style.height='977px';}                     
var scriptElement = document.createElement('script');                    
scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    
vizElement.parentNode.insertBefore(scriptElement, vizElement);                
</script> """

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
    c04.text_area('Info_underdog', value=st.session_state.ud_text, label_visibility='hidden')
    c02.text_area('Info', value=st.session_state.fav_text, label_visibility='hidden')
    # switch_page('predict')


col1, col2, col3 = st.columns((0.8, 1, 2))

# predict button
with col3:
    predict_btn = st.button("Predict", on_click=btn_click)


c01, c02, c03, c04, c05 = st.columns([1,2,2,2,1])

with c01:
    st.image(left, use_column_width=True)

with c02:
    st.title('#')
    st.title('#')



with c04:
    st.title('#')
    st.title('#')



with c05:
    st.image(right, use_column_width=True)


col1,col2,col3 = st.columns([0.5,0.2,0.5])

col1.markdown(tableau_embedded_code, unsafe_allow_html=True)
# col3.markdown(tableau_embedded_code_2, unsafe_allow_html=True)

