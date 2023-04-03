#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
import numpy.ma as ma
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from surprise import Reader, Dataset, SVD
import streamlit as st


# In[2]:


from PIL import Image

image_path = 'd6yqvmu-5d594ce1-02d7-4e14-af93-5649fbc89a84.jpg'
image = Image.open(image_path)


# In[3]:


# Add the @st.cache decorator to cache the model loading

def load_model():
    model = tf.keras.models.load_model('my_model.h5')
    return model

# Load your model here
model = load_model()


# In[4]:



def load_encoder(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

# In your app.py or any other script where you need the encoder
team_position_encoder = load_encoder('teamPosition_encoder.pkl')


# In[5]:


with open('winPct_standard_scaler.pkl', 'rb') as f:
    winPct_standard_scaler = pickle.load(f)

# Load the saved MinMaxScaler for the 'winPct' column
with open('winPct_minmax_scaler.pkl', 'rb') as f:
    winPct_minmax_scaler = pickle.load(f)


# In[6]:


# # Standardize the 'winPct' column using the loaded StandardScaler
# winPct = winPct_standard_scaler.transform([[0.54]])

# # Normalize the 'winPct' column using the loaded MinMaxScaler
# winPct = winPct_minmax_scaler.transform(winPct)
# winPct


# In[7]:


champion_encoded = pd.read_csv('champion_encoded1.csv')
LOL_data_copy = pd.read_csv('LOL_data_copy.csv')
df = pd.read_csv('df.csv')


# In[8]:


champion_df = pd.read_csv('champion_df.csv')
# create champion dictionary 
champ_dict = dict((champ, index) for index, champ in enumerate(champion_df['id']))
# Create a dictionary that maps champion names to their keys
name_to_key = dict((champ, key) for key, champ in enumerate(champion_df['id']))

key_to_name = {key:champ for champ, key in name_to_key.items()}

# In[9]:


def predict(champ1,champ2,champ3,champ4,champ5,champ6,champ7,champ8,champ9,hotstreak,teamPosition,winPct):
    #preprocessing
    test_text = np.array([champ1,champ2,champ3,champ4,champ5,champ6,champ7,champ8,champ9,hotstreak,teamPosition,winPct])    
    champ1_key = name_to_key[champ1]
    champ2_key = name_to_key[champ2]
    champ3_key = name_to_key[champ3]
    champ4_key = name_to_key[champ4]
    champ5_key = name_to_key[champ5]
    champ6_key = name_to_key[champ6]
    champ7_key = name_to_key[champ7]
    champ8_key = name_to_key[champ8]
    champ9_key = name_to_key[champ9]
    if hotstreak == "Yes":
        hotstreak = 1
    else:
        hotstreak = 0
    teamPosition = team_position_encoder.transform([teamPosition])
    test = np.concatenate(([champ1_key,champ2_key,champ3_key,champ4_key,champ5_key,champ6_key,champ7_key,champ8_key,champ9_key,hotstreak], teamPosition,[winPct]))
    test_match = np.repeat(test.reshape(1, -1), champion_encoded.shape[0], axis=0)
    

    # Define the column names for the DataFrame
    columns = ['champ1', 'champ2', 'champ3', 'champ4', 'champ5', 'champ6', 'champ7', 'champ8', 'champ9', 'hotstreak', 'teamPosition', 'winPct']

    # Create a NumPy array of imputed values for the columns
    imputed_values = test_text

    # Create a DataFrame with the imputed values repeated 162 times
    repeated_df = pd.DataFrame([imputed_values] * 162, columns=columns)
  
    # champ_df supposed to be 162 champs
    prediction = model.predict([test_match, champion_encoded.iloc[:, 1:23]]) 
    # sort the results, highest prediction first
    sorted_index = np.argsort(-prediction,axis=0).reshape(-1).tolist()  #negate to get largest rating first
    ## sorted  percentage of winning
    sorted_ypu   = prediction[sorted_index]
    ##sorted champion list from highest winning to lowest winning
    sorted_champion  = champion_encoded.loc[sorted_index]
    sorted_champion['Winning Rate'] = sorted_ypu
    sorted_champion = pd.concat([sorted_champion, sorted_champion.reset_index(drop=True)], axis=1)
    sorted_champion = sorted_champion.loc[:,~sorted_champion.columns.duplicated()]
    
    
    # Merge the dataframes based on the common column
    merged_df = pd.merge(sorted_champion, LOL_data_copy, left_on='id', right_on='champion')

    # Add the desired columns from LOL_data_copy to sorted_champion
    sorted_champion[['difficulty', 'tags', 'hp', 'hpperlevel', 'mp', 'mpperlevel',
                     'movespeed', 'armor', 'armorperlevel', 'spellblock',
                     'spellblockperlevel', 'attackrange', 'hpregen', 'hpregenperlevel',
                     'mpregen', 'mpregenperlevel', 'attackdamage', 'attackdamageperlevel',
                     'attackspeedperlevel', 'attackspeed', 'Best_Partner', 'Counters',
                     'Countered_by']] = merged_df[['difficulty_y', 'tags',
           'hp_y', 'hpperlevel_y', 'mp_y', 'mpperlevel_y', 'movespeed_y',
           'armor_y', 'armorperlevel_y', 'spellblock_y', 'spellblockperlevel_y',
           'attackrange_y', 'hpregen_y', 'hpregenperlevel_y', 'mpregen_y',
           'mpregenperlevel_y', 'attackdamage_y', 'attackdamageperlevel_y',
           'attackspeedperlevel_y', 'attackspeed_y', 'Best_Partner', 'Counters',
           'Countered_by']]
    ##Combine everything together and sort by winning rate
    concatenated_df = pd.concat([repeated_df, sorted_champion], axis=1)
    columns_to_drop = ['difficulty', 'hp', 'hpperlevel', 'mp', 'mpperlevel',  'movespeed', 'armor', 'armorperlevel', 'spellblock',  'spellblockperlevel', 'attackrange', 'hpregen', 'hpregenperlevel',  'mpregen', 'mpregenperlevel', 'attackdamage', 'attackdamageperlevel',                   'attackspeedperlevel', 'attackspeed']
    concatenated_df = concatenated_df.drop(columns=columns_to_drop)
    new_column_order = ['id', 'Winning Rate', 'tags', 'Best_Partner', 'Counters', 'Countered_by', 'champ1', 'champ2', 'champ3', 'champ4', 'champ5', 'champ6', 'champ7', 'champ8', 'champ9', 'hotstreak', 'teamPosition', 'winPct']
    concatenated_df = concatenated_df[new_column_order]
    concatenated_df = concatenated_df.sort_values(by='Winning Rate', ascending=False)
    merged_df = concatenated_df.merge(df, left_on='id', right_on='Champion', how='left')
    merged_df = merged_df.drop(columns=['Champion'])
    # Define a dictionary to map each champion ID to its role
    champion_roles = {
        'Velkoz': 'Mage',
        'Chogath': 'Tank, Mage',
        'Nilah': 'Assassin, Mage',
        'KSante': 'Assassin',
        'Khazix': 'Assassin',
        'JarvanIV': 'Fighter, Tank',
        'RekSai': 'Fighter, Tank',
        'Kaisa': 'Marksman, Assassin',
        'Belveth': 'Fighter, Tank',
        'Leblanc': 'Assassin, Mage',
        'KogMaw': 'Marksman',
        'DrMundo': 'Fighter, Tank',
        'TwistedFate': 'Mage',
        'MissFortune': 'Marksman',
        'Zeri': 'Mage, Support',
        'Vex': 'Mage',
        'Akshan': 'Assassin, Marksman',
        'Gwen': 'Fighter',
        'LeeSin': 'Fighter',
        'Renata': 'Fighter, Tank',
        'Nunu': 'Tank, Mage',
        'MonkeyKing': 'Fighter',
        'MasterYi': 'Assassin',
        'XinZhao': 'Fighter, Assassin',
        'TahmKench': 'Tank, Support',
        'AurelionSol': 'Mage',
        'Milio' : 'Support',
    }

    # Fill in the 'Roles' column based on the 'id' column
    merged_df.loc[merged_df['id'].isin(champion_roles.keys()), 'Roles'] = merged_df['id'].map(champion_roles)
        # Define the roles for each team position
    roles = {
        'TOP': ['Mage, Support', 'Mage', "Mage, Fighter",'Mage, Assassin', 'Assassin, Mage', 'Marksman, Mage', 'Marksman, Assassin', 'Support'],
        'JUNGLE': ['Mage, Assassin', 'Tank, Support', 'Tank, Mage', 'Mage, Support', 'Mage', 'Marksman', 'Marksman, Support', 'Mage, Marksman', 'Support, Mage', 'Support, Tank', 'Marksman, Mage', 'Support, Assassin', 'Marksman, Assassin', 'Support', 'Tank', 'Support, Fighter'],
        'MIDDLE': ["Support, Mage",'Fighter, Tank', 'Tank, Support', 'Marksman', 'Marksman, Support', 'Mage, Marksman', 'Support, Mage', 'Support, Tank', 'Marksman, Mage', 'Support, Assassin', 'Marksman, Assassin', 'Support', 'Tank', 'Support, Fighter', 'Tank, Mage'],
        'BOTTOM': ['Fighter, Tank', 'Mage, Assassin', 'Assassin', 'Tank, Support', 'Support, Mage', 'Tank, Fighter', 'Support, Tank', 'Fighter, Mage', 'Fighter, Assassin', 'Fighter', 'Support', 'Tank', 'Support, Fighter', 'Assassin, Fighter', 'Mage, Fighter', 'Assassin, Mage', 'Fighter, Support', 'Support, Assassin', 'Tank, Mage', 'Mage, Support', 'Mage'],
        'UTILITY': ['Fighter, Tank', 'Mage, Assassin', 'Assassin', 'Tank, Mage', 'Mage', 'Marksman', 'Tank, Fighter', 'Fighter, Mage', 'Assassin, Fighter', 'Mage, Fighter', 'Assassin, Mage', 'Marksman, Mage', 'Mage, Marksman', 'Fighter, Assassin', 'Fighter', 'Fighter, Marksman', 'Marksman, Assassin']
    }

    # Initialize an empty DataFrame to store the sorted rows
    sorted_df = pd.DataFrame()

    # Iterate through team positions and roles
    for position, role_list in roles.items():
        # Filter rows based on team position and role
        in_role = merged_df[(merged_df['teamPosition'] == position) & (merged_df['Roles'].isin(role_list))]
        not_in_role = merged_df[(merged_df['teamPosition'] == position) & (~merged_df['Roles'].isin(role_list))]

        # Sort the in_role rows by 'Winning Rate'
        sorted_rows = in_role.sort_values(by='Winning Rate', ascending=False)

        # Concatenate the sorted_rows and not_in_role DataFrames
        position_df = pd.concat([not_in_role, sorted_rows], ignore_index=True)

        # Append the position_df to the sorted_df DataFrame
        sorted_df = pd.concat([sorted_df, position_df], ignore_index=True)

    # Reset the index of the sorted DataFrame
    sorted_df.reset_index(drop=True, inplace=True)
    ##construct the final dataframe that takes in consideration of the roles of the champion needed 
    sorted_df.drop(columns=['tags'], inplace=True)
    sorted_df = sorted_df[['id', 'Winning Rate', 'Roles', 'Best_Partner', 'Counters', 'Countered_by', 'champ1', 'champ2', 'champ3', 'champ4', 'champ5', 'champ6', 'champ7', 'champ8', 'champ9', 'hotstreak', 'teamPosition', 'winPct']]
    sorted_df = sorted_df.rename(columns={'id': 'champion_rec'})
    return(sorted_df.head(10))


# collaborative filtering 


recommend_df = pd.read_csv('playerCP.csv')

most_played_champions = []
most_played_CP = []

CPscaler = MinMaxScaler(feature_range=(0.5, 5))

def SVD_recommend(summoner_name, most_played_champions, most_played_CP, recommend_df): 
    player_df = pd.DataFrame()
    for i in range(5):
        mastery_info =  {"playerId":[summoner_name], "championId": [name_to_key[most_played_champions[i]]], "championPoints": [most_played_CP[i]]}
        player_df = pd.concat([player_df, pd.DataFrame(mastery_info)], axis=0)
    player_df.championPoints = CPscaler.fit_transform(player_df.championPoints.values.reshape(-1, 1))
    recommend_df = pd.concat([recommend_df, player_df], axis=0)

    mean_CP = recommend_df.groupby('championId', as_index=False)['championPoints'].mean()
    normalized_CP = pd.merge(recommend_df, mean_CP, on='championId')
    normalized_CP['championPoints'] = normalized_CP['championPoints_x'] - normalized_CP['championPoints_y']

    points_dict = {'playerId': list(normalized_CP.playerId), 
                   'championId': list(normalized_CP.championId), 
                   'championPoints': list(normalized_CP.championPoints)}

    reader = Reader(rating_scale=(0.5, 5))

    data = Dataset.load_from_df(normalized_CP[['playerId', 'championId', 'championPoints']], reader)

    trainset = data.build_full_trainset()
    svd = SVD() 
    svd.fit(trainset) 
    
    recommended = {}
    for key, champ in key_to_name.items():
        #for key in df.championId.unique():
        if (champ not in most_played_champions):  
            pred = svd.predict(summoner_name, key, verbose=False)
            recommended.update({key_to_name[pred.iid]: pred.est})
    top_five = sorted(recommended.items(), key=lambda x:x[1], reverse=True)[0:5]
    recommended_champs = [champ[0] for champ in top_five]
    return recommended_champs



# In[10]:

# streamlit display
st.title('LoL Champion Recommender Prototype')
st.image(image, caption='Death is like the wind - always by my side', use_column_width=True)

# App introduction
st.markdown("""
Explore our interactive dashboard, featuring deep learning and SVG algorithms, to experience content-based and collaborative recommendation systems for the perfect League of Legends champion pairing. By analyzing match-ups, player traits, and champion attributes, our model identifies champions with the highest win probability, tailored to your interests. Designed for scenarios with nine pre-selected champions, our app guides you in choosing the final, game-changing pick. Here is the link to our GitHub and article: 
https://github.com/spearsheep/LoL-DeepLearning-Champion-Recommender 
https://www.bruinsportsanalytics.com/post/lol-recommender
""")

summoner_name = st.text_input("Enter your Name: ", key="name")
st.header('Enter the champions that your teammates have selected: ')

champion_selections = {}
for i in range(1, 5):
    st.subheader(f"Please select champion {i}")
    left_column, right_column = st.columns(2)
    with left_column:
        champion_selections[f'champion_{i}'] = st.selectbox('Select a champion:', np.unique(champion_encoded['id']), key=f'option_{i}')

st.header('Enter the champions that your opponents have selected: ')
for i in range(5, 10):
    st.subheader(f"Please select champion {i}")
    left_column, right_column = st.columns(2)
    with left_column:
        champion_selections[f'champion_{i}'] = st.selectbox('Select a champion:', np.unique(champion_encoded['id']), key=f'option_{i}')

team_positions = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']

selected_team_position = st.select_slider("Select team position:", options=team_positions, key="slider2")

hotstreak_options = ['Yes', 'No']
selected_hotstreak = st.selectbox("Is the player on a hot streak?", options=hotstreak_options, key="slider3")

win_pct = st.slider("Win percentage (%)", min_value=0.0, max_value=100.0, step=0.1, key="slider4")
win_pct = win_pct/100

if st.button('Recommend top 10 Champions for your game', key="button1"):
    prediction = predict(*champion_selections.values(), selected_hotstreak, selected_team_position, win_pct)
    st.write(prediction)

st.markdown("""Leveraging deep learning, our model recommends champions based on interactions between opponent and teammate champions. To provide personalized options for you to explore and enjoy, we utilize the SVG algorithm, suggesting champions tailored to your preferences.""")

st.header('Enter your top five most played champions and mastery points')

most_played_champions = []
most_played_CP = []
for i in range(5):
    most_played_champions.append(st.selectbox('Select a champion:', np.unique(champion_encoded['id']), key=f'MP_champ_{i}'))
    most_played_CP.append(st.number_input('Enter the mastery points for this champion: ', value=0, step=1000, format="%d", key=f'CP_{i}'))

if st.button('Recommend top 5 Champions based on your preferences', key="button2"):
    predictionCP = SVD_recommend(summoner_name, most_played_champions, most_played_CP, recommend_df)
    for i in range(1,6):
        st.write(f'{i}: ' + predictionCP[i-1])
    most_played_champions = []
    most_played_CP = []

                                        

#predict("Darius","Darius","Darius","Darius","Darius","Darius","Darius","Darius","Darius","yes","TOP",0.53)


# In[ ]:





# In[ ]:





# In[ ]:




