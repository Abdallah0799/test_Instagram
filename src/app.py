from flask import Flask, render_template, request
from numpy import vectorize
import pandas as pd
import requests
import json
from joblib import load
from scipy.spatial import distance
import numpy as np
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances

app = Flask(__name__)

base_url = 'https://www.instagram.com/'

data = pd.read_csv("final_database.csv", index_col=0)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    username = request.form['username']
    n = request.form['n']
    sessionid = request.form['sessionid']
    url = base_url+username+'/?__a=1'
    
    try:
        n =int(n)
    except:
        return render_template('index.html', alert="Veuillez saisir un numero valide")
    
    try:
        r = requests.get(
                    "https://www.instagram.com/{}/?__a=1".format(str(username)),
                    headers={
                        "Accept-Encoding": "gzip, deflate",
                        "Accept": "*/*",
                        "Connection": "keep-alive",
                        "Cookie": "sessionid=" + sessionid,
                    },
                )
        doc = r.json()
    except:
        return render_template('index.html', alert = 'Ce nom d utilisateur nexiste pas ou alors Instagram limite votre activité')
    
    
    #f = open('test2.json')
    #doc = json.load(f)
    
    titre_et_bio, description, followers, url_profile_pic = getInfosFromJson(doc)
    d_tb, d_p, d_f, key_words_bio_et_titre, key_words_posts = getDistances(titre_et_bio, description, followers)
    names, scores, scores_bio, scores_posts, followers_others, bios, key_words_bt, key_words_p = get_Names_And_Scores(n,d_tb, d_p, d_f)
    
    objects = []
    objects2 = []
   
    for name, score, s_b, s_p, fol, b, k_bt, k_p in zip(names, scores, scores_bio, scores_posts, followers_others, bios, key_words_bt, key_words_p):
        objects2.append([name, round(score,3), k_bt, k_p, round(s_b, 3), round(s_p,3), b, fol])
        
    
    objects = [[username, titre_et_bio, followers, key_words_bio_et_titre, key_words_posts]]
    
    headers = ['Username', 'Biography', 'Number of followers', 'Key words in Bio', 'Key words in posts']
    headers2 = ["Usernames","Similarity score", 'Key words in Bio', 'Key words in posts', "Similarity score on biography (intermediate score with a weight of 50%)", "Similarity score on posts (intermediate score with a weight of 50%)", "Biography", "Number of followers"]
    
    
    return render_template('result.html', username=username, n=n, usernames = names, headers=headers, headers2 =headers2, objects = objects, objects2 = objects2)


def getInfosFromJson(doc):
    
    description = []
    
    try:
        titre = doc['graphql']['user']['category_name']
    except:
        titre =""
        
    try:
        bio = doc['graphql']['user']['biography']
    except:
        bio=""
    
    try:
        posts_ = doc['graphql']['user']['edge_owner_to_timeline_media']['edges']
        for post in posts_:
            post_description = post["node"]["edge_media_to_caption"]["edges"][0]["node"]["text"]
            description.append(post_description)
    except:
        pass
    
    try:
        followers = doc['graphql']['user']['edge_followed_by']['count']
    except:
        followers = ""
    
    try:
        url_profile_pic = doc['graphql']['user']['profile_pic_url']
    except:
        url_profile_pic = None
    
    description = ','.join(description)
    titre_et_bio = bio + ' / ' + titre
    
    return titre_et_bio, description, followers, url_profile_pic

def getDistances(titre_et_bio, posts_description, followers):
    
    vectorizer_bio_et_titre = load("vectorize_bio_et_titre.joblib")
    vectorizer_posts = load("vectorize_posts.joblib")
    tfid_bio_et_titre = load("tfid_bio_et_titre.joblib")
    tfid_posts = load("tfid_posts.joblib")
    
    ## For bio and titre
    X_bt = data['titre_et_biographie'].values
    
    input_vectorizer_bt = vectorizer_bio_et_titre.transform(X_bt)
    M = tfid_bio_et_titre.transform(input_vectorizer_bt)
    vector_to_compare = vectorizer_bio_et_titre.transform([titre_et_bio])
    inverse_transformation = vectorizer_bio_et_titre.inverse_transform(vector_to_compare)
    if len(inverse_transformation) > 0:
        key_words_bio_et_titre = inverse_transformation[0]
        key_words_bio_et_titre = list(key_words_bio_et_titre)
        key_words_bio_et_titre = ','.join(sorted(key_words_bio_et_titre))
    else:
        key_words_bio_et_titre = "No relevant keywords found"
    
    vector_to_compare__tfid = tfid_bio_et_titre.transform(vector_to_compare)
    
    null_vector = np.zeros(vector_to_compare__tfid.toarray().shape)
    null_distance = distance.cityblock(vector_to_compare__tfid.toarray(), null_vector)
    
    distances_tb = manhattan_distances(M.toarray(), vector_to_compare__tfid.reshape(1, M.toarray().shape[1]))
    distances_tb[distances_tb==null_distance]=np.max(distances_tb)
    distances_tb = distances_tb.reshape(1,M.toarray().shape[0])[0]
    
    normalized_distances_tb = (distances_tb - np.min(distances_tb))/np.max(distances_tb)
    
    ## For posts
    
    X_p = data['description_of_posts'].values
    
    input_vectorizer_p = vectorizer_posts.transform(X_p)
    M = tfid_posts.transform(input_vectorizer_p)
    vector_to_compare = vectorizer_posts.transform([posts_description])
    inverse_transformation = vectorizer_posts.inverse_transform(vector_to_compare)
    if len(inverse_transformation)>0:
        key_words_post = inverse_transformation[0]
        key_words_post = list(key_words_post)
        key_words_post = ','.join(sorted(key_words_post))
    else:
        key_words_post = "No relevant key words found"
    
    vector_to_compare__tfid = tfid_posts.transform(vector_to_compare)
    
    null_vector = np.zeros(vector_to_compare__tfid.toarray().shape)
    null_distance = distance.cityblock(vector_to_compare__tfid.toarray(), null_vector)
    
    distances_p = manhattan_distances(M.toarray(), vector_to_compare__tfid.reshape(1, M.toarray().shape[1]))
    distances_p[distances_p==null_distance]=np.max(distances_p)
    distances_p = distances_p.reshape(1,M.toarray().shape[0])[0]
    
    normalized_distances_p = (distances_p - np.min(distances_p))/np.max(distances_p)
    
    ## For followers
    
    X_f = data['followers'].values
    
    followers = np.array([followers])
    n = X_f.shape[0]
    distances_f = euclidean_distances(X_f.reshape(n, 1), followers.reshape(1,1))
    distances_f = distances_f.reshape(1,n)[0]
    
    normalized_distances_f = (distances_f - np.min(distances_f))/np.max(distances_f)
    
    return normalized_distances_tb, normalized_distances_p, normalized_distances_f, key_words_bio_et_titre, key_words_post

def get_Names_And_Scores(n, d_tb, d_p, d_f):
    
    scores_bio = (1 - d_tb)
    scores_posts = (1 - d_p)
    
    scores = scores_bio*0.5 + scores_posts*0.5
    
    idx = np.flipud(np.argsort(scores)[-n:])
    scores = np.flipud(np.sort(scores)[-n:])
    
    names = []
    list_scores_bio = []
    list_scores_posts = []
    followers = []
    bios = []
    key_words_bt = []
    key_words_p = []
    for i in idx:
        name = data.iloc[i][0]
        names.append(name)
        list_scores_bio.append(scores_bio[i])
        list_scores_posts.append(scores_posts[i])
        followers.append(data.iloc[i][2])
        bios.append(data.iloc[i][8])
        
        bt_el = data.iloc[i][10]
        if type(bt_el) == str:
            key_words_bt.append(bt_el)
        else:
            key_words_bt.append('No relevant keywords found')
        
        p_el = data.iloc[i][11]
        if type(p_el) == str:
            key_words_p.append(p_el)
        else:
            key_words_p.append('No relevant keywords found')
    
    return names, scores, list_scores_bio, list_scores_posts, followers, bios, key_words_bt, key_words_p
    
    
    
    
if __name__ == '__main__':
      app.run(host='localhost', port=5000, debug=True)
