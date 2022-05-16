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

data = pd.read_csv("final_database.csv")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    username = request.form['username']
    n = request.form['n']
    url = base_url+username+'/?__a=1'
    
    try:
        n =int(n)
    except:
        return render_template('index.html', alert="Veuillez saisir un numero valide")
    
    try:
        r = requests.get(url)
        doc = r.json()
    except:
        return render_template('index.html', alert = 'Ce nom d utilisateur nexiste pas ou alors Instagram limite votre activité')
    
    
    #f = open('test2.json')
    #doc = json.load(f)
    
    titre_et_bio, description, followers = getInfosFromJson(doc)
    d_tb, d_p, d_f = getDistances(titre_et_bio, description, followers)
    names, scores = get_Names_And_Scores(n,d_tb, d_p, d_f)
    
    objects = []
    for name, score in zip(names, scores):
        objects.append([name, score])
    
    
    return render_template('result.html', headers=["Username", "Similarity score"], objects = objects)


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
        followers = None
    
    description = ','.join(description)
    titre_et_bio = bio + ' / ' + titre
    
    return titre_et_bio, description, followers

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
    
    return normalized_distances_tb, normalized_distances_p, normalized_distances_f

def get_Names_And_Scores(n, d_tb, d_p, d_f):
    
    scores = (1 - d_tb) * 0.5 + (1 - d_p) * 0.5
    
    idx = np.flipud(np.argsort(scores)[-n:])
    scores = np.flipud(np.sort(scores)[-n:])
    
    names = []
    for i in idx:
        name = data.iloc[i][3]
        names.append(name)
    
    return names, scores
    
    
    
    
if __name__ == '__main__':
      app.run(host='localhost', port=5000, debug=True)
