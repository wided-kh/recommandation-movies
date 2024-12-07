from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from datetime import datetime

# Initialiser Flask
app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])  # Origine du frontend

# Charger les données
movies = pd.read_csv('movies.csv')  # movieId, title, genres
ratings = pd.read_csv('ratings.csv')  # userId, movieId, rating

# Nettoyage des données
def clean_data():
    # Suppression des films sans genre
    movies.dropna(subset=['genres'], inplace=True)
    
    # Conversion de genres en une liste de genres
    movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))

    # Supprimer les doublons dans les évaluations
    ratings.drop_duplicates(subset=['userId', 'movieId'], inplace=True)

clean_data()

# Fusionner les données
movie_data = pd.merge(ratings, movies, on='movieId')

# Créer une matrice utilisateur-film
user_movie_matrix = movie_data.pivot_table(index='userId', columns='title', values='rating')
user_movie_matrix.fillna(0, inplace=True)

# Calculer la similarité cosinus
cosine_sim = cosine_similarity(user_movie_matrix.T)
cosine_sim_df = pd.DataFrame(cosine_sim, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

# Modèle KNN
knn_model = NearestNeighbors(metric='cosine', n_neighbors=6)
knn_model.fit(user_movie_matrix.T)

# Fonction de recommandation KNN
def recommend_knn(movie_name, n=5):
    if movie_name not in user_movie_matrix.columns:
        return ["Film non trouvé"]

    # Trouver l'index du film
    movie_idx = user_movie_matrix.columns.get_loc(movie_name)

    # Trouver les voisins avec KNN
    distances, indices = knn_model.kneighbors([user_movie_matrix.T.iloc[movie_idx]], n_neighbors=n+1)

    # Récupérer les titres des films recommandés
    recommended_movies = [user_movie_matrix.columns[i] for i in indices[0][1:]]

    # Ajouter les genres correspondants
    movie_details = []
    for title in recommended_movies:
        movie_info = movies[movies['title'] == title].iloc[0]
        movie_details.append({
            "title": movie_info['title'],
            "genres": movie_info['genres']
        })

    return movie_details

# K-Means clustering pour la recommandation basée sur les ratings
def recommend_kmeans(movie_name, n_clusters=5, n_recommendations=5):
    # Normaliser les ratings des films
    scaler = StandardScaler()
    user_movie_matrix_scaled = scaler.fit_transform(user_movie_matrix.T)

    # Appliquer KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    movie_clusters = kmeans.fit_predict(user_movie_matrix_scaled)

    # Trouver le cluster du film donné
    movie_idx = user_movie_matrix.columns.get_loc(movie_name)
    movie_cluster = movie_clusters[movie_idx]

    # Trouver les indices des films dans le même cluster
    cluster_indices = [i for i, cluster in enumerate(movie_clusters) if cluster == movie_cluster]

    # Obtenir les films recommandés dans le même cluster
    recommended_movies = [user_movie_matrix.columns[i] for i in cluster_indices if user_movie_matrix.columns[i] != movie_name]

    # Limiter le nombre de recommandations
    recommended_movies = recommended_movies[:n_recommendations]

    # Ajouter les genres correspondants
    movie_details = []
    for title in recommended_movies:
        movie_info = movies[movies['title'] == title].iloc[0]
        movie_details.append({
            "title": movie_info['title'],
            "genres": movie_info['genres']
        })

    return movie_details

# Modèle de classification des genres (Random Forest)
def genre_classification(movie_name):
    # Encoding des genres
    le = LabelEncoder()
    movies['encoded_genres'] = le.fit_transform(movies['genres'].apply(lambda x: '|'.join(x)))

    # Séparer les caractéristiques et labels
    X = movie_data[['userId', 'movieId']].values
    y = movie_data['encoded_genres']

    # Séparer en train et test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Modèle Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    # Évaluer le modèle
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

# Liste des événements et des genres associés
event_genres = {
    "World Cup": ["Sports", "Action", "Sports Documentary"],
    "Halloween": ["Horror", "Thriller", "Fantasy"],
    "Christmas": ["Comedy", "Family", "Adventure"],
    "Valentine's Day": ["Romance", "Romantic Comedy"]
}


def get_event_for_today():
    current_date = datetime.now()

    # Exemple de détection d'événements basés sur la date
    if current_date.month == 12 and current_date.day == 25:
        return "Christmas"
    elif current_date.month == 10 and current_date.day == 31:
        return "Halloween"
    elif current_date.month == 2 and current_date.day == 14:
        return "Valentine's Day"
    elif current_date.month == 6 and current_date.day == 1:
        return "World Cup"  # À ajuster en fonction des années
    else:
        return None

# Exemple de fonction de recommandations qui renvoie un maximum de 10 films
def get_recommendations(event=None):
    # Exemple de liste de films
    all_movies = [
        {"title": "Movie 1", "genres": ["Action", "Adventure"]},
        {"title": "Movie 2", "genres": ["Drama"]},
        {"title": "Movie 3", "genres": ["Comedy"]},
        {"title": "Movie 4", "genres": ["Action", "Thriller"]},
        {"title": "Movie 5", "genres": ["Drama", "Romance"]},
        {"title": "Movie 6", "genres": ["Sci-Fi"]},
        {"title": "Movie 7", "genres": ["Fantasy"]},
        {"title": "Movie 8", "genres": ["Horror"]},
        {"title": "Movie 9", "genres": ["Action", "Drama"]},
        {"title": "Movie 10", "genres": ["Comedy", "Romance"]},
        {"title": "Movie 11", "genres": ["Thriller"]},
        {"title": "Movie 12", "genres": ["Drama", "Sci-Fi"]},
        {"title": "Movie 13", "genres": ["Adventure", "Fantasy"]},
        {"title": "Movie 14", "genres": ["Action", "Sci-Fi"]},
        {"title": "Movie 15", "genres": ["Romance"]},
    ]
    
    # Si un événement est spécifié, filtrer les films selon l'événement
    if event:
        all_movies = [movie for movie in all_movies if event.lower() in movie["genres"]]
    
    # Retourner un maximum de 10 films
    return all_movies[:10]

# Exemple d'utilisation
event_today = get_event_for_today()
recommendations = get_recommendations(event_today)

# Afficher les recommandations
for movie in recommendations:
    print(f"{movie['title']} - Genres: {', '.join(movie['genres'])}")

# Fonction de recommandation basée sur l'événement
def recommend_based_on_event():
    event = get_event_for_today()

    if event is None:
        return {"message": "Aucun événement spécifique aujourd'hui."}
    
    # Récupérer les genres associés à l'événement
    genres_to_recommend = event_genres.get(event, [])

    # Filtrer les films basés sur les genres de l'événement
    recommended_movies = movies[movies['genres'].apply(lambda x: any(genre in genres_to_recommend for genre in x))]
    
    # Retourner les titres des films recommandés
    recommended_movies_list = recommended_movies[['title', 'genres']].to_dict(orient='records')
    
    return recommended_movies_list

# Endpoint pour obtenir des recommandations par KNN
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        movie_name = data.get('movie_name', '').strip()

        if not movie_name:
            return jsonify({"error": "Movie name is required"}), 400

        # Obtenir les recommandations avec titre et genres
        recommendations = recommend_knn(movie_name)

        return jsonify({"recommendations": recommendations})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred"}), 500

# Endpoint pour obtenir des recommandations par K-Means
@app.route('/kmeans-recommend', methods=['POST'])
def kmeans_recommend():
    try:
        data = request.json
        movie_name = data.get('movie_name', '').strip()

        if not movie_name:
            return jsonify({"error": "Movie name is required"}), 400

        # Obtenir les recommandations par K-Means
        recommendations = recommend_kmeans(movie_name)

        return jsonify({"recommendations": recommendations})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred"}), 500

# Endpoint pour obtenir la classification des genres
@app.route('/genre-classification', methods=['POST'])
def classify_genre():
    try:
        data = request.json
        movie_name = data.get('movie_name', '').strip()

        if not movie_name:
            return jsonify({"error": "Movie name is required"}), 400

        # Obtenir la classification des genres
        accuracy = genre_classification(movie_name)

        return jsonify({"accuracy": accuracy})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred"}), 500

# Endpoint pour obtenir des recommandations basées sur un événement spécifique
@app.route('/event-recommend', methods=['GET'])
def event_recommend():
    try:
        # Obtenir les recommandations basées sur l'événement
        recommendations = recommend_based_on_event()

        return jsonify({"recommendations": recommendations})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred"}), 500

# Lancer l'application Flask
if __name__ == '__main__':
    app.run(debug=True)
