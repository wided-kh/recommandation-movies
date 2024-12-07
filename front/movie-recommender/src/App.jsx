import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [movieName, setMovieName] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [error, setError] = useState('');
  const [eventRecommendations, setEventRecommendations] = useState([]);

  // Fonction pour appeler l'API de recommandation par KNN
  const recommendKNN = async () => {
    try {
      const response = await axios.post('http://localhost:5000/recommend', {
        movie_name: movieName
      });
      setRecommendations(response.data.recommendations);
      setError('');
    } catch (err) {
      setError('Une erreur s\'est produite lors de la récupération des recommandations.');
    }
  };

  // Fonction pour appeler l'API de recommandation par K-Means
  const recommendKMeans = async () => {
    try {
      const response = await axios.post('http://localhost:5000/kmeans-recommend', {
        movie_name: movieName
      });
      setRecommendations(response.data.recommendations);
      setError('');
    } catch (err) {
      setError('Une erreur s\'est produite lors de la récupération des recommandations.');
    }
  };

  // Fonction pour appeler l'API de recommandations basées sur un événement
  const recommendBasedOnEvent = async () => {
    try {
      const response = await axios.get('http://localhost:5000/event-recommend');
      setEventRecommendations(response.data.recommendations);
    } catch (err) {
      setError('Une erreur s\'est produite lors de la récupération des recommandations d\'événement.');
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Recommandations de Films</h1>

      {/* Input pour le nom du film */}
      <div className="mb-4">
        <input
          type="text"
          placeholder="Entrez le nom d'un film"
          value={movieName}
          onChange={(e) => setMovieName(e.target.value)}
          className="px-4 py-2 border rounded-md"
        />
      </div>

      {/* Boutons pour les recommandations */}
      <div className="mb-4">
        <button
          onClick={recommendKNN}
          className="bg-blue-500 text-white px-4 py-2 rounded-md mr-2"
        >
          Recommandations KNN
        </button>
        <button
          onClick={recommendKMeans}
          className="bg-green-500 text-white px-4 py-2 rounded-md"
        >
          Recommandations K-Means
        </button>
        <button
          onClick={recommendBasedOnEvent}
          className="bg-purple-500 text-white px-4 py-2 rounded-md ml-2"
        >
          Recommandations Basées sur l'Événement
        </button>
      </div>

      {/* Affichage des recommandations */}
      {error && <div className="text-red-500 mb-4">{error}</div>}

      <div className="mb-4">
        <h2 className="text-xl font-semibold">Recommandations de Films :</h2>
        <ul>
          {recommendations.length > 0 ? (
            recommendations.map((movie, index) => (
              <li key={index} className="mb-2">
                <strong>{movie.title}</strong> - Genres: {movie.genres.join(', ')}
              </li>
            ))
          ) : (
            <p>Aucune recommandation disponible</p>
          )}
        </ul>
      </div>

      {/* Affichage des recommandations basées sur l'événement */}
      <div className="mb-4">
        <h2 className="text-xl font-semibold">Recommandations Basées sur l'Événement :</h2>
        <ul>
          {eventRecommendations.length > 0 ? (
            eventRecommendations.map((movie, index) => (
              <li key={index} className="mb-2">
                <strong>{movie.title}</strong> - Genres: {movie.genres.join(', ')}
              </li>
            ))
          ) : (
            <p>Aucune recommandation d'événement disponible</p>
          )}
        </ul>
      </div>
    </div>
  );
}

export default App;
