import { useState, useEffect } from 'react';
import {
  AlertCircle,
  Loader,
} from 'lucide-react';
import api from './services/api';
import InventoryPage from './components/InventoryPage';
import TransferSuggestionsPage from './components/TransferSuggestionsPage';
import LandingPage from './components/LandingPage';
import UploadPage from './components/UploadPage';
import Navbar from './components/Navbar';


// Predictions Page Component
const PredictionsPage = () => {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchPredictions();
  }, []);

  const fetchPredictions = async () => {
    try {
      setLoading(true);
      const response = await api.get('/api/predict/');
      setPredictions(response.predictions);
    } catch (err) {
      setError('Failed to fetch predictions');
    } finally {
      setLoading(false);
    }
  };

  const getRecommendationColor = (recommendation) => {
    if (recommendation.includes('Restock')) return 'bg-red-100 text-red-800';
    if (recommendation.includes('Monitor')) return 'bg-yellow-100 text-yellow-800';
    if (recommendation.includes('Overstock')) return 'bg-purple-100 text-purple-800';
    return 'bg-green-100 text-green-800';
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.9) return 'text-green-600';
    if (confidence >= 0.8) return 'text-yellow-600';
    return 'text-red-600';
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-96">
        <Loader className="h-8 w-8 animate-spin text-blue-600" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex justify-center items-center h-96">
        <div className="text-center">
          <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <p className="text-red-600">{error}</p>
          <button
            onClick={fetchPredictions}
            className="mt-4 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">Demand Predictions</h1>
        <p className="text-gray-600">AI-powered demand forecasting and inventory recommendations</p>
      </div>

      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
        {predictions.map((prediction, index) => (
          <div key={index} className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">{prediction.product}</h3>

            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Predicted Demand:</span>
                <span className="font-semibold text-blue-600">{prediction.predictedDemand}</span>
              </div>

              <div className="flex justify-between">
                <span className="text-gray-600">Current Stock:</span>
                <span className="font-semibold">{prediction.currentStock}</span>
              </div>

              <div className="flex justify-between">
                <span className="text-gray-600">Confidence:</span>
                <span className={`font-semibold ${getConfidenceColor(prediction.confidence)}`}>
                  {(prediction.confidence * 100).toFixed(0)}%
                </span>
              </div>

              <div className="pt-2">
                <span className={`inline-flex px-3 py-1 text-xs font-semibold rounded-full ${getRecommendationColor(prediction.recommendation)}`}>
                  {prediction.recommendation}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};



// Main App Component
const App = () => {
  const [currentPage, setCurrentPage] = useState('home');
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const renderCurrentPage = () => {
    switch (currentPage) {
      case 'home':
        return <LandingPage setCurrentPage={setCurrentPage} />;
      case 'inventory':
        return <InventoryPage />;
      case 'upload':
        return <UploadPage />;
      case 'predictions':
        return <PredictionsPage />;
      case 'transfers':
        return <TransferSuggestionsPage />;
      default:
        return <LandingPage setCurrentPage={setCurrentPage} />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar
        currentPage={currentPage}
        setCurrentPage={setCurrentPage}
        isMobileMenuOpen={isMobileMenuOpen}
        setIsMobileMenuOpen={setIsMobileMenuOpen}
      />
      <main className="pb-8">
        {renderCurrentPage()}
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-white py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <p className="text-gray-400">
              © 2025 SmartStockX - Walmart Hackathon Project
            </p>
            <p className="text-gray-400 mt-2">
              Optimizing retail inventory with AI and machine learning
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;