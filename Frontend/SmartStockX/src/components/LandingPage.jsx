import { ArrowRightLeft, Package, TrendingUp } from "lucide-react";

// Landing Page Component
const LandingPage = ({ setCurrentPage }) => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Hero Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="text-center">
          <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
            Smart<span className="text-blue-600">Stock</span>X
          </h1>
          <p className="text-xl md:text-2xl text-gray-600 mb-8 max-w-3xl mx-auto">
            Optimize inventory transfer between retail stores using ML-powered demand prediction and intelligent logistics
          </p>
          <div className="flex flex-col sm:flex-row justify-center gap-4 mb-12">
            <button
              onClick={() => setCurrentPage('inventory')}
              className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-lg font-semibold transition-colors cursor-pointer"
            >
              View Inventory
            </button>
            <button
              onClick={() => setCurrentPage('predictions')}
              className="bg-white hover:bg-gray-50 text-blue-600 border-2 border-blue-600 px-8 py-3 rounded-lg font-semibold transition-colors cursor-pointer"
            >
              See Predictions
            </button>
          </div>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-3 gap-8 mt-20">
          <div className="bg-white rounded-lg shadow-lg p-6 text-center">
            <TrendingUp className="h-12 w-12 text-blue-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold mb-2">ML Predictions</h3>
            <p className="text-gray-600">Advanced machine learning algorithms predict demand patterns and optimize inventory levels</p>
          </div>
          <div className="bg-white rounded-lg shadow-lg p-6 text-center">
            <ArrowRightLeft className="h-12 w-12 text-blue-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold mb-2">Smart Transfers</h3>
            <p className="text-gray-600">Automated transfer recommendations based on real-time demand and expiry analysis</p>
          </div>
          <div className="bg-white rounded-lg shadow-lg p-6 text-center">
            <Package className="h-12 w-12 text-blue-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold mb-2">Inventory Management</h3>
            <p className="text-gray-600">Comprehensive inventory tracking with real-time updates across all store locations</p>
          </div>
        </div>
      </div>
    </div>
  );
};


export default LandingPage;