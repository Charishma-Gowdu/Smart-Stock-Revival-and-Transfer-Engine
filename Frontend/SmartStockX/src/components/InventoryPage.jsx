import { useEffect, useState } from 'react';
import { Loader, AlertCircle, RefreshCw, Search, ChevronDown, Package, TrendingUp, Calendar, DollarSign } from 'lucide-react';
import api from '../services/api';


// Stats Card Component
const StatsCard = ({ title, value, icon: Icon, color = "blue" }) => {
    const colorClasses = {
        blue: "bg-blue-50 text-blue-600",
        green: "bg-green-50 text-green-600",
        yellow: "bg-yellow-50 text-yellow-600",
        red: "bg-red-50 text-red-600"
    };

    return (
        <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
                <div>
                    <p className="text-sm font-medium text-gray-600">{title}</p>
                    <p className="text-2xl font-bold text-gray-900">{value}</p>
                </div>
                <div className={`p-3 rounded-full ${colorClasses[color]}`}>
                    <Icon className="h-6 w-6" />
                </div>
            </div>
        </div>
    );
};

// Search and Filter Component
const SearchAndFilter = ({ 
    searchTerm, 
    setSearchTerm, 
    filterLocation, 
    setFilterLocation, 
    locations, 
    onRefresh 
}) => {
    return (
        <div className="flex flex-col sm:flex-row gap-4 mb-6">
            <div className="flex-1 relative">
                <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                <input
                    type="text"
                    placeholder="Search product or store ID..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
            </div>
            <select
                value={filterLocation}
                onChange={(e) => setFilterLocation(e.target.value)}
                className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
                {locations.map(location => (
                    <option key={location} value={location}>{location}</option>
                ))}
            </select>
            <button
                onClick={onRefresh}
                className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
            >
                <RefreshCw className="h-4 w-4" />
                <span>Refresh</span>
            </button>
        </div>
    );
};

// Inventory Table Component
const InventoryTable = ({ inventory, showMore, onShowMore, hasMore }) => {
    const classifyDemand = (value) => {
        if (value > 250) return 'High';
        if (value > 100) return 'Medium';
        return 'Low';
    };
    console.log(inventory);
    

    const getDemandColor = (demand) => {
        switch (demand) {
            case 'High': return 'bg-red-100 text-red-800';
            case 'Medium': return 'bg-yellow-100 text-yellow-800';
            case 'Low': return 'bg-green-100 text-green-800';
            default: return 'bg-gray-100 text-gray-800';
        }
    };

    const getExpiryStatus = (daysToExpiry) => {
        if (daysToExpiry <= 7) return { text: 'Critical', color: 'bg-red-100 text-red-800' };
        if (daysToExpiry <= 30) return { text: 'Warning', color: 'bg-yellow-100 text-yellow-800' };
        return { text: 'Good', color: 'bg-green-100 text-green-800' };
    };

    const formatCurrency = (value) => {
        return new Intl.NumberFormat('en-IN', {
            style: 'currency',
            currency: 'INR',
            minimumFractionDigits: 0,
            maximumFractionDigits: 2
        }).format(value);
    };

    const formatDate = (dateString) => {
        return new Date(dateString).toLocaleDateString('en-IN', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    };

    return (
        <div className="bg-white rounded-lg shadow overflow-hidden">
            <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                        <tr>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Product Details</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Store</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Stock</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Expiry</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Demand</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Pricing</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Sales Info</th>
                        </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                        {inventory.slice(0, showMore ? inventory.length : 20).map(item => {
                            const demand = classifyDemand(item.predicted_demand);
                            const expiryStatus = getExpiryStatus(item.days_to_expiry);
                            const discountPercentage = (item.discount * 100).toFixed(0);
                            
                            return (
                                <tr key={item.id} className="hover:bg-gray-50">
                                    <td className="px-6 py-4 whitespace-nowrap">
                                        <div className="flex items-center">
                                            <div className="flex-shrink-0 h-10 w-10">
                                                <div className="h-10 w-10 rounded-full bg-gray-200 flex items-center justify-center">
                                                    <Package className="h-5 w-5 text-gray-500" />
                                                </div>
                                            </div>
                                            <div className="ml-4">
                                                <div className="text-sm font-medium text-gray-900">{item.product_name}</div>
                                                <div className="text-sm text-gray-500">ID: {item.id}</div>
                                            </div>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{item.store_id}</td>
                                    <td className="px-6 py-4 whitespace-nowrap">
                                        <div className="text-sm text-gray-900">{item.stock} units</div>
                                        <div className="text-sm text-gray-500">Shelf life: {item.shelf_life_days} days</div>
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap">
                                        <div className="text-sm text-gray-900">{formatDate(item.expiry_date)}</div>
                                        <div className="text-sm text-gray-500">{item.days_to_expiry} days left</div>
                                        <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${expiryStatus.color}`}>
                                            {expiryStatus.text}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap">
                                        <div className="text-sm text-gray-900">{Math.round(item.predicted_demand)}</div>
                                        <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getDemandColor(demand)}`}>
                                            {demand}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap">
                                        <div className="text-sm text-gray-900">{formatCurrency(item.final_price)}</div>
                                        <div className="text-sm text-gray-500">
                                            MRP: {formatCurrency(item.MRP)}
                                            {item.discount > 0 && (
                                                <span className="ml-1 text-green-600">({discountPercentage}% off)</span>
                                            )}
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap">
                                        <div className="text-sm text-gray-900">{item.avg_daily_sales}/day</div>
                                        <div className="text-sm text-gray-500">Expected: {item.expected_sales}</div>
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
            
            {!showMore && hasMore && (
                <div className="px-6 py-4 bg-gray-50 border-t border-gray-200">
                    <button
                        onClick={onShowMore}
                        className="w-full flex items-center justify-center space-x-2 text-blue-600 hover:text-blue-800 font-medium"
                    >
                        <span>Show More Items</span>
                        <ChevronDown className="h-4 w-4" />
                    </button>
                </div>
            )}
        </div>
    );
};

const InventoryPage = () => {
    const [inventory, setInventory] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [searchTerm, setSearchTerm] = useState('');
    const [filterLocation, setFilterLocation] = useState('All');
    const [showMore, setShowMore] = useState(false);

    useEffect(() => {
        fetchInventory();
    }, []);

    const fetchInventory = async () => {
        try {
            setLoading(true);
            const data = await api.get('/api/inventory/');
            setInventory(data);
        } catch (err) {
            console.error(err);
            setError('Failed to fetch inventory data');
            setInventory([]);
        } finally {
            setLoading(false);
        }
    };

    const filteredInventory = inventory.filter(item => {
        const matchesSearch =
            item.product_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
            item.store_id.toLowerCase().includes(searchTerm.toLowerCase());
        const matchesLocation =
            filterLocation === 'All' || item.store_id === filterLocation;
        return matchesSearch && matchesLocation;
    });

    const locations = ['All', ...new Set(inventory.map(item => item.store_id))];

    const stats = {
        totalItems: inventory.length,
        totalStock: inventory.reduce((sum, item) => sum + item.stock, 0),
        expiringItems: inventory.filter(item => item.days_to_expiry <= 7).length,
        highDemandItems: inventory.filter(item => item.predicted_demand > 250).length
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
                        onClick={fetchInventory}
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
                <h1 className="text-3xl font-bold text-gray-900 mb-2">Inventory Management</h1>
                <p className="text-gray-600">Monitor stock levels, expiry dates, and demand predictions</p>
            </div>

            {/* Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <StatsCard 
                    title="Total Items" 
                    value={stats.totalItems.toLocaleString()} 
                    icon={Package} 
                    color="blue" 
                />
                <StatsCard 
                    title="Total Stock" 
                    value={stats.totalStock.toLocaleString()} 
                    icon={TrendingUp} 
                    color="green" 
                />
                <StatsCard 
                    title="Expiring Soon" 
                    value={stats.expiringItems.toLocaleString()} 
                    icon={Calendar} 
                    color="red" 
                />
                <StatsCard 
                    title="High Demand" 
                    value={stats.highDemandItems.toLocaleString()} 
                    icon={DollarSign} 
                    color="yellow" 
                />
            </div>

            {/* Search and Filter */}
            <SearchAndFilter
                searchTerm={searchTerm}
                setSearchTerm={setSearchTerm}
                filterLocation={filterLocation}
                setFilterLocation={setFilterLocation}
                locations={locations}
                onRefresh={fetchInventory}
            />

            {/* Inventory Table */}
            <InventoryTable
                inventory={filteredInventory}
                showMore={showMore}
                onShowMore={() => setShowMore(true)}
                hasMore={filteredInventory.length > 20}
            />

            {/* Results Count */}
            <div className="mt-4 text-sm text-gray-600">
                Showing {showMore ? filteredInventory.length : Math.min(20, filteredInventory.length)} of {filteredInventory.length} items
            </div>
        </div>
    );
};

export default InventoryPage;