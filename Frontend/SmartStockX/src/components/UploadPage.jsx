import { useState } from "react";
import { AlertCircle, CheckCircle, Loader, Upload, Package, MapPin, FileText, Database } from "lucide-react";
import api from "../services/api";

const FileFormatGuide = ({ fileType }) => {
    const formats = {
        inventory: {
            title: "Inventory Data CSV Format",
            description: "Upload inventory data with product information, stock levels, and sales data.",
            example: "store_id,product_id,product_name,category,expiry_date,stock,avg_daily_sales,MRP,shelf_life_days",
            sampleData: "S001,P001,Beverage Item 01,Beverage,2025-08-18,62,5,63,41"
        },
        distances: {
            title: "Store Distances CSV Format",
            description: "Upload distance data between different store locations.",
            example: "from_store,to_store,distance_km",
            sampleData: "S001,S002,5.5"
        }
    };

    const format = formats[fileType];

    return (
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-5 rounded-xl mb-6 border border-blue-100 shadow-sm">
            <h3 className="font-semibold mb-3 flex items-center space-x-2 text-blue-900">
                <FileText className="h-5 w-5 text-blue-600" />
                <span>{format.title}</span>
            </h3>
            <p className="text-sm text-blue-700 mb-4 leading-relaxed">{format.description}</p>

            <div className="space-y-3">
                <div className="bg-white p-4 rounded-lg border border-blue-200 shadow-sm">
                    <div className="flex items-center space-x-2 mb-2">
                        <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                        <p className="text-xs font-medium text-gray-600 uppercase tracking-wide">Header Row</p>
                    </div>
                    <code className="text-sm text-gray-800 font-mono block overflow-x-auto whitespace-nowrap bg-gray-50 p-2 rounded border-l-4 border-green-400">
                        {format.example}
                    </code>
                </div>

                <div className="bg-white p-4 rounded-lg border border-blue-200 shadow-sm">
                    <div className="flex items-center space-x-2 mb-2">
                        <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                        <p className="text-xs font-medium text-gray-600 uppercase tracking-wide">Sample Data</p>
                    </div>
                    <code className="text-sm text-gray-800 font-mono block overflow-x-auto whitespace-nowrap bg-gray-50 p-2 rounded border-l-4 border-blue-400">
                        {format.sampleData}
                    </code>
                </div>
            </div>
        </div>
    );
};

const UploadSection = ({
    fileType,
    title,
    icon: Icon,
    file,
    setFile,
    uploading,
    message,
    onUpload,
    isIndividualUpload = true
}) => {
    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile && selectedFile.type === 'text/csv') {
            setFile(selectedFile);
        } else {
            console.error('Please select a valid CSV file');
        }
    };

    return (
        <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center space-x-3 mb-6">
                <Icon className="h-6 w-6 text-blue-600" />
                <h2 className="text-xl font-semibold text-gray-900">{title}</h2>
            </div>

            <FileFormatGuide fileType={fileType} />

            {/* File Input */}
            <div className="mb-6">
                <label htmlFor={`${fileType}-input`} className="block text-sm font-medium text-gray-700 mb-2">
                    Choose CSV File
                </label>
                <input
                    id={`${fileType}-input`}
                    type="file"
                    accept=".csv"
                    onChange={handleFileChange}
                    className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                />
            </div>

            {/* Selected File Info */}
            {file && (
                <div className="mb-6 p-4 bg-blue-50 rounded-lg">
                    <p className="text-sm text-blue-800">
                        <strong>Selected file:</strong> {file.name} ({(file.size / 1024).toFixed(1)} KB)
                    </p>
                </div>
            )}

            {/* Upload Button - Only show if individual upload is enabled */}
            {isIndividualUpload && (
                <button
                    onClick={onUpload}
                    disabled={!file || uploading}
                    className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-6 py-3 rounded-lg font-semibold transition-colors w-full sm:w-auto"
                >
                    {uploading ? (
                        <>
                            <Loader className="h-4 w-4 animate-spin" />
                            <span>Uploading...</span>
                        </>
                    ) : (
                        <>
                            <Upload className="h-4 w-4" />
                            <span>Upload {title}</span>
                        </>
                    )}
                </button>
            )}

            {/* Message Display */}
            {message.content && (
                <div className={`mt-6 p-4 rounded-lg flex items-center space-x-2 ${message.type === 'success' ? 'bg-green-50 text-green-800' : 'bg-red-50 text-red-800'
                    }`}>
                    {message.type === 'success' ? (
                        <CheckCircle className="h-5 w-5" />
                    ) : (
                        <AlertCircle className="h-5 w-5" />
                    )}
                    <span>{message.content}</span>
                </div>
            )}
        </div>
    );
};

// Main Upload Page Component
const UploadPage = () => {
    const [inventoryFile, setInventoryFile] = useState(null);
    const [distanceFile, setDistanceFile] = useState(null);
    const [uploading, setUploading] = useState(false);
    const [inventoryMessage, setInventoryMessage] = useState({ type: '', content: '' });
    const [distanceMessage, setDistanceMessage] = useState({ type: '', content: '' });
    const [mainMessage, setMainMessage] = useState({ type: '', content: '' });


    const handleBothFilesUpload = async () => {
        // Reset all messages
        setInventoryMessage({ type: '', content: '' });
        setDistanceMessage({ type: '', content: '' });
        setMainMessage({ type: '', content: '' });

        // Validate both files are selected
        if (!inventoryFile) {
            setInventoryMessage({ type: 'error', content: 'Please select inventory file' });
            return;
        }

        if (!distanceFile) {
            setDistanceMessage({ type: 'error', content: 'Please select distance file' });
            return;
        }

        try {
            setUploading(true);

            const formData = new FormData();
            formData.append('inventory_file', inventoryFile);
            formData.append('distance_file', distanceFile);
            
            try {
                const data = await api.post('/api/run/', formData, true);

                console.log(data);
                
            } catch (err) {
                console.error(err);
                setError('Failed to upload inventory data');
            }

            setMainMessage({
                type: 'success',
                content: `Upload successful! Inventory Updated Successfully.`
            });

            // Reset files and inputs
            setInventoryFile(null);
            setDistanceFile(null);
            document.getElementById('inventory-input').value = '';
            document.getElementById('distances-input').value = '';
        } catch (error) {
            setMainMessage({
                type: 'error',
                content: 'Upload failed. Please try again.'
            });
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gray-50">

            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <div className="mb-8">
                    <h1 className="text-3xl font-bold text-gray-900 mb-2">Upload Data</h1>
                    <p className="text-gray-600">Upload both inventory and distance data files to process</p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {/* Inventory Upload Section */}
                    <UploadSection
                        fileType="inventory"
                        title="Inventory Data"
                        icon={Package}
                        file={inventoryFile}
                        setFile={setInventoryFile}
                        uploading={uploading}
                        message={inventoryMessage}
                        onUpload={() => { }} // Disabled individual upload
                        isIndividualUpload={false}
                    />

                    {/* Distance Upload Section */}
                    <UploadSection
                        fileType="distances"
                        title="Store Distances"
                        icon={MapPin}
                        file={distanceFile}
                        setFile={setDistanceFile}
                        uploading={uploading}
                        message={distanceMessage}
                        onUpload={() => { }} // Disabled individual upload
                        isIndividualUpload={false}
                    />
                </div>

                {/* Main Upload Button */}
                <div className="mt-8">
                    <div className="bg-white rounded-lg shadow-lg p-6">
                        <div className="text-center">
                            <h3 className="text-lg font-semibold text-gray-900 mb-4">
                                Process Both Files
                            </h3>
                            <p className="text-gray-600 mb-6">
                                Both inventory and distance files must be selected before processing
                            </p>

                            <button
                                onClick={handleBothFilesUpload}
                                disabled={!inventoryFile || !distanceFile || uploading}
                                className="flex items-center justify-center space-x-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white px-8 py-3 rounded-lg font-semibold transition-colors mx-auto"
                            >
                                {uploading ? (
                                    <>
                                        <Loader className="h-5 w-5 animate-spin" />
                                        <span>Processing Files...</span>
                                    </>
                                ) : (
                                    <>
                                        <Upload className="h-5 w-5" />
                                        <span>Upload & Process Both Files</span>
                                    </>
                                )}
                            </button>

                            {/* Main Message Display */}
                            {mainMessage.content && (
                                <div className={`mt-6 p-4 rounded-lg flex items-center justify-center space-x-2 ${mainMessage.type === 'success' ? 'bg-green-50 text-green-800' : 'bg-red-50 text-red-800'
                                    }`}>
                                    {mainMessage.type === 'success' ? (
                                        <CheckCircle className="h-5 w-5" />
                                    ) : (
                                        <AlertCircle className="h-5 w-5" />
                                    )}
                                    <span>{mainMessage.content}</span>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
};

export default UploadPage;