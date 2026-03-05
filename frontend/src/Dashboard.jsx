import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar
} from 'recharts';
import {
    Activity, Zap, ThermometerIcon, TrendingUp, AlertTriangle, CheckCircle, Clock, Cloud, Wind, Droplets, Upload, Shield
} from 'lucide-react';
import './Dashboard.css';

const Dashboard = () => {
    const [inverters, setInverters] = useState(["INV-01"]);
    const [selectedInverter, setSelectedInverter] = useState("INV-01");
    const [predictionMode, setPredictionMode] = useState("internal"); // "internal" or "internal+external"
    const [dataHistory, setDataHistory] = useState([]);
    const [currentStatus, setCurrentStatus] = useState(null);
    const [alerts, setAlerts] = useState([]);
    const [isUploading, setIsUploading] = useState(false);

    const telemetryInputRef = useRef(null);
    const weatherInputRef = useRef(null);

    useEffect(() => {
        const fetchInverters = async () => {
            try {
                const response = await axios.get('/api/inverters');
                setInverters(response.data);
                if (response.data.length > 0) {
                    setSelectedInverter(response.data[0]);
                }
            } catch (error) {
                console.error("Error fetching inverters:", error);
            }
        };
        fetchInverters();
    }, []);

    useEffect(() => {
        const fetchTelemetry = async () => {
            try {
                const response = await axios.get(`/api/telemetry/current?inverter_id=${selectedInverter}`);
                const { telemetry, prediction } = response.data;

                const timeStr = new Date(telemetry.timestamp).toLocaleTimeString();

                const newDataPoint = {
                    time: timeStr,
                    voltage: telemetry.grid_voltage,
                    power: telemetry.dc_power,
                    efficiency: telemetry.inverter_efficiency,
                    temp: telemetry.inverter_temperature,
                    cloud: telemetry.cloud_cover * 100,
                    wind: telemetry.wind_speed,
                    rain: telemetry.rainfall * 10
                };

                setDataHistory(prev => {
                    const newHistory = [...prev, newDataPoint];
                    if (newHistory.length > 20) return newHistory.slice(newHistory.length - 20);
                    return newHistory;
                });

                setCurrentStatus({ telemetry, prediction });
                setPredictionMode(prediction.mode);

                if (prediction.anomaly_detected) {
                    setAlerts(prev => {
                        const newAlert = {
                            id: Date.now(),
                            time: timeStr,
                            level: prediction.risk_level.toLowerCase(),
                            cause: prediction.root_cause || "Anomaly Detected",
                            desc: prediction.maintenance_recommendation,
                            reasons: prediction.reasons
                        };
                        const exists = prev.some(a => a.cause === newAlert.cause && a.time === newAlert.time);
                        if (exists) return prev;
                        const newAlerts = [newAlert, ...prev];
                        return newAlerts.length > 5 ? newAlerts.slice(0, 5) : newAlerts;
                    });
                }
            } catch (error) {
                console.error("Error fetching telemetry:", error);
            }
        };

        fetchTelemetry();
        const intervalId = setInterval(fetchTelemetry, 2500);
        return () => clearInterval(intervalId);
    }, [selectedInverter]);

    const handleInverterChange = (e) => {
        setSelectedInverter(e.target.value);
        setDataHistory([]);
        setAlerts([]);
    };

    const handleFileUpload = async (type, file) => {
        if (!file) return;
        setIsUploading(true);
        const formData = new FormData();
        formData.append('file', file);

        try {
            const endpoint = type === 'telemetry' ? '/api/upload-telemetry' : '/api/upload-weather';
            await axios.post(endpoint, formData);
            alert(`${type === 'telemetry' ? 'Telemetry' : 'Weather'} data synced successfully!`);
        } catch (error) {
            console.error("Upload failed:", error);
            alert("Upload failed. Please check the file format.");
        } finally {
            setIsUploading(false);
        }
    };

    if (!currentStatus) return <div className="dashboard-container loading">Loading System Diagnostics...</div>;

    const { telemetry, prediction } = currentStatus;

    const renderStatusBadge = () => {
        const icons = { Low: CheckCircle, Medium: AlertTriangle, High: Activity, Critical: Activity };
        const Icon = icons[prediction.risk_level] || AlertTriangle;
        const classes = { Low: 'healthy', Medium: 'warning', High: 'critical', Critical: 'critical' };

        return (
            <div className={`status-badge ${classes[prediction.risk_level]} glass-panel`}>
                <Icon size={18} /> {prediction.risk_level === 'Low' ? 'System Optimal' : `${prediction.risk_level} Risk`}
            </div>
        );
    };

    return (
        <div className="dashboard-container">
            <header className="dashboard-header">
                <div>
                    <h1 className="dashboard-title text-gradient">SolarIQ Pulse</h1>
                    <p className="dashboard-subtitle">Next-Gen Predictive Maintenance Interface</p>
                </div>
                <div className="header-actions">
                    <div className="mode-selector glass-panel">
                        <div
                            className={`mode-option ${predictionMode === 'internal' ? 'active' : ''}`}
                            onClick={() => setPredictionMode('internal')}
                        >
                            Telemetry
                        </div>
                        <div
                            className={`mode-option ${predictionMode === 'internal+external' ? 'active' : ''}`}
                            onClick={() => setPredictionMode('internal+external')}
                        >
                            Enhanced
                        </div>
                    </div>

                    <div className="device-selector glass-panel">
                        <select value={selectedInverter} onChange={handleInverterChange}>
                            {inverters.map(id => (
                                <option key={id} value={id}>{id}</option>
                            ))}
                        </select>
                    </div>
                    {renderStatusBadge()}
                </div>
            </header>

            <div className="grid-layout">
                {/* Internal Metrics */}
                <div className="metric-card glass-panel">
                    <div className="metric-header">
                        <span>Grid Stability</span>
                        <div className="metric-icon"><Zap size={20} /></div>
                    </div>
                    <div className="metric-value">
                        {telemetry.grid_voltage.toFixed(1)}<span className="metric-unit">V</span>
                    </div>
                    <div className="metric-footer">{telemetry.grid_frequency.toFixed(2)} Hz</div>
                </div>

                <div className="metric-card glass-panel">
                    <div className="metric-header">
                        <span>Thermal Load</span>
                        <div className="metric-icon"><ThermometerIcon size={20} /></div>
                    </div>
                    <div className="metric-value">
                        {telemetry.inverter_temperature.toFixed(1)}<span className="metric-unit">°C</span>
                    </div>
                    <div className="metric-footer">Ambient: {telemetry.ambient_temperature.toFixed(1)}°C</div>
                </div>

                {/* Mode Sensitive Metric 1 */}
                <div className="metric-card glass-panel highlight">
                    <div className="metric-header">
                        <span>Failure Probability</span>
                        <div className="metric-icon"><Shield size={20} /></div>
                    </div>
                    <div className="metric-value">
                        {(prediction.failure_probability * 100).toFixed(1)}<span className="metric-unit">%</span>
                    </div>
                    <div className="metric-footer">Confidence: {predictionMode === 'internal+external' ? 'High (94-96%)' : 'Standard (88-92%)'}</div>
                </div>

                {/* Weather Metrics (Enhanced Mode Only) */}
                {predictionMode === 'internal+external' ? (
                    <div className="metric-card glass-panel weather">
                        <div className="metric-header">
                            <span>Environmental risk</span>
                            <div className="metric-icon"><Cloud size={20} /></div>
                        </div>
                        <div className="metric-value">
                            {Math.round(telemetry.cloud_cover * 100)}<span className="metric-unit">%</span>
                        </div>
                        <div className="metric-footer">
                            <Wind size={14} /> {telemetry.wind_speed} km/h | <Droplets size={14} /> {telemetry.rainfall} mm
                        </div>
                    </div>
                ) : (
                    <div className="metric-card glass-panel upload-prompt" onClick={() => weatherInputRef.current.click()}>
                        <div className="metric-header">
                            <span>Enhanced Mode</span>
                            <div className="metric-icon"><Upload size={20} /></div>
                        </div>
                        <div className="metric-value action-text">Sync Weather</div>
                        <div className="metric-footer">Unlock +4% Accuracy</div>
                        <input
                            type="file"
                            ref={weatherInputRef}
                            style={{ display: 'none' }}
                            accept=".csv"
                            onChange={(e) => handleFileUpload('weather', e.target.files[0])}
                        />
                    </div>
                )}
            </div>

            <div className="main-content-grid">
                <div className="chart-area">
                    <div className="chart-card glass-panel">
                        <div className="chart-header">
                            <h3>Performance Dynamics</h3>
                            <div className="chart-controls">
                                <button className="upload-btn" onClick={() => telemetryInputRef.current.click()}>
                                    <Upload size={16} /> Sync Telemetry
                                </button>
                                <input
                                    type="file"
                                    ref={telemetryInputRef}
                                    style={{ display: 'none' }}
                                    accept=".csv"
                                    onChange={(e) => handleFileUpload('telemetry', e.target.files[0])}
                                />
                            </div>
                        </div>
                        <div style={{ height: '350px', width: '100%' }}>
                            <ResponsiveContainer>
                                <AreaChart data={dataHistory}>
                                    <defs>
                                        <linearGradient id="colorPower" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                        </linearGradient>
                                        <linearGradient id="colorWeather" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.2} />
                                            <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                                    <XAxis dataKey="time" stroke="#64748b" fontSize={10} hide />
                                    <YAxis stroke="#64748b" fontSize={10} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', backdropFilter: 'blur(8px)' }}
                                    />
                                    <Area type="monotone" dataKey="power" stroke="#3b82f6" fillOpacity={1} fill="url(#colorPower)" strokeWidth={2} name="DC Power" />
                                    {predictionMode === 'internal+external' && (
                                        <Area type="monotone" dataKey="cloud" stroke="#f59e0b" fillOpacity={1} fill="url(#colorWeather)" strokeWidth={2} name="Cloud Cover %" />
                                    )}
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>

                <div className="side-panels">
                    <div className="risk-analysis-panel glass-panel">
                        <h3>Risk Analysis Factors</h3>
                        <div className="reasons-list">
                            {prediction.reasons.map((reason, idx) => (
                                <div key={idx} className="reason-item">
                                    <div className="reason-bullet"></div>
                                    <span>{reason}</span>
                                </div>
                            ))}
                        </div>
                        <div className="anomaly-check">
                            <span>Anomaly Status:</span>
                            <span className={prediction.anomaly_detected ? 'detected' : 'normal'}>
                                {prediction.anomaly_detected ? 'UNUSUAL' : 'STABLE'}
                            </span>
                        </div>
                    </div>

                    <div className="alert-history-panel glass-panel">
                        <div className="panel-header">
                            <h3>Diagnostic Events</h3>
                            {alerts.length > 0 && <span className="alert-count">{alerts.length}</span>}
                        </div>
                        <div className="alert-list">
                            {alerts.length === 0 ? (
                                <div className="empty-alerts">
                                    <CheckCircle size={32} opacity={0.2} />
                                    <p>No active anomalies</p>
                                </div>
                            ) : (
                                alerts.map(alert => (
                                    <div key={alert.id} className={`alert-card ${alert.level}`}>
                                        <div className="alert-card-header">
                                            <span className="alert-cause">{alert.cause}</span>
                                            <span className="alert-time">{alert.time}</span>
                                        </div>
                                        <p className="alert-desc">{alert.desc}</p>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                </div>
            </div>
            {isUploading && (
                <div className="upload-overlay">
                    <div className="spinner"></div>
                    <p>Syncing Neural Assets...</p>
                </div>
            )}
        </div>
    );
};

export default Dashboard;

