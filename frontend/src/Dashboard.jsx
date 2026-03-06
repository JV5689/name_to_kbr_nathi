import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar, ComposedChart
} from 'recharts';
import {
    Activity, Zap, ThermometerIcon, TrendingUp, AlertTriangle, CheckCircle, Clock, Cloud, Wind, Droplets, Upload, Shield, Settings as SettingsIcon, LayoutDashboard, Brain, Database, RefreshCcw
} from 'lucide-react';
import './Dashboard.css';

const Dashboard = () => {
    const [view, setView] = useState('dashboard'); // 'dashboard', 'settings', or 'lab'
    const [analysisResult, setAnalysisResult] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [inverters, setInverters] = useState(["INV-01"]);
    const [selectedInverter, setSelectedInverter] = useState("INV-01");
    const [dataHistory, setDataHistory] = useState([]);
    const [currentStatus, setCurrentStatus] = useState(null);
    const [alerts, setAlerts] = useState([]);
    const [settings, setSettings] = useState({
        gemini_api_key: "",
        trend_window_hours: 48,
        refresh_rate_ms: 2500,
        risk_thresholds: { Medium: 0.5, High: 0.8, Critical: 0.95 },
        simulation_speed: 1.0
    });
    const [isSaving, setIsSaving] = useState(false);
    const wsRef = useRef(null);

    // Fetch initial data and settings
    useEffect(() => {
        const fetchInitial = async () => {
            try {
                const [invertersRes, settingsRes, historyRes] = await Promise.all([
                    axios.get('/api/inverters'),
                    axios.get('/api/settings'),
                    axios.get(`/api/telemetry/history?inverter_id=${selectedInverter}`)
                ]);
                setInverters(invertersRes.data);
                setSettings(settingsRes.data);

                // Process history for charts
                const historyData = historyRes.data.map(h => ({
                    time: new Date(h.timestamp).toLocaleTimeString(),
                    voltage: h.grid_voltage,
                    power: h.dc_power,
                    efficiency: h.inverter_efficiency,
                    temp: h.inverter_temperature,
                    full_timestamp: h.timestamp,
                    ...h
                }));
                setDataHistory(historyData);

                if (invertersRes.data.length > 0 && !selectedInverter) {
                    setSelectedInverter(invertersRes.data[0]);
                }
            } catch (error) {
                console.error("Error fetching initial data:", error);
            }
        };
        fetchInitial();
    }, [selectedInverter]);

    // WebSocket Connection
    useEffect(() => {
        if (wsRef.current) wsRef.current.close();

        // Standardizing host for development/production
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsHost = window.location.host === 'localhost:5173' ? 'localhost:8000' : window.location.host;
        const wsUrl = `${wsProtocol}//${wsHost}/ws/telemetry/${selectedInverter}`;

        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            const { telemetry, prediction } = data;
            const timeStr = new Date(telemetry.timestamp).toLocaleTimeString();

            const newDataPoint = {
                time: timeStr,
                voltage: telemetry.grid_voltage,
                power: telemetry.dc_power,
                efficiency: telemetry.inverter_efficiency,
                temp: telemetry.inverter_temperature,
                full_timestamp: telemetry.timestamp,
                ...telemetry
            };

            setDataHistory(prev => {
                const newHistory = [...prev, newDataPoint];
                return newHistory.length > 100 ? newHistory.slice(newHistory.length - 100) : newHistory;
            });

            setCurrentStatus({ telemetry, prediction });

            if (prediction.anomaly_detected || prediction.risk_level !== 'Low') {
                setAlerts(prev => {
                    const newAlert = {
                        id: Date.now(),
                        time: timeStr,
                        level: prediction.risk_level.toLowerCase(),
                        cause: prediction.root_cause || "System Event",
                        desc: prediction.maintenance_recommendation,
                        ai: prediction.ai_analysis
                    };
                    const exists = prev.some(a => a.cause === newAlert.cause && a.time === newAlert.time);
                    if (exists) return prev;
                    return [newAlert, ...prev].slice(0, 10);
                });
            }
        };

        ws.onerror = (err) => console.error("WS Error:", err);
        return () => ws.close();
    }, [selectedInverter]);

    const handleSaveSettings = async (e) => {
        e.preventDefault();
        setIsSaving(true);
        try {
            await axios.post('/api/settings', settings);
            alert("Settings synchronized with neural core.");
        } catch (error) {
            console.error("Save error:", error);
            alert("Update failed.");
        } finally {
            setIsSaving(false);
        }
    };

    if (!currentStatus && view === 'dashboard') return <div className="dashboard-container loading-full">Initializing Neural Monitoring...</div>;

    // Dataset Upload Logic
    const handleDatasetUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        setIsAnalyzing(true);
        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await axios.post('/api/dataset/analyze', formData);
            setAnalysisResult(res.data);
            if (res.data.status === 'Success') {
                alert(`Analysis complete! Processed ${res.data.summary.processed_rows} rows.`);
            } else if (res.data.status === 'Error') {
                alert(`Analysis failed: ${res.data.message}`);
            }
        } catch (error) {
            console.error("Analysis error:", error);
            const msg = error.response?.data?.detail || "Analysis failed. Ensure CSV format is correct.";
            alert(msg);
        } finally {
            setIsAnalyzing(false);
        }
    };

    const renderLab = () => (
        <div className="lab-view fade-in">
            <div className="glass-panel lab-container">
                <div className="lab-header">
                    <h2>Dataset Lab</h2>
                    <p>Upload historical CSV datasets for batch predictive analysis and anomaly detection.</p>
                </div>

                <div className="upload-section">
                    <label className="upload-dropzone">
                        <div className="upload-icon-container">
                            {isAnalyzing ? <RefreshCcw className="animate-spin" size={32} /> : <Upload size={32} />}
                        </div>
                        <span>{isAnalyzing ? "Analyzing Neural Patterns..." : "Drop Hardware CSV here or Click to Upload"}</span>
                        <input type="file" accept=".csv" onChange={handleDatasetUpload} hidden disabled={isAnalyzing} />
                    </label>
                </div>

                {analysisResult && (
                    <div className="analysis-results mt-4">
                        <div className="summary-grid">
                            <div className="summary-card glass-panel">
                                <span>Total Rows</span>
                                <strong>{analysisResult.summary.total_rows}</strong>
                            </div>
                            <div className="summary-card glass-panel">
                                <span>Processed</span>
                                <strong className="text-success">{analysisResult.summary.processed_rows}</strong>
                            </div>
                            <div className="summary-card glass-panel">
                                <span>Skipped</span>
                                <strong className="text-warning">{analysisResult.summary.skipped_rows}</strong>
                            </div>
                            <div className="summary-card glass-panel highlight">
                                <span>Critical Points</span>
                                <strong className="text-danger">{analysisResult.summary.critical_points_found}</strong>
                            </div>
                        </div>

                        {analysisResult.summary.missing_signals.length > 0 && (
                            <div className="alert-banner warning glass-panel">
                                <AlertTriangle size={18} />
                                <span>Warning: Missing critical features: {analysisResult.summary.missing_signals.join(', ')}</span>
                            </div>
                        )}

                        <div className="highlights-section mt-4">
                            <h3>Significant Events Found</h3>
                            <div className="table-container">
                                <table className="telemetry-table">
                                    <thead>
                                        <tr>
                                            <th>Timestamp</th>
                                            <th>Prob %</th>
                                            <th>Risk</th>
                                            <th>Power (W)</th>
                                            <th>Temp (°C)</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {analysisResult.highlights.map((point, i) => (
                                            <tr key={i}>
                                                <td>{point.timestamp}</td>
                                                <td>{(point.failure_probability * 100).toFixed(1)}%</td>
                                                <td className={`risk-text ${point.risk_level.toLowerCase()}`}>{point.risk_level}</td>
                                                <td>{point.dc_power}</td>
                                                <td>{point.temp}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );

    const renderDashboard = () => {
        const { telemetry, prediction } = currentStatus;
        const trend = prediction.trend_analysis || {};

        return (
            <div className="dashboard-content fade-in">
                <div className="grid-layout">
                    {/* Live Metrics */}
                    <div className="metric-card glass-panel">
                        <div className="metric-header">
                            <span>Electrical Signature</span>
                            <div className="metric-icon"><Zap size={20} /></div>
                        </div>
                        <div className="metric-value">
                            {telemetry.grid_voltage.toFixed(1)}<span className="metric-unit">V</span>
                        </div>
                        <div className="metric-footer">
                            <Activity size={12} /> {telemetry.grid_frequency.toFixed(2)} Hz | {trend.voltage_stability === 'High' ? 'Stable' : 'Unstable'}
                        </div>
                    </div>

                    <div className="metric-card glass-panel">
                        <div className="metric-header">
                            <span>Thermal Profile</span>
                            <div className="metric-icon"><ThermometerIcon size={20} /></div>
                        </div>
                        <div className="metric-value">
                            {telemetry.inverter_temperature.toFixed(1)}<span className="metric-unit">°C</span>
                        </div>
                        <div className="metric-footer">
                            Trend: <span className={trend.temp_trend === 'Increasing' ? 'text-warning' : ''}>{trend.temp_trend}</span>
                        </div>
                    </div>

                    <div className="metric-card glass-panel highlight">
                        <div className="metric-header">
                            <span>Failure Probability</span>
                            <div className="metric-icon"><Shield size={20} /></div>
                        </div>
                        <div className="metric-value">
                            {(prediction.failure_probability * 100).toFixed(1)}<span className="metric-unit">%</span>
                        </div>
                        <div className="metric-footer">Risk: <span className={`risk-text ${prediction.risk_level.toLowerCase()}`}>{prediction.risk_level}</span></div>
                    </div>

                    <div className="metric-card glass-panel efficiency">
                        <div className="metric-header">
                            <span>Conversion Yield</span>
                            <div className="metric-icon"><TrendingUp size={20} /></div>
                        </div>
                        <div className="metric-value">
                            {telemetry.inverter_efficiency.toFixed(1)}<span className="metric-unit">%</span>
                        </div>
                        <div className="metric-footer">Delta: {trend.efficiency_delta > 0 ? '+' : ''}{trend.efficiency_delta}% (30m)</div>
                    </div>
                </div>

                <div className="main-content-grid">
                    <div className="chart-area">
                        <div className="chart-card glass-panel">
                            <div className="chart-header">
                                <h3>Trend Analysis (Internal Data)</h3>
                                <div className="chart-legend">
                                    <span className="legend-item"><div className="dot power"></div> Power (kW)</span>
                                    <span className="legend-item"><div className="dot temp"></div> Temperature (°C)</span>
                                </div>
                            </div>
                            <div style={{ height: '380px', width: '100%' }}>
                                <ResponsiveContainer>
                                    <ComposedChart data={dataHistory}>
                                        <defs>
                                            <linearGradient id="colorPower" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                                                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                                        <XAxis dataKey="time" stroke="#64748b" fontSize={10} minTickGap={30} />
                                        <YAxis yAxisId="left" stroke="#64748b" fontSize={10} name="Power" />
                                        <YAxis yAxisId="right" orientation="right" stroke="#64748b" fontSize={10} name="Temp" />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', backdropFilter: 'blur(8px)' }}
                                        />
                                        <Area yAxisId="left" type="monotone" dataKey="power" stroke="#3b82f6" fillOpacity={1} fill="url(#colorPower)" strokeWidth={2} name="DC Power" />
                                        <Line yAxisId="right" type="monotone" dataKey="temp" stroke="#ef4444" strokeWidth={2} dot={false} name="Inverter Temp" />
                                    </ComposedChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        {/* All Data Explorer */}
                        <div className="data-explorer-card glass-panel mt-4">
                            <div className="panel-header">
                                <h3>Telemetric Matrix</h3>
                                <Database size={18} opacity={0.5} />
                            </div>
                            <div className="table-container">
                                <table className="telemetry-table">
                                    <thead>
                                        <tr>
                                            <th>Signal Name</th>
                                            <th>Value</th>
                                            <th>Nominal Range</th>
                                            <th>Status</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {Object.entries(telemetry).filter(([k]) => k !== 'timestamp').map(([key, val]) => (
                                            <tr key={key}>
                                                <td className="signal-key">{key.replace(/_/g, ' ')}</td>
                                                <td className="signal-val">{typeof val === 'number' ? val.toFixed(2) : val}</td>
                                                <td>-</td>
                                                <td><span className="status-dot-small active"></span></td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>

                    <div className="side-panels">
                        <div className="genai-panel glass-panel">
                            <div className="panel-header">
                                <h3>GenAI Analysis</h3>
                                <Brain size={18} className="text-primary-light" />
                            </div>
                            <div className="ai-content">
                                {prediction.ai_analysis ? (
                                    <div className="ai-bubble">
                                        <p>{prediction.ai_analysis}</p>
                                    </div>
                                ) : (
                                    <div className="ai-placeholder">
                                        <Clock size={32} opacity={0.2} />
                                        <p>Awaiting trend saturation for AI diagnostic...</p>
                                    </div>
                                )}
                            </div>
                        </div>

                        <div className="alert-history-panel glass-panel">
                            <div className="panel-header">
                                <h3>Neural Events</h3>
                                {alerts.length > 0 && <span className="alert-count">{alerts.length}</span>}
                            </div>
                            <div className="alert-list">
                                {alerts.length === 0 ? (
                                    <div className="empty-alerts">
                                        <CheckCircle size={32} opacity={0.2} />
                                        <p>No anomalies detected</p>
                                    </div>
                                ) : (
                                    alerts.map(alert => (
                                        <div key={alert.id} className={`alert-card ${alert.level}`}>
                                            <div className="alert-card-header">
                                                <span className="alert-cause">{alert.cause}</span>
                                                <span className="alert-time">{alert.time}</span>
                                            </div>
                                            <p className="alert-desc">{alert.desc}</p>
                                            {alert.ai && <div className="alert-ai-insight">AI Insight: {alert.ai}</div>}
                                        </div>
                                    ))
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        );
    };

    const renderSettings = () => (
        <div className="settings-view fade-in">
            <div className="glass-panel settings-form-container">
                <div className="settings-header">
                    <h2>Neural Core Configuration</h2>
                    <p>Manage API keys, temporal thresholds, and simulation parameters.</p>
                </div>

                <form onSubmit={handleSaveSettings} className="settings-form">
                    <div className="form-group">
                        <label>Google Gemini API Key</label>
                        <input
                            type="password"
                            value={settings.gemini_api_key}
                            onChange={(e) => setSettings({ ...settings, gemini_api_key: e.target.value })}
                            placeholder="Enter Key for AI Analytics"
                        />
                        <span className="input-hint">Used for trend-based root cause identification.</span>
                    </div>

                    <div className="form-row">
                        <div className="form-group">
                            <label>Trend Window (Hours)</label>
                            <input
                                type="number"
                                value={settings.trend_window_hours}
                                onChange={(e) => setSettings({ ...settings, trend_window_hours: parseInt(e.target.value) })}
                            />
                        </div>
                        <div className="form-group">
                            <label>Refresh Rate (ms)</label>
                            <input
                                type="number"
                                value={settings.refresh_rate_ms}
                                onChange={(e) => setSettings({ ...settings, refresh_rate_ms: parseInt(e.target.value) })}
                            />
                        </div>
                    </div>

                    <div className="thresholds-section">
                        <h3>Risk Sensitivity</h3>
                        {Object.entries(settings.risk_thresholds).map(([level, val]) => (
                            <div key={level} className="threshold-row">
                                <label>{level}</label>
                                <input
                                    type="range" min="0" max="1" step="0.05"
                                    value={val}
                                    onChange={(e) => {
                                        const newThr = { ...settings.risk_thresholds, [level]: parseFloat(e.target.value) };
                                        setSettings({ ...settings, risk_thresholds: newThr });
                                    }}
                                />
                                <span>{Math.round(val * 100)}%</span>
                            </div>
                        ))}
                    </div>

                    <button type="submit" className="save-btn" disabled={isSaving}>
                        {isSaving ? <RefreshCcw className="animate-spin" size={18} /> : <Shield size={18} />}
                        Sync Configuration
                    </button>
                </form>
            </div>
        </div>
    );

    return (
        <div className="dashboard-container">
            <header className="dashboard-header">
                <div>
                    <h1 className="dashboard-title text-gradient">SolarIQ Pulse</h1>
                    <p className="dashboard-subtitle">Trend-Based Predictive Maintenance</p>
                </div>

                <nav className="main-nav glass-panel">
                    <button
                        className={`nav-item ${view === 'dashboard' ? 'active' : ''}`}
                        onClick={() => setView('dashboard')}
                    >
                        <LayoutDashboard size={18} /> Dashboard
                    </button>
                    <button
                        className={`nav-item ${view === 'settings' ? 'active' : ''}`}
                        onClick={() => setView('settings')}
                    >
                        <SettingsIcon size={18} /> Settings
                    </button>
                    <button
                        className={`nav-item ${view === 'lab' ? 'active' : ''}`}
                        onClick={() => setView('lab')}
                    >
                        <Brain size={18} /> Dataset Lab
                    </button>
                </nav>

                <div className="header-actions">
                    <div className="device-selector glass-panel">
                        <select value={selectedInverter} onChange={(e) => setSelectedInverter(e.target.value)}>
                            {inverters.map(id => (
                                <option key={id} value={id}>{id}</option>
                            ))}
                        </select>
                    </div>
                </div>
            </header>

            {view === 'dashboard' ? renderDashboard() : (view === 'settings' ? renderSettings() : renderLab())}
        </div>
    );
};

export default Dashboard;

