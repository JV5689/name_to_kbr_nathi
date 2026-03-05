import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area
} from 'recharts';
import {
    Activity, Zap, ThermometerIcon, TrendingUp, AlertTriangle, CheckCircle, Clock
} from 'lucide-react';
import './Dashboard.css';

const Dashboard = () => {
    const [inverters, setInverters] = useState(["INV-01"]);
    const [selectedInverter, setSelectedInverter] = useState("INV-01");
    const [dataHistory, setDataHistory] = useState([]);
    const [currentStatus, setCurrentStatus] = useState(null);
    const [alerts, setAlerts] = useState([]);

    useEffect(() => {
        const fetchInverters = async () => {
            try {
                const response = await axios.get('/api/inverters');
                setInverters(response.data);
                if (response.data.length > 0) {
                    setSelectedInverter(response.data[0]); // Set initial selected inverter
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

                // Use a formatted time string for the graph X axis
                const timeStr = new Date(telemetry.timestamp).toLocaleTimeString();

                const newDataPoint = {
                    time: timeStr,
                    voltage: telemetry.grid_voltage,
                    power: telemetry.dc_power,
                    efficiency: telemetry.inverter_efficiency,
                    temp: telemetry.inverter_temperature
                };

                setDataHistory(prev => {
                    const newHistory = [...prev, newDataPoint];
                    // Keep last 20 data points for the graph
                    if (newHistory.length > 20) return newHistory.slice(newHistory.length - 20);
                    return newHistory;
                });

                setCurrentStatus({ telemetry, prediction });

                // Add anomaly to alerts if detected
                if (prediction.anomaly_detected) {
                    setAlerts(prev => {
                        const newAlert = {
                            id: Date.now(),
                            time: timeStr,
                            level: prediction.risk_level.toLowerCase(),
                            cause: prediction.root_cause,
                            desc: prediction.maintenance_recommendation
                        };
                        const newAlerts = [newAlert, ...prev];
                        return newAlerts.length > 5 ? newAlerts.slice(0, 5) : newAlerts;
                    });
                }
            } catch (error) {
                console.error("Error fetching telemetry:", error);
            }
        };

        fetchTelemetry();
        const intervalId = setInterval(fetchTelemetry, 2000);
        return () => clearInterval(intervalId);
    }, [selectedInverter]);

    const handleInverterChange = (e) => {
        setSelectedInverter(e.target.value);
        setDataHistory([]); // Reset graph when switching inverters
        setAlerts([]); // Optional: reset alerts as well for clarity
    };

    if (!currentStatus) return <div className="dashboard-container" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>Loading System Diagnostics...</div>;

    const { telemetry, prediction } = currentStatus;

    const renderStatusBadge = () => {
        if (prediction.risk_level === "Low") {
            return (
                <div className="status-badge healthy glass-panel">
                    <CheckCircle size={18} /> System Optimal
                </div>
            );
        }
        if (prediction.risk_level === "Medium") {
            return (
                <div className="status-badge warning glass-panel">
                    <AlertTriangle size={18} /> Warning Detected
                </div>
            );
        }
        return (
            <div className="status-badge critical glass-panel">
                <Activity size={18} /> Critical Anomaly
            </div>
        );
    };

    return (
        <div className="dashboard-container">
            <header className="dashboard-header">
                <div>
                    <h1 className="dashboard-title text-gradient">SolarIQ Pulse</h1>
                    <p className="dashboard-subtitle">Real-time Inverter Telemetry & Failure Prediction</p>
                </div>
                <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                    <div className="glass-panel" style={{ padding: '0.5rem 1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Device:</span>
                        <select
                            value={selectedInverter}
                            onChange={handleInverterChange}
                            style={{
                                background: 'transparent',
                                color: 'white',
                                border: 'none',
                                fontWeight: 600,
                                cursor: 'pointer',
                                outline: 'none'
                            }}
                        >
                            {inverters.map(id => (
                                <option key={id} value={id} style={{ background: 'var(--surface)' }}>{id}</option>
                            ))}
                        </select>
                    </div>
                    {renderStatusBadge()}
                </div>
            </header>

            <div className="grid-layout">
                <div className="metric-card glass-panel">
                    <div className="metric-header">
                        <span>Grid Voltage</span>
                        <div className="metric-icon"><Zap size={20} /></div>
                    </div>
                    <div className="metric-value">
                        {telemetry.grid_voltage.toFixed(1)} <span className="metric-unit">V</span>
                    </div>
                </div>

                <div className="metric-card glass-panel">
                    <div className="metric-header">
                        <span>DC Power Output</span>
                        <div className="metric-icon"><Activity size={20} /></div>
                    </div>
                    <div className="metric-value">
                        {Math.round(telemetry.dc_power)} <span className="metric-unit">W</span>
                    </div>
                </div>

                <div className="metric-card glass-panel">
                    <div className="metric-header">
                        <span>Inverter Temp</span>
                        <div className="metric-icon"><ThermometerIcon size={20} /></div>
                    </div>
                    <div className="metric-value">
                        {telemetry.inverter_temperature.toFixed(1)} <span className="metric-unit">°C</span>
                    </div>
                </div>

                <div className="metric-card glass-panel">
                    <div className="metric-header">
                        <span>System Efficiency</span>
                        <div className="metric-icon"><TrendingUp size={20} /></div>
                    </div>
                    <div className="metric-value">
                        {telemetry.inverter_efficiency.toFixed(1)} <span className="metric-unit">%</span>
                    </div>
                </div>
            </div>

            <div className="chart-section">
                <div className="chart-card glass-panel">
                    <h3>Telemetry Timeline <span className="dashboard-subtitle" style={{ fontWeight: 400, marginLeft: 8 }}>(Live Voltage)</span></h3>
                    <div style={{ height: '300px', width: '100%' }}>
                        <ResponsiveContainer>
                            <AreaChart data={dataHistory} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                                <defs>
                                    <linearGradient id="colorVoltage" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8} />
                                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                                <XAxis dataKey="time" stroke="#94a3b8" fontSize={12} tickMargin={10} />
                                <YAxis domain={['auto', 'auto']} stroke="#94a3b8" fontSize={12} />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                                    itemStyle={{ color: '#f8fafc' }}
                                />
                                <Area type="monotone" dataKey="voltage" stroke="#3b82f6" fillOpacity={1} fill="url(#colorVoltage)" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="alert-panel glass-panel">
                    <h3>Diagnostic Alerts</h3>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem' }}>
                        <span style={{ color: 'var(--text-muted)' }}>AI Prediction Risk:</span>
                        <span style={{ fontWeight: 600, color: prediction.risk_level === 'Low' ? 'var(--secondary)' : (prediction.risk_level === 'Medium' ? 'var(--warning)' : 'var(--danger)') }}>
                            {(prediction.failure_probability * 100).toFixed(1)}% ({prediction.risk_level})
                        </span>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', overflowY: 'auto' }}>
                        {alerts.length === 0 ? (
                            <div style={{ textAlign: 'center', padding: '2rem 0', color: 'var(--text-muted)' }}>
                                <CheckCircle size={40} style={{ opacity: 0.2, marginBottom: '1rem', margin: '0 auto' }} />
                                <p>No anomalies detected</p>
                            </div>
                        ) : (
                            alerts.map(alert => (
                                <div key={alert.id} className={`alert-item ${alert.level} glass-panel`} style={{ padding: '0.75rem', borderRadius: '8px' }}>
                                    <div className="alert-header">
                                        <span>{alert.cause}</span>
                                        <span className="alert-time"><Clock size={12} style={{ display: 'inline', marginRight: 4, verticalAlign: 'middle' }} />{alert.time}</span>
                                    </div>
                                    <div className="alert-desc">{alert.desc}</div>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
