import React, { useState, useEffect } from 'react';
import { 
  Upload, 
  Image as ImageIcon, 
  Terminal, 
  Zap, 
  Activity, 
  Cpu, 
  CheckCircle, 
  AlertTriangle,
  ChevronDown, 
  ChevronUp, 
  RefreshCw,
  X,
  Sparkles
} from 'lucide-react';

const API_BASE = window.location.origin.includes('5173') || window.location.origin.includes('3000')
  ? 'http://127.0.0.1:8000'
  : window.location.origin;

const CLASS_DISPLAY_NAMES = {
  Zea_mays_Chulpi_Cancha: {
    title: "Zea mays Chulpi Cancha",
    alias: "Peruvian Cancha / Toasted Corn",
    fillClass: "fill-chulpi",
    description: "Traditional soft-kernel Peruvian corn variety used for toasted Cancha snacks."
  },
  Zea_mays_Indurata: {
    title: "Zea mays Indurata",
    alias: "Flint Corn / Hard Kernel",
    fillClass: "fill-indurata",
    description: "Hard outer layer flint corn with high endosperm density, commonly cultivated in South America."
  },
  Zea_mays_Rugosa: {
    title: "Zea mays Rugosa",
    alias: "Sweet Corn / Wrinkled Kernel",
    fillClass: "fill-rugosa",
    description: "Wrinkled sugary corn variant standard in sweet corn food production."
  }
};

export default function App() {
  const [health, setHealth] = useState(null);
  const [activeModel, setActiveModel] = useState('mobilenet_finetuned');
  const [models, setModels] = useState({});
  const [switchingModel, setSwitchingModel] = useState(false);
  
  const [tab, setTab] = useState('upload'); // 'upload' | 'samples' | 'path'
  
  // Input states
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  
  const [samples, setSamples] = useState([]);
  const [selectedSample, setSelectedSample] = useState(null);
  
  const [serverPathInput, setServerPathInput] = useState('./test_images/chulpi_cancha 3.jpg');
  
  // Result state
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showJson, setShowJson] = useState(false);

  // Fetch health and models on load
  const fetchHealthAndModels = async () => {
    try {
      const hRes = await fetch(`${API_BASE}/health`);
      if (hRes.ok) {
        const hData = await hRes.json();
        setHealth(hData);
        if (hData.model_name) setActiveModel(hData.model_name);
      } else {
        setHealth({ status: 'offline' });
      }

      const mRes = await fetch(`${API_BASE}/model`);
      if (mRes.ok) {
        const mData = await mRes.json();
        if (mData.available_models) {
          setModels(mData.available_models);
        }
      }

      const sRes = await fetch(`${API_BASE}/samples`);
      if (sRes.ok) {
        const sData = await sRes.json();
        setSamples(sData);
        if (sData.length > 0) {
          setSelectedSample(sData[0]);
        }
      }
    } catch (err) {
      console.error('API connection failed:', err);
      setHealth({ status: 'offline' });
    }
  };

  useEffect(() => {
    fetchHealthAndModels();
  }, []);

  const handleSwitchModel = async (modelName) => {
    if (modelName === activeModel || switchingModel) return;
    setSwitchingModel(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/models/switch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: modelName })
      });
      const data = await res.json();
      if (res.ok) {
        setActiveModel(data.active_model);
        fetchHealthAndModels();
      } else {
        setError(`Failed to switch model: ${data.detail}`);
      }
    } catch (err) {
      setError(`Model switch failed: ${err.message}`);
    } finally {
      setSwitchingModel(false);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setError(null);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setError(null);
    }
  };

  const runPrediction = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      if (tab === 'upload') {
        if (!selectedFile) {
          setError("Please select or drop an image file first.");
          setLoading(false);
          return;
        }
        const formData = new FormData();
        formData.append('file', selectedFile);

        const res = await fetch(`${API_BASE}/predict/upload`, {
          method: 'POST',
          body: formData
        });
        const data = await res.json();
        if (res.ok) {
          setResult(data);
        } else {
          setError(data.detail || "Prediction request failed.");
        }
      } else if (tab === 'samples') {
        if (!selectedSample) {
          setError("No sample image selected.");
          setLoading(false);
          return;
        }
        const res = await fetch(`${API_BASE}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_path: selectedSample.server_path })
        });
        const data = await res.json();
        if (res.ok) {
          setResult(data);
        } else {
          setError(data.detail || "Prediction request failed.");
        }
      } else if (tab === 'path') {
        if (!serverPathInput.trim()) {
          setError("Please enter a valid server image path.");
          setLoading(false);
          return;
        }
        const res = await fetch(`${API_BASE}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_path: serverPathInput.trim() })
        });
        const data = await res.json();
        if (res.ok) {
          setResult(data);
        } else {
          setError(data.detail || "Prediction request failed.");
        }
      }
    } catch (err) {
      setError(`API connection error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Header Bar */}
      <header className="header">
        <div className="brand">
          <div className="logo-icon">🌽</div>
          <div>
            <h1 className="brand-title">Corn Vision AI</h1>
            <p className="brand-subtitle">FastAPI MVP Image Classifier with React UI</p>
          </div>
        </div>

        <div className="status-bar">
          {health?.status === 'healthy' ? (
            <div className="badge badge-healthy">
              <span className="status-dot pulse"></span>
              FastAPI Online ({health.model_name})
            </div>
          ) : (
            <div className="badge badge-offline">
              <span className="status-dot"></span>
              FastAPI Disconnected
            </div>
          )}

          <button className="badge" onClick={fetchHealthAndModels} title="Refresh connection">
            <RefreshCw size={14} />
            Check API
          </button>
        </div>
      </header>

      {/* Model Selection Control Bar */}
      <section className="model-section">
        <div className="section-label">
          <Cpu size={16} /> Select Active Neural Model
        </div>
        <div className="model-grid">
          {/* MobileNet Finetuned */}
          <div 
            className={`model-card ${activeModel === 'mobilenet_finetuned' ? 'active' : ''}`}
            onClick={() => handleSwitchModel('mobilenet_finetuned')}
          >
            <div className="model-card-header">
              <span className="model-name">MobileNetV2 Fine-Tuned</span>
              <span className="model-tag" style={{ color: '#818cf8' }}>Keras · 23.7 MB</span>
            </div>
            <p className="model-desc">Full fine-tuned deep architecture. Highest accuracy & generalizability.</p>
          </div>

          {/* Light TFLite */}
          <div 
            className={`model-card ${activeModel === 'light' ? 'active' : ''}`}
            onClick={() => handleSwitchModel('light')}
          >
            <div className="model-card-header">
              <span className="model-name">Light Quantized</span>
              <span className="model-tag" style={{ color: '#34d399' }}>TFLite · 2.67 MB</span>
            </div>
            <p className="model-desc">Dynamic int8 quantized TFLite edge model. Maximum speed & low memory.</p>
          </div>

          {/* Baseline */}
          <div 
            className={`model-card ${activeModel === 'baseline' ? 'active' : ''}`}
            onClick={() => handleSwitchModel('baseline')}
          >
            <div className="model-card-header">
              <span className="model-name">Baseline Head</span>
              <span className="model-tag" style={{ color: '#fbbf24' }}>Keras · 11.6 MB</span>
            </div>
            <p className="model-desc">Frozen backbone head classifier model. Fast baseline comparison.</p>
          </div>
        </div>
      </section>

      {/* Main Grid: Inputs on Left, Results on Right */}
      <div className="main-grid">
        {/* Left Side: Input Workspace */}
        <div className="glass-card">
          {/* Tab Navigation */}
          <div className="tab-header">
            <button 
              className={`tab-btn ${tab === 'upload' ? 'active' : ''}`}
              onClick={() => setTab('upload')}
            >
              <Upload size={16} /> Drag & Drop
            </button>
            <button 
              className={`tab-btn ${tab === 'samples' ? 'active' : ''}`}
              onClick={() => setTab('samples')}
            >
              <ImageIcon size={16} /> Samples Gallery
            </button>
            <button 
              className={`tab-btn ${tab === 'path' ? 'active' : ''}`}
              onClick={() => setTab('path')}
            >
              <Terminal size={16} /> Server Path
            </button>
          </div>

          {/* Tab 1: Upload */}
          {tab === 'upload' && (
            <div>
              {previewUrl ? (
                <div className="preview-container">
                  <img src={previewUrl} alt="Upload preview" className="preview-image" />
                  <button 
                    className="remove-btn" 
                    onClick={() => { setSelectedFile(null); setPreviewUrl(null); }}
                  >
                    <X size={16} />
                  </button>
                </div>
              ) : (
                <div 
                  className={`drop-zone ${dragActive ? 'dragging' : ''}`}
                  onDragEnter={handleDrag}
                  onDragOver={handleDrag}
                  onDragLeave={handleDrag}
                  onDrop={handleDrop}
                  onClick={() => document.getElementById('file-input').click()}
                >
                  <Upload className="drop-icon" />
                  <p style={{ fontWeight: 600, marginBottom: '0.3rem' }}>Drop an image here or click to browse</p>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Supports JPG, JPEG, PNG, WEBP</p>
                  <input 
                    id="file-input" 
                    type="file" 
                    accept="image/*" 
                    style={{ display: 'none' }} 
                    onChange={handleFileChange}
                  />
                </div>
              )}
            </div>
          )}

          {/* Tab 2: Samples Gallery */}
          {tab === 'samples' && (
            <div>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.85rem' }}>
                Select a pre-packaged test image from the server:
              </p>
              <div className="samples-grid">
                {samples.map((s, idx) => (
                  <div 
                    key={idx}
                    className={`sample-card ${selectedSample?.filename === s.filename ? 'selected' : ''}`}
                    onClick={() => { setSelectedSample(s); setPreviewUrl(s.url); }}
                  >
                    <img src={s.url} alt={s.filename} className="sample-img" />
                    <div className="sample-label">{s.filename}</div>
                  </div>
                ))}
              </div>

              {selectedSample && (
                <div className="preview-container" style={{ maxHeight: '200px' }}>
                  <img src={selectedSample.url} alt="Selected sample" className="preview-image" />
                </div>
              )}
            </div>
          )}

          {/* Tab 3: Server Path */}
          {tab === 'path' && (
            <div>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.85rem' }}>
                Specify an image path local to the FastAPI server:
              </p>
              <input 
                type="text" 
                className="input-field" 
                value={serverPathInput} 
                onChange={(e) => setServerPathInput(e.target.value)}
                placeholder="/absolute/path/or/relative/test.jpg"
              />
            </div>
          )}

          {error && (
            <div style={{ padding: '0.85rem', background: 'rgba(239, 68, 68, 0.12)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: 'var(--radius-md)', color: '#f87171', fontSize: '0.85rem', marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <AlertTriangle size={16} />
              {error}
            </div>
          )}

          <div style={{ marginTop: '1.5rem' }}>
            <button 
              className="btn-primary"
              onClick={runPrediction}
              disabled={loading || health?.status !== 'healthy'}
            >
              {loading ? (
                <>
                  <RefreshCw className="spin" size={18} />
                  Executing Inference...
                </>
              ) : (
                <>
                  <Zap size={18} />
                  Classify Image
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right Side: Results & Metrics */}
        <div className="glass-card">
          {result ? (
            <div>
              {/* Result Header */}
              <div className="result-header">
                <div className="top-prediction">
                  <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)' }}>
                    Top Predicted Class
                  </div>
                  <div className="top-class-title">
                    {CLASS_DISPLAY_NAMES[result.predicted_class]?.title || result.predicted_class}
                  </div>
                  <div className="top-class-alias">
                    {CLASS_DISPLAY_NAMES[result.predicted_class]?.alias}
                  </div>
                </div>
                <div className="confidence-pill">
                  {(result.confidence * 100).toFixed(2)}%
                </div>
              </div>

              {/* Class Probabilities Distribution */}
              <div className="prob-list">
                <div style={{ fontSize: '0.82rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', fontWeight: 600 }}>
                  Probability Distribution
                </div>
                {Object.entries(result.probabilities).map(([cls, prob]) => {
                  const info = CLASS_DISPLAY_NAMES[cls] || { title: cls, fillClass: 'fill-indurata' };
                  const pct = (prob * 100).toFixed(2);
                  return (
                    <div key={cls} className="prob-item">
                      <div className="prob-labels">
                        <span className="prob-class-name">{info.title}</span>
                        <span className="prob-value">{pct}%</span>
                      </div>
                        <div className="prob-track">
                        <div 
                          className={`prob-fill ${info.fillClass}`} 
                          style={{ width: `${pct}%` }}
                        ></div>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Timing & Performance Metrics */}
              <div className="metrics-row">
                <div className="metric-card">
                  <div className="metric-val">{result.timing_ms.preprocessing} <span style={{ fontSize: '0.7rem' }}>ms</span></div>
                  <div className="metric-lbl">Preprocessing</div>
                </div>
                <div className="metric-card">
                  <div className="metric-val" style={{ color: '#818cf8' }}>{result.timing_ms.inference} <span style={{ fontSize: '0.7rem' }}>ms</span></div>
                  <div className="metric-lbl">Inference</div>
                </div>
                <div className="metric-card">
                  <div className="metric-val" style={{ color: '#34d399' }}>{result.timing_ms.total} <span style={{ fontSize: '0.7rem' }}>ms</span></div>
                  <div className="metric-lbl">Total Roundtrip</div>
                </div>
              </div>

              {/* Developer JSON view */}
              <div>
                <button className="json-toggle" onClick={() => setShowJson(!showJson)}>
                  <Activity size={14} />
                  {showJson ? 'Hide Raw API Response' : 'Show Raw API Response'}
                  {showJson ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                </button>
                {showJson && (
                  <pre className="json-box">
                    {JSON.stringify(result, null, 2)}
                  </pre>
                )}
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <Sparkles className="empty-icon" />
              <h3>Ready for Inference</h3>
              <p style={{ fontSize: '0.85rem', maxWidth: '300px' }}>
                Upload an image or pick a sample on the left, then click <strong>Classify Image</strong> to view neural network prediction metrics.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
