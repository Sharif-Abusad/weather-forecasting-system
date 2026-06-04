/* ═══════════════════════════════════════════════
   AtmoSense — Weather Intelligence
   script.js — fully aligned to new backend schema
   ═══════════════════════════════════════════════ */

// ── Icon mapping (Open-Meteo weathercode icons) ──
const ICON_MAP = {
  '01d':'☀️','01n':'🌙','02d':'⛅','02n':'☁️',
  '03d':'☁️','03n':'☁️','04d':'☁️','04n':'☁️',
  '09d':'🌧️','09n':'🌧️','10d':'🌦️','10n':'🌧️',
  '11d':'⛈️','11n':'⛈️','13d':'❄️','13n':'❄️',
  '50d':'🌫️','50n':'🌫️'
};

// ── Derive weather icon from cloud / precipitation data ──
function deriveIcon(c) {
  const precip = c.precipitation || 0;
  const cloud  = (c.total_cloud_cover || 0) * 100; // 0–1 → 0–100
  if (precip > 2)    return '⛈️';
  if (precip > 0.5)  return '🌧️';
  if (cloud > 85)    return '☁️';
  if (cloud > 40)    return '⛅';
  return '☀️';
}

// ── Derive text description from cloud/precip data ──
function deriveDesc(c) {
  const precip = c.precipitation || 0;
  const cloud  = (c.total_cloud_cover || 0) * 100;
  if (precip > 2)   return 'thunderstorm';
  if (precip > 0.5) return 'rain';
  if (cloud > 85)   return 'overcast';
  if (cloud > 60)   return 'mostly cloudy';
  if (cloud > 30)   return 'partly cloudy';
  return 'clear sky';
}

let PER_STEP_MAE = {};
let LSTM_META = {};
let XGB_META = {};
let XGB_IMPORTANCE = {};
let XGB_MAX = 0;

async function initMae() {
    const response = await fetch("/model-meta");
    const meta = await response.json();

    // Per-step MAE
    PER_STEP_MAE = Object.fromEntries(
        Object.entries(meta.lstm.per_step_metrics)
            .map(([step, metrics]) => [step, metrics.mae])
    );

    // lstm model metadata
    LSTM_META = {
        lookback: meta.lstm.lookback,
        forecast_n: meta.lstm.forecast_n,
        train_samples: meta.lstm.train_samples,
        n_features: meta.lstm.n_features,

        mae_celsius: meta.lstm.metrics.mae_celsius,
        rmse_celsius: meta.lstm.metrics.rmse_celsius,
        r2_score: meta.lstm.metrics.r2_score
    };

    // xgboost model metadata
    XGB_META = {
      accuracy: meta.xgboost.metrics_at_best_threshold.accuracy,
      auc: meta.xgboost.roc_auc,
      feature_importance: meta.xgboost.feature_importance_gain
    };
    
    XGB_IMPORTANCE = XGB_META.feature_importance;
    XGB_MAX = Math.max(...Object.values(XGB_IMPORTANCE));

    document.getElementById('lstm-mae').innerHTML=`
              <span class="pill-dot dot-lstm"></span>
          BiLSTM · MAE ±${LSTM_META.mae_celsius}°C
    `
    document.getElementById('xgboost-auc').innerHTML=`
          <span class="pill-dot dot-xgb"></span>
          XGBoost · AUC ${XGB_META.auc}
    `
}
initMae();
// ── XGBoost feature importance (from model_meta.json) ──
// const XGB_IMPORTANCE = {
//   precipitation:    125.12,
//   month:            33.19,
//   low_cloud_cover:  23.99,
//   high_cloud_cover: 19.50,
//   surface_pressure: 10.25,
//   total_cloud_cover:10.10,
//   temperature:       7.95,
//   wind_speed:        4.70,
//   humidity:          5.72,
//   temp_rolling_6:    6.22,
//   temp_lag_1:        6.19,
//   temp_lag_24:       5.77,
//   medium_cloud_cover:7.99,
// };



// ── Loader messages ──
const LOADER_STEPS = [
  'Fetching Open-Meteo data…',
  'Computing lag features…',
  'Running XGBoost classifier…',
  'Generating LSTM forecast…',
];
let loaderInterval = null;

// ── State management ─────────────────────────────
function setState(s) {
  clearInterval(loaderInterval);

  const panels = ['welcomeState','loadingState','errorState'];
  panels.forEach(id => {
    const el = document.getElementById(id);
    el.classList.remove('active');
    el.style.display = 'none';
  });

  const dash = document.getElementById('dashboard');
  dash.classList.remove('active');
  dash.style.display = 'none';

  if (s === 'welcome') {
    const el = document.getElementById('welcomeState');
    el.style.display = 'flex';
    el.classList.add('active');
  } else if (s === 'loading') {
    const el = document.getElementById('loadingState');
    el.style.display = 'flex';
    el.classList.add('active');
    // Cycle loader messages
    let i = 0;
    document.getElementById('loaderStep').textContent = LOADER_STEPS[0];
    loaderInterval = setInterval(() => {
      i = (i + 1) % LOADER_STEPS.length;
      document.getElementById('loaderStep').textContent = LOADER_STEPS[i];
    }, 900);
  } else if (s === 'error') {
    const el = document.getElementById('errorState');
    el.style.display = 'flex';
    el.classList.add('active');
  } else if (s === 'dashboard') {
    dash.style.display = 'block';
    // Small delay triggers CSS animation
    requestAnimationFrame(() => dash.classList.add('active'));
  }
}

// ── Fetch ────────────────────────────────────────
async function fetchWeather() {
  const city = document.getElementById('cityInput').value.trim();
  if (!city) return;
  setState('loading');

  try {
    const res = await fetch(`/weather/${encodeURIComponent(city)}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      document.getElementById('errorMsg').textContent = err.detail || 'City not found.';
      setState('error');
      return;
    }
    const data = await res.json();
    renderDashboard(data);
    setState('dashboard');
  } catch (e) {
    document.getElementById('errorMsg').textContent = 'Failed to connect. Is the server running?';
    setState('error');
  }
}

// ── Render ───────────────────────────────────────
function renderDashboard(data) {
  console.log((data))
  const c    = data.current;        // fetch_current_weather() dict
  const rain = data.rain_prediction; // { rain_tomorrow, probability, confidence, threshold_used }
  const hourly = data.hourly || []; // [{ time: "+1h", temp, humidity }]
  const forecast = data.forecast || []; // 5-day

  // ── City & meta ──────────────────────────────
  document.getElementById('cityLabel').textContent = c.city;
  const d = new Date();
  document.getElementById('cityMeta').textContent =
    `${d.toLocaleDateString('en-IN', { weekday:'long', day:'numeric', month:'long', year:'numeric' })} · Open-Meteo`;

  // ── Core stats strip ─────────────────────────
  const coreEl = document.getElementById('coreStats');
  const coreItems = [
    { val: `${c.wind_speed} m/s`, label: 'Wind Speed' },
    null, // divider
    { val: `${(c.total_cloud_cover * 100).toFixed(0)}%`, label: 'Cloud Cover' },
  ];
  coreEl.innerHTML = coreItems.map(item =>
    item === null
      ? `<div class="core-stat-divider"></div>`
      : `<div class="core-stat">
           <div class="core-stat-val">${item.val}</div>
           <div class="core-stat-label">${item.label}</div>
         </div>`
  ).join('');

  // ── Hero temperature ─────────────────────────
  document.getElementById('tempValue').textContent = Math.round(c.temperature);
  document.getElementById('feelsLike').textContent = `${c.feels_like}°C`;
  document.getElementById('heroIcon').textContent  = deriveIcon(c);
  document.getElementById('heroDesc').textContent  = deriveDesc(c);

  // Temperature range bar
  const minT = c.temp_min, maxT = c.temp_max, curT = c.temperature;
  const pct  = maxT > minT ? ((curT - minT) / (maxT - minT)) * 100 : 50;
  document.getElementById('tempRangeBar').innerHTML = `
    <span class="range-label range-min">${Math.round(minT)}°</span>
    <div class="range-track">
      <div class="range-segment" style="
        left: 0; width: ${pct}%;
        background: linear-gradient(90deg, var(--accent), var(--amber));
      "></div>
    </div>
    <span class="range-label range-max">${Math.round(maxT)}°</span>
  `;

  // ── Rain banner ──────────────────────────────
  const prob     = rain.probability;          // 0–1
  const probPct  = Math.round(prob * 100);
  const willRain = rain.rain_tomorrow;
  const conf     = rain.confidence;

  const rainEl = document.getElementById('rainBanner');
  const rainColor = willRain
    ? 'var(--accent)'
    : (prob > 0.35 ? 'var(--amber)' : 'var(--green)');

  document.getElementById('rainIcon').textContent    = willRain ? '🌧️' : (prob > 0.35 ? '🌤️' : '☀️');
  document.getElementById('rainTitle').textContent   = willRain ? 'Rain expected tomorrow' : 'No rain tomorrow';
  document.getElementById('rainSub').textContent     = conf;
  document.getElementById('rainProbVal').textContent = `${probPct}%`;
  document.getElementById('rainProbVal').style.color = rainColor;

  const fill = document.getElementById('rainProbFill');
  fill.style.background = `linear-gradient(90deg, ${rainColor}, ${rainColor}cc)`;
  fill.style.width = '0%';
  requestAnimationFrame(() => {
    setTimeout(() => { fill.style.width = `${probPct}%`; }, 200);
  });

  rainEl.style.borderColor = willRain
    ? 'rgba(59,158,255,0.25)'
    : (prob > 0.35 ? 'rgba(255,176,56,0.2)' : 'rgba(61,255,160,0.2)');

  // ── Cloud layers ─────────────────────────────
  const cloudLayers = [
    { name: 'High Cloud',   pct: Math.round(c.high_cloud_cover * 100),   cls: 'fill-high',   icon: '☁' },
    { name: 'Medium Cloud', pct: Math.round(c.medium_cloud_cover * 100), cls: 'fill-medium', icon: '⛅' },
    { name: 'Low Cloud',    pct: Math.round(c.low_cloud_cover * 100),    cls: 'fill-low',    icon: '🌫' },
  ];
  document.getElementById('cloudLayers').innerHTML = cloudLayers.map(layer => `
    <div class="cloud-layer">
      <div class="cloud-layer-header">
        <span class="cloud-layer-name">${layer.name}</span>
        <span class="cloud-layer-pct">${layer.pct}%</span>
      </div>
      <div class="cloud-layer-bar">
        <div class="cloud-layer-fill ${layer.cls}" style="width:${layer.pct}%"></div>
      </div>
    </div>
  `).join('');

  document.getElementById('cloudTotal').innerHTML = `
    <div>
      <div class="cloud-total-label">Total Coverage</div>
      <div style="display:flex;align-items:baseline;gap:4px">
        <div class="cloud-total-val">${Math.round(c.total_cloud_cover * 100)}</div>
        <div class="cloud-total-unit">%</div>
      </div>
    </div>
    <div style="text-align:right">
      <div class="cloud-total-label">Precipitation</div>
      <div style="display:flex;align-items:baseline;gap:4px;justify-content:flex-end">
        <div class="cloud-total-val" style="font-size:28px">${c.precipitation.toFixed(1)}</div>
        <div class="cloud-total-unit">mm</div>
      </div>
    </div>
  `;

  // ── LSTM Hourly forecast strip ───────────────
  const strip = document.getElementById('forecastStrip');
  if (hourly.length === 0) {
    strip.innerHTML = `<div style="color:var(--text-3);font-family:var(--mono);font-size:13px;padding:16px 0">No LSTM forecast available</div>`;
  } else {
    strip.innerHTML = hourly.map(h => `
      <div class="hour-card">
        <div class="hour-label">${h.time}</div>
        <div class="hour-temp-val">${Math.round(h.temp)}</div>
        <div class="hour-deg">°C</div>
      </div>
    `).join('');
  }
  document.getElementById('model-badge-lstm').innerHTML= `
    <span class="badge-dot"></span>
    Encoder-Decoder BiLSTM · MAE ±${LSTM_META.mae_celsius}°C
  `
  // ── Atmospheric card ─────────────────────────
  const atmoEl = document.getElementById('atmoStats');
  atmoEl.innerHTML = [
    { icon: '🌬️', label: 'Wind Speed',    val: c.wind_speed,     unit: 'm/s' },
    { icon: '📊',  label: 'Pressure',      val: c.surface_pressure, unit: 'hPa' },
    { icon: '🌡️', label: 'Temp (–1h lag)', val: c.temp_lag_1,     unit: '°C' },
    { icon: '⏱️',  label: 'Temp (–24h)',   val: c.temp_lag_24,    unit: '°C' },
    { icon: '📉',  label: 'Temp (6h avg)', val: c.temp_rolling_6, unit: '°C' },
  ].map(row => `
    <div class="atmo-row">
      <div class="atmo-icon-label">
        <div class="atmo-icon">${row.icon}</div>
        <div class="atmo-label">${row.label}</div>
      </div>
      <div class="atmo-val">${row.val}<span class="atmo-unit"> ${row.unit}</span></div>
    </div>
  `).join('');

  // ── Hydrology card ───────────────────────────
  const hydroEl = document.getElementById('hydroStats');
  const hydro = [
    { label: 'Humidity',    val: `${Math.round(c.humidity)}%`,   raw: c.humidity,           max: 100, cls: 'fill-humidity' },
    { label: 'Precipitation', val: `${c.precipitation.toFixed(1)} mm`, raw: Math.min(c.precipitation * 10, 100), max: 100, cls: 'fill-precip' },
    { label: 'Pressure',    val: `${c.surface_pressure} hPa`,   raw: ((c.surface_pressure - 980) / 60) * 100, max: 100, cls: 'fill-pressure' },
  ];
  hydroEl.innerHTML = hydro.map(h => `
    <div class="hydro-item">
      <div class="hydro-header">
        <div class="hydro-label">${h.label}</div>
        <div class="hydro-val">${h.val}</div>
      </div>
      <div class="hydro-bar-bg">
        <div class="hydro-bar-fill ${h.cls}" style="width:${Math.max(0, Math.min(100, h.raw))}%"></div>
      </div>
    </div>
  `).join('');

  // ── 5-day forecast ───────────────────────────
  const fl = document.getElementById('fiveDayList');
  fl.innerHTML = forecast.map(f => `
    <div class="forecast-row">
      <div class="forecast-date">${f.date}</div>
      <div class="forecast-icon">${ICON_MAP[f.icon] || '🌤️'}</div>
      <div class="forecast-desc">${f.description}</div>
      <div class="forecast-temps">
        <span class="f-max">${f.temp_max}°</span>
        <span class="f-sep">/</span>
        <span class="f-min">${f.temp_min}°</span>
      </div>
    </div>
  `).join('');

  // ── LSTM model info card ─────────────────────
  // const meta = data.lstm_meta || {};

  // Architecture chips — from model_meta.json
  const LSTM_ARCH = [
    { val: `${LSTM_META.lookback}`,  label: 'Lookback hrs' },
    { val: `${LSTM_META.forecast_n}`,   label: 'Forecast hrs' },
    { val: `${LSTM_META.n_features}`,   label: 'Features' },
    { val: 'BiLSTM', label: 'Architecture' },
    { val: `${LSTM_META.train_samples}`, label: 'Train samples' },
  ];
  document.getElementById('lstmArch').innerHTML = LSTM_ARCH.map(a => `
    <div class="lstm-arch-chip">
      <div class="lstm-arch-chip-val">${a.val}</div>
      <div class="lstm-arch-chip-label">${a.label}</div>
    </div>
  `).join('');

  // Per-step MAE bars — from model_meta.json
  // const PER_STEP_MAE = await loadMaeValues(); // { '+1h': 3.71, '+2h': 3.11, '+3h': 3.31, '+4h': 3.50, '+5h': 3.57 };
  const maxMae = Math.max(...Object.values(PER_STEP_MAE));
  document.getElementById('lstmMaeBars').innerHTML = Object.entries(PER_STEP_MAE).map(([step, mae]) => {
    const heightPct = Math.round((mae / (maxMae * 1.2)) * 100);
    return `
      <div class="lstm-mae-col">
        <div class="lstm-mae-val">${mae}</div>
        <div class="lstm-mae-bar-wrap">
          <div class="lstm-mae-bar" data-mae="${mae}" style="height:0%"
            data-h="${heightPct}"></div>
        </div>
        <div class="lstm-mae-step">${step}</div>
      </div>
    `;
  }).join('');

  // Animate LSTM bars
  requestAnimationFrame(() => {
    setTimeout(() => {
      document.querySelectorAll('.lstm-mae-bar').forEach(el => {
        el.style.height = el.dataset.h + '%';
      });
    }, 400);
  });

  // Key metrics
  const LSTM_METRICS = [
    { val: `±${LSTM_META.mae_celsius}°`,              label: 'Overall MAE',  cls: 'metric-blue' },
    { val: `${LSTM_META.rmse_celsius}°`,      label: 'RMSE',         cls: 'metric-purple' },
    { val: `${LSTM_META.r2_score}`,           label: 'R² Score',     cls: 'metric-green' },
  ];
  document.getElementById('lstmMetrics').innerHTML = LSTM_METRICS.map(m => `
    <div class="lstm-metric">
      <div class="lstm-metric-val ${m.cls}">${m.val}</div>
      <div class="lstm-metric-label">${m.label}</div>
    </div>
  `).join('');
  // <span class="badge-dot badge-xgb"></span>
  // ── XGBoost feature importance ───────────────
  // document.getElementById('model-badge').innerHTML = 
  document.getElementById('model-badge').innerHTML = `
  <span class="badge-dot badge-xgb"></span>
  Accuracy ${(XGB_META.accuracy * 100).toFixed(1)}% · AUC ${XGB_META.auc.toFixed(2)}
`;
  const xgbEl = document.getElementById('xgbFeatures');
  // Highlight features actually present in current observation
  const liveFeats = new Set(Object.keys(c));
  xgbEl.innerHTML = Object.entries(XGB_IMPORTANCE)
    .sort((a, b) => b[1] - a[1])
    .map(([name, score]) => {
      const barPct = Math.round((score / XGB_MAX) * 100);
      const isLive = liveFeats.has(name);
      return `
        <div class="xgb-feat">
          <div class="xgb-feat-header">
            <span class="xgb-feat-name" style="${isLive ? 'color:var(--text)' : ''}">${name}</span>
            <span class="xgb-feat-score">${score.toFixed(1)}</span>
          </div>
          <div class="xgb-feat-bar">
            <div class="xgb-feat-fill" style="width:0%;opacity:${isLive ? 1 : 0.45}"
              data-w="${barPct}"></div>
          </div>
        </div>
      `;
    }).join('');

  // Animate bars after paint
  requestAnimationFrame(() => {
    setTimeout(() => {
      document.querySelectorAll('.xgb-feat-fill').forEach(el => {
        el.style.width = el.dataset.w + '%';
      });
    }, 300);
  });
}

// ── Boot ─────────────────────────────────────────
setState('welcome');