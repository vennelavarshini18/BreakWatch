const TelanganaBounds = [[12.5, 76.5], [19.9, 82.5]];

const state = {
  map: null,
  drawnItems: null,
  polygon: null,
  runId: null,
  result: null,
  sseController: null,
};

const els = {
  asOfInput: document.getElementById('asOfInput'),
  cropInput: document.getElementById('cropInput'),
  soilInput: document.getElementById('soilInput'),
  irrigationInput: document.getElementById('irrigationInput'),
  analyzeBtn: document.getElementById('analyzeBtn'),
  formMeta: document.getElementById('formMeta'),
  statusRow: document.getElementById('statusRow'),
  gaugeRing: document.getElementById('gaugeRing'),
  gaugeValue: document.getElementById('gaugeValue'),
  gaugeLabel: document.getElementById('gaugeLabel'),
  gaugeConfidence: document.getElementById('gaugeConfidence'),
  forecastPanel: document.getElementById('forecastPanel'),
  weatherChart: document.getElementById('weatherChart'),
  soilSummary: document.getElementById('soilSummary'),
  triggerList: document.getElementById('triggerList'),
  advisoryList: document.getElementById('advisoryList'),
  sourceChips: document.getElementById('sourceChips'),
  eventBanners: document.getElementById('eventBanners'),
  citationModal: document.getElementById('citationModal'),
  citationTitle: document.getElementById('citationTitle'),
  citationBody: document.getElementById('citationBody'),
  closeCitationModal: document.getElementById('closeCitationModal'),
};

function todayIso() {
  const now = new Date();
  const tzOffset = now.getTimezoneOffset() * 60000;
  return new Date(now.getTime() - tzOffset).toISOString().slice(0, 10);
}

function initMap() {
  state.map = L.map('map', { scrollWheelZoom: false }).fitBounds(TelanganaBounds);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 18,
    attribution: '&copy; OpenStreetMap contributors',
  }).addTo(state.map);

  state.drawnItems = new L.FeatureGroup();
  state.map.addLayer(state.drawnItems);

  const drawControl = new L.Control.Draw({
    edit: { featureGroup: state.drawnItems },
    draw: {
      polygon: true,
      polyline: false,
      rectangle: false,
      circle: false,
      marker: false,
      circlemarker: false,
      marker: false,
    },
  });

  state.map.addControl(drawControl);

  state.map.on(L.Draw.Event.CREATED, (event) => {
    const layer = event.layer;
    state.drawnItems.clearLayers();
    state.drawnItems.addLayer(layer);
    state.polygon = layer.toGeoJSON();
    els.formMeta.textContent = 'Polygon ready. Submit for analysis.';
  });
}

function setDefaults() {
  els.asOfInput.value = todayIso();
}

function modeBadge(result) {
  const mode = result && result.mode ? result.mode : 'replay';
  if (mode === 'live') return 'LIVE';
  const asOf = (result && result.field && result.field.as_of) || els.asOfInput.value;
  return `REPLAY . as of ${asOf} . real archived data, no look-ahead`;
}

function showStatus(events) {
  els.statusRow.innerHTML = '';
  if (!events || !events.length) {
    const badge = document.createElement('span');
    badge.className = 'badge insufficient';
    badge.textContent = 'Awaiting analysis';
    els.statusRow.appendChild(badge);
    return;
  }

  const unique = [...new Map(events.map((item) => [item.node + item.status + item.msg, item])).values()];
  unique.slice(0, 8).forEach((event) => {
    const badge = document.createElement('span');
    badge.className = 'badge';
    badge.textContent = `${event.node}:${event.status}`;
    els.statusRow.appendChild(badge);
  });
}

function renderGauge(result) {
  const metric = getMetricFromLedger(result, ['stress', 'p_stress', 'latest_std', 'latest_mean', 'stress_index']);
  const risk = (result && result.risk_level) || 'insufficient_data';

  if (!metric || risk === 'insufficient_data') {
    els.gaugeValue.innerHTML = 'N/A<small>stress</small>';
    els.gaugeLabel.textContent = 'Insufficient data';
    els.gaugeConfidence.textContent = 'No validated stress metric is available from the current result JSON.';
    els.gaugeRing.style.background = 'conic-gradient(#cbd5e1 0deg 360deg)';
    return;
  }

  const stress = Number(metric.value ?? 0);
  const confidence = getLedgerMetric(result, 'latest_std') || getLedgerMetric(result, 'confidence') || null;
  const safeStress = Number.isFinite(stress) ? stress : 0;
  const angle = Math.max(0, Math.min(360, safeStress * 180));
  els.gaugeRing.style.background = `conic-gradient(var(--accent) 0deg ${angle}deg, #e5e7eb ${angle}deg 360deg)`;
  els.gaugeValue.innerHTML = `${Number.isFinite(stress) ? stress.toFixed(2) : 'N/A'}<small>stress</small>`;
  els.gaugeLabel.textContent = risk === 'low' ? 'Low stress' : risk === 'watch' ? 'Watch' : 'High stress';
  if (confidence !== null) {
    els.gaugeConfidence.textContent = `Confidence from latest_std: ${Number(confidence).toFixed(2)}`;
  } else {
    els.gaugeConfidence.textContent = 'Confidence available from the ledger when present.';
  }
}

function renderForecast(result) {
  const forecast = result && result.forecast ? result.forecast : null;
  if (!forecast) {
    els.forecastPanel.innerHTML = '<div class="empty">No forecast output available.</div>';
    return;
  }

  if (forecast.withheld_reason) {
    const banner = document.createElement('div');
    banner.className = 'banner warn';
    banner.textContent = forecast.withheld_reason;
    els.forecastPanel.innerHTML = '';
    els.forecastPanel.appendChild(banner);
    return;
  }

  const interval = Array.isArray(forecast.stress_interval) ? forecast.stress_interval : [null, null, null];
  const values = interval.map((value) => (value == null ? null : Number(value)));
  const canvas = document.createElement('canvas');
  canvas.height = 180;
  els.forecastPanel.innerHTML = '';
  els.forecastPanel.appendChild(canvas);

  if (window.Chart) {
    new Chart(canvas, {
      type: 'line',
      data: {
        labels: ['low', 'mid', 'high'],
        datasets: [{
          label: 'Forecast stress interval',
          data: values,
          borderColor: '#0c7b5c',
          backgroundColor: 'rgba(12, 123, 92, 0.16)',
          pointRadius: 4,
          tension: 0.35,
          fill: true,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: { beginAtZero: false },
        },
      },
    });
  }
}

function renderWeather(result) {
  const weather = Array.isArray(result && result.weather_daily) ? result.weather_daily : [];
  if (!weather.length) {
    els.weatherChart.innerHTML = '<div class="empty">No weather data available.</div>';
    return;
  }

  const labels = weather.map((d) => d.day || d.date || 'day');
  const precip = weather.map((d) => (d.precip_mm == null ? null : Number(d.precip_mm)));
  const et0 = weather.map((d) => (d.et0_mm == null ? null : Number(d.et0_mm)));

  const canvas = document.createElement('canvas');
  els.weatherChart.innerHTML = '';
  els.weatherChart.appendChild(canvas);

  if (window.Chart) {
    new Chart(canvas, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            type: 'bar',
            label: 'Precipitation',
            data: precip,
            backgroundColor: '#60a5fa',
            borderRadius: 3,
          },
          {
            type: 'line',
            label: 'ET0',
            data: et0,
            borderColor: '#f97316',
            backgroundColor: 'rgba(249,115,22,0.2)',
            tension: 0.2,
            fill: false,
            pointRadius: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { x: { stacked: false }, y: { beginAtZero: false } },
      },
    });
  }
}

function renderSoilSummary(result) {
  const climatology = result && result.climatology ? result.climatology : {};
  const percentile = climatology.sm_root_percentile ?? climatology.sm_shallow_percentile ?? null;
  if (percentile == null) {
    els.soilSummary.innerHTML = '<div class="empty">No soil-moisture benchmark available.</div>';
    return;
  }
  els.soilSummary.innerHTML = `<strong>${Number(percentile).toFixed(1)}%</strong> soil moisture percentile from the benchmark.`;
}

function parseEvents(result) {
  const events = Array.isArray(result && result.events) ? result.events : [];
  const banners = [];
  events.forEach((event) => {
    const text = typeof event === 'string' ? event : event && event.msg ? event.msg : '';
    if (!text) return;
    const kind = text.includes('module_unavailable') ? 'warn' : text.includes('optical') ? 'info' : text.includes('cached') || text.includes('stale') ? 'warn' : 'ok';
    banners.push({ text, kind });
  });
  return banners;
}

function renderEventBanners(result) {
  const banners = parseEvents(result);
  els.eventBanners.innerHTML = '';
  if (!banners.length) {
    els.eventBanners.innerHTML = '<div class="empty">No quality or event banners from the current run.</div>';
    return;
  }
  banners.forEach((banner) => {
    const node = document.createElement('div');
    node.className = `banner ${banner.kind === 'warn' ? 'warn' : banner.kind === 'info' ? 'info' : 'ok'}`;
    node.textContent = banner.text;
    els.eventBanners.appendChild(node);
  });
}

function renderTriggers(result) {
  const triggers = Array.isArray(result && result.trigger_list) ? result.trigger_list : [];
  els.triggerList.innerHTML = '';
  if (!triggers.length) {
    const li = document.createElement('li');
    li.className = 'empty';
    li.textContent = 'No triggers were fired in the current result.';
    els.triggerList.appendChild(li);
    return;
  }

  triggers.forEach((trigger) => {
    const li = document.createElement('li');
    const name = trigger.name || trigger.id || 'Trigger';
    const detail = trigger.detail || '';
    const chunk = trigger.chunk_id || '';
    li.innerHTML = `<strong>${name}</strong><br>${detail}${chunk ? `<small>Chunk: ${chunk}</small>` : ''}`;
    if (chunk) {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'citation-chip';
      chip.textContent = 'ICAR citation';
      chip.addEventListener('click', () => openCitation(chunk, result));
      li.appendChild(chip);
    }
    els.triggerList.appendChild(li);
  });
}

function renderAdvisory(result) {
  const advisory = result && result.advisory ? result.advisory : null;
  const claims = advisory && Array.isArray(advisory.claims) ? advisory.claims : [];
  els.advisoryList.innerHTML = '';
  if (!claims.length) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = 'No advisory claims were produced.';
    els.advisoryList.appendChild(empty);
    return;
  }

  claims.forEach((claim) => {
    const item = document.createElement('div');
    item.className = 'list';
    item.innerHTML = `<div><strong>${claim.text || 'Advisory claim'}</strong></div>`;
    const chips = document.createElement('div');
    chips.className = 'citation-row';
    const chunkIds = Array.isArray(claim.chunk_ids) ? claim.chunk_ids : [];
    if (!chunkIds.length) {
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.textContent = 'No ICAR citation';
      chips.appendChild(chip);
    } else {
      chunkIds.forEach((chunkId) => {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'citation-chip';
        chip.textContent = `chunk:${chunkId}`;
        chip.addEventListener('click', () => openCitation(chunkId, result));
        chips.appendChild(chip);
      });
    }
    item.appendChild(chips);
    els.advisoryList.appendChild(item);
  });
}

function renderSourceChips(result) {
  const ledger = Array.isArray(result && result.ledger) ? result.ledger : [];
  const sensors = [...new Set(ledger.map((entry) => entry.key).filter(Boolean).filter((key) => /s1_|s2_|modis|soil/.test(key)))];
  els.sourceChips.innerHTML = '';
  if (!sensors.length) {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = 'No source chips available';
    els.sourceChips.appendChild(chip);
    return;
  }

  sensors.forEach((sensor) => {
    const chip = document.createElement('span');
    chip.className = `chip ${sensor.includes('s1_') ? 'experimental' : ''}`;
    chip.textContent = sensor.includes('s1_') ? `${sensor} experimental` : sensor;
    els.sourceChips.appendChild(chip);
  });
}

function renderResult(result) {
  state.result = result || {};
  els.statusRow.innerHTML = '';
  const badge = document.createElement('span');
  badge.className = `badge ${state.result.risk_level === 'low' ? 'low' : state.result.risk_level === 'watch' ? 'watch' : state.result.risk_level === 'high' ? 'high' : 'insufficient'}`;
  badge.textContent = state.result.risk_level === 'insufficient_data' ? 'Insufficient data' : state.result.risk_level || 'Unknown';
  els.statusRow.appendChild(badge);

  const mode = document.createElement('span');
  mode.className = 'badge';
  mode.textContent = modeBadge(state.result);
  els.statusRow.appendChild(mode);

  renderGauge(state.result);
  renderForecast(state.result);
  renderWeather(state.result);
  renderSoilSummary(state.result);
  renderTriggers(state.result);
  renderAdvisory(state.result);
  renderSourceChips(state.result);
  renderEventBanners(state.result);
}

function getMetricFromLedger(result, keys) {
  const ledger = Array.isArray(result && result.ledger) ? result.ledger : [];
  for (const key of keys) {
    const match = ledger.find((entry) => entry.key && entry.key.toLowerCase() === key.toLowerCase());
    if (match) return match;
  }
  return null;
}

function getLedgerMetric(result, key) {
  const metric = getMetricFromLedger(result, [key]);
  if (!metric) return null;
  const value = Number(metric.value);
  return Number.isFinite(value) ? value : null;
}

function openCitation(chunkId, result) {
  const passages = Array.isArray(result && result.passages) ? result.passages : [];
  const passage = passages.find((item) => item.chunk_id === chunkId) || null;

  els.citationTitle.textContent = `ICAR citation: ${chunkId}`;
  if (!passage) {
    els.citationBody.innerHTML = '<div class="empty">No ICAR passage data was present in this run result.</div>';
  } else {
    const text = passage.text || 'No text available.';
    const meta = [passage.doc, passage.district, passage.plan_year, passage.page].filter(Boolean).join(' • ');
    els.citationBody.innerHTML = `<div>${text}</div><br><small>${meta}</small>`;
  }
  els.citationModal.classList.add('visible');
}

function closeCitation() {
  els.citationModal.classList.remove('visible');
}

async function handleSubmit() {
  if (!state.polygon) {
    alert('Draw a field polygon before submitting.');
    return;
  }

  const payload = {
    polygon: state.polygon,
    as_of: els.asOfInput.value,
    crop: els.cropInput.value || null,
    soil: els.soilInput.value || null,
    irrigation_budget_mm: Number(els.irrigationInput.value || 0),
  };

  els.analyzeBtn.disabled = true;
  const response = await fetch('/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unable to start run.' }));
    alert(error.detail || 'Unable to start run.');
    els.analyzeBtn.disabled = false;
    return;
  }

  const { run_id } = await response.json();
  state.runId = run_id;
  showStatus([]);
  await monitorRun(run_id);
  els.analyzeBtn.disabled = false;
}

async function monitorRun(runId) {
  const stream = await fetch(`/runs/${runId}/events`);
  const reader = stream.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split(/\n\n/);
    buffer = lines.pop() || '';

    for (const block of lines) {
      const eventLine = block.split('\n').find((line) => line.startsWith('data:'));
      if (!eventLine) continue;
      try {
        const payload = JSON.parse(eventLine.replace(/^data:\s*/, ''));
        showStatus([{ node: payload.node, status: payload.status, msg: payload.msg }]);
      } catch (error) {
        // ignore malformed fragments while streaming event updates
      }
    }
  }

  const result = await fetch(`/runs/${runId}`);
  if (result.ok) {
    const json = await result.json();
    renderResult(json);
  }
}

function bindEvents() {
  els.analyzeBtn.addEventListener('click', handleSubmit);
  els.closeCitationModal.addEventListener('click', closeCitation);
  els.citationModal.addEventListener('click', (event) => {
    if (event.target === els.citationModal) closeCitation();
  });
}

document.addEventListener('DOMContentLoaded', () => {
  setDefaults();
  initMap();
  renderResult({ risk_level: 'insufficient_data', mode: 'replay', forecast: { withheld_reason: 'Awaiting forecast model validation.' }, events: [], ledger: [], weather_daily: [], climatology: {}, trigger_list: [], advisory: { claims: [] } });
  bindEvents();
});
