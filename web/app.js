const TelanganaBounds = [[15.7, 76.8], [19.9, 81.8]];

const state = {
  map: null,
  drawnItems: null,
  polygon: null,
  runId: null,
  result: null,
};

const els = {
  asOfInput: document.getElementById('asOfInput'),
  cropInput: document.getElementById('cropInput'),
  soilInput: document.getElementById('soilInput'),
  irrigationInput: document.getElementById('irrigationInput'),
  polygonFallback: document.getElementById('polygonFallback'),
  polygonInput: document.getElementById('polygonInput'),
  analyzeBtn: document.getElementById('analyzeBtn'),
  formMeta: document.getElementById('formMeta'),
  runError: document.getElementById('runError'),
  dataWarning: document.getElementById('dataWarning'),
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
  resultsPanel: document.getElementById('resultsPanel'),
  mapStage: document.getElementById('mapStage'),
};

function todayIso() {
  const now = new Date();
  const tzOffset = now.getTimezoneOffset() * 60000;
  return new Date(now.getTime() - tzOffset).toISOString().slice(0, 10);
}

function initMap() {
  if (!window.L) {
    showPolygonFallback('Map library unavailable. Paste a GeoJSON Polygon to continue.');
    return;
  }

  state.map = L.map('map', { scrollWheelZoom: false, zoomControl: false }).fitBounds(TelanganaBounds);
  L.control.zoom({ position: 'bottomright' }).addTo(state.map);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 18,
    attribution: '&copy; OpenStreetMap contributors',
  }).addTo(state.map);

  state.drawnItems = new L.FeatureGroup();
  state.map.addLayer(state.drawnItems);

  if (!L.Control || typeof L.Control.Draw !== 'function' || !L.Draw) {
    showPolygonFallback('Drawing controls unavailable. Paste a GeoJSON Polygon to continue.');
    return;
  }

  const drawControl = new L.Control.Draw({
    position: 'topright',
    edit: { featureGroup: state.drawnItems },
    draw: {
      polygon: {
        shapeOptions: {
          color: '#0b765d',
          fillColor: '#35a66f',
          fillOpacity: 0.24,
          weight: 2,
        },
      },
      polyline: false,
      rectangle: false,
      circle: false,
      marker: false,
      circlemarker: false,
    },
  });

  state.map.addControl(drawControl);

  state.map.on(L.Draw.Event.CREATED, (event) => {
    const layer = event.layer;
    state.drawnItems.clearLayers();
    state.drawnItems.addLayer(layer);
    state.polygon = layer.toGeoJSON().geometry;
    els.formMeta.textContent = 'Polygon ready. Submit for analysis.';
  });
}

function showPolygonFallback(message) {
  els.polygonFallback.classList.remove('hidden');
  els.formMeta.textContent = message;
}

function polygonFromInput() {
  const input = els.polygonInput.value.trim();
  if (!input) return null;

  const parsed = JSON.parse(input);
  const geometry = parsed.type === 'Feature' ? parsed.geometry : parsed;
  if (
    !geometry ||
    geometry.type !== 'Polygon' ||
    !Array.isArray(geometry.coordinates) ||
    geometry.coordinates.length === 0
  ) {
    throw new Error('Enter a GeoJSON Polygon with coordinates.');
  }
  return geometry;
}

function setDefaults() {
  els.asOfInput.value = todayIso();
}

function modeBadge(result) {
  if (!result || !result.mode) return 'READY TO ANALYZE';
  const mode = result.mode;
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
    badge.textContent = event.node === 'observations' && event.status === 'start'
      ? 'Loading weather and satellite data'
      : `${event.node}:${event.status}`;
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

  const interval = [
    forecast.ndvi_z_p10,
    forecast.ndvi_z_p50,
    forecast.ndvi_z_p90,
  ].map((value) => (value == null ? null : Number(value)));
  if (!forecast.validated_for_season || interval.some((value) => value === null)) {
    els.forecastPanel.textContent =
      forecast.withheld_reason || 'No validated forecast values are available.';
    return;
  }

  const summary = document.createElement('p');
  summary.textContent =
    `14-day NDVI-z interval: ${interval[0].toFixed(2)} to ${interval[2].toFixed(2)} ` +
    `(median ${interval[1].toFixed(2)}).`;
  els.forecastPanel.innerHTML = '';
  els.forecastPanel.appendChild(summary);
  if (forecast.prob_event != null && Number.isFinite(Number(forecast.prob_event))) {
    const probability = document.createElement('p');
    probability.textContent =
      `Probability of NDVI-z at or below -1: ${(Number(forecast.prob_event) * 100).toFixed(1)}%.`;
    els.forecastPanel.appendChild(probability);
  }

  if (!window.Chart) {
    const note = document.createElement('small');
    note.className = 'muted';
    note.textContent = 'Chart library unavailable; numeric forecast is shown above.';
    els.forecastPanel.appendChild(note);
    return;
  }

  const canvas = document.createElement('canvas');
  canvas.height = 180;
  els.forecastPanel.appendChild(canvas);
  new Chart(canvas, {
    type: 'line',
    data: {
      labels: ['p10', 'p50', 'p90'],
      datasets: [{
        label: '14-day NDVI-z forecast interval',
        data: interval,
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
  } else {
    const latest = weather[weather.length - 1];
    els.weatherChart.textContent =
      `Chart library unavailable. ${weather.length} weather rows loaded through ` +
      `${latest.day || latest.date || 'the latest available day'}.`;
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
    const lowerText = text.toLowerCase();
    let kind = 'ok';
    if (
      lowerText.includes('error:') ||
      lowerText.includes('failed') ||
      lowerText.includes('unreachable') ||
      lowerText.includes('module_unavailable') ||
      lowerText.includes('cached') ||
      lowerText.includes('stale')
    ) {
      kind = 'warn';
    } else if (lowerText.includes('optical')) {
      kind = 'info';
    }
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
    const heading = document.createElement('strong');
    heading.textContent = name;
    li.append(heading, document.createElement('br'), document.createTextNode(detail));
    if (chunk) {
      const chunkLabel = document.createElement('small');
      chunkLabel.textContent = `Chunk: ${chunk}`;
      li.appendChild(chunkLabel);
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
    const claimText = document.createElement('div');
    const claimHeading = document.createElement('strong');
    claimHeading.textContent = claim.text || 'Advisory claim';
    claimText.appendChild(claimHeading);
    item.appendChild(claimText);
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
  const observationError = (Array.isArray(state.result.events) ? state.result.events : [])
    .find((event) => typeof event === 'string' && event.includes('error:geo.build_observations:'));
  if (observationError) {
    const timedOut = observationError.includes(':timeout:');
    els.dataWarning.textContent = timedOut
      ? 'Weather and satellite data loading timed out. No observations were used, so this result cannot provide a reliable field assessment. Check your connection or provider availability, then retry.'
      : `Weather and satellite data are unavailable (${observationError}). No observations were used, so this result is incomplete. Check provider availability and retry.`;
    els.dataWarning.classList.remove('hidden');
  } else {
    els.dataWarning.textContent = '';
    els.dataWarning.classList.add('hidden');
  }
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
  els.citationBody.innerHTML = '';
  if (!passage) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = 'No ICAR passage data was present in this run result.';
    els.citationBody.appendChild(empty);
  } else {
    const text = passage.text || 'No text available.';
    const meta = [passage.doc, passage.district, passage.plan_year, passage.page].filter(Boolean).join(' • ');
    const passageText = document.createElement('div');
    passageText.textContent = text;
    const metadata = document.createElement('small');
    metadata.textContent = meta;
    els.citationBody.append(passageText, document.createElement('br'), metadata);
  }
  els.citationModal.classList.add('visible');
}

function closeCitation() {
  els.citationModal.classList.remove('visible');
}

async function handleSubmit() {
  let polygon = state.polygon;
  try {
    polygon = polygon || polygonFromInput();
  } catch (error) {
    showRunError(error.message);
    return;
  }
  if (!polygon) {
    showRunError('Draw a field polygon or enter a GeoJSON Polygon before submitting.');
    return;
  }
  els.runError.textContent = '';
  els.runError.classList.add('hidden');
  els.dataWarning.textContent = '';
  els.dataWarning.classList.add('hidden');

  const payload = {
    polygon,
    as_of: els.asOfInput.value,
    crop: els.cropInput.value || null,
    soil: els.soilInput.value || null,
    irrigation_budget_mm: Number(els.irrigationInput.value || 0),
  };

  els.analyzeBtn.disabled = true;
  try {
    const response = await fetch('/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unable to start run.' }));
      showRunError(formatApiError(error.detail || 'Unable to start run.'));
      return;
    }

    const { run_id: runId } = await response.json();
    state.runId = runId;
    showStatus([]);
    await monitorRun(runId);
  } catch (error) {
    showRunError(`Analysis request failed: ${error.message}`);
  } finally {
    els.analyzeBtn.disabled = false;
  }
}

function formatApiError(detail) {
  if (Array.isArray(detail)) {
    return detail.map((item) => item.msg || JSON.stringify(item)).join('; ');
  }
  if (detail && typeof detail === 'object') {
    return JSON.stringify(detail);
  }
  return String(detail);
}

function showRunError(message) {
  els.runError.textContent = message;
  els.runError.classList.remove('hidden');
}

async function monitorRun(runId) {
  const stream = await fetch(`/runs/${runId}/events`);
  if (!stream.ok || !stream.body) {
    throw new Error(`Could not read run progress (HTTP ${stream.status}).`);
  }
  const reader = stream.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const runEvents = [];

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
        runEvents.push({ node: payload.node, status: payload.status, msg: payload.msg });
        showStatus(runEvents);
      } catch (error) {
        // ignore malformed fragments while streaming event updates
      }
    }
  }

  const result = await fetch(`/runs/${runId}`);
  if (!result.ok) {
    throw new Error(`Could not load run result (HTTP ${result.status}).`);
  }
  const json = await result.json();
  if (json.status === 'error') {
    showRunError(json.error || 'Analysis failed.');
    return;
  }
  renderResult(json);
}

function bindEvents() {
  els.analyzeBtn.addEventListener('click', handleSubmit);
  document.querySelectorAll('.rail-action[data-target]').forEach((button) => {
    button.addEventListener('click', () => {
      document.querySelectorAll('.rail-action[data-target]').forEach((item) => item.classList.remove('active'));
      button.classList.add('active');
      const target = document.getElementById(button.dataset.target);
      if (target === els.resultsPanel) target.scrollTop = 0;
      target?.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'nearest' });
      if (target === els.mapStage && state.map) state.map.invalidateSize();
    });
  });
  els.closeCitationModal.addEventListener('click', closeCitation);
  els.citationModal.addEventListener('click', (event) => {
    if (event.target === els.citationModal) closeCitation();
  });
}

document.addEventListener('DOMContentLoaded', () => {
  setDefaults();
  initMap();
  renderResult({ risk_level: 'insufficient_data', forecast: { withheld_reason: 'Awaiting forecast model validation.' }, events: [], ledger: [], weather_daily: [], climatology: {}, trigger_list: [], advisory: { claims: [] } });
  bindEvents();
});
