// AtlasCare — SRE & Engineering Dashboard

const API_BASE = window.location.origin;

const state = {
  traces: [],
  activeTraceId: null,
  currentFilter: 'all',
  logFilter: 'all',
  autoScroll: true,
};

// ==================== INIT ====================

document.addEventListener('DOMContentLoaded', () => {
  loadTraces();
  setupFilters();
  setupActions();
  loadRecentLogs();
  startLogStream();

  // React when customer tab writes new traces
  window.addEventListener('storage', (e) => {
    if (e.key === 'atlascare_traces' || e.key === 'atlascare_stats') {
      loadTraces();
    }
  });
});

// ==================== GLOBAL STATS ====================

function loadTraces() {
  try {
    const raw = localStorage.getItem('atlascare_traces');
    const rawStats = localStorage.getItem('atlascare_stats');
    state.traces = raw ? JSON.parse(raw) : [];
    renderTraceList();
    renderGlobalStats(rawStats ? JSON.parse(rawStats) : null);
  } catch (err) {
    console.error('ops: failed to load traces', err);
  }
}

function renderGlobalStats(stats) {
  if (!stats) return;
  document.getElementById('totalRequests').textContent = stats.totalRequests;
  const avg = stats.totalRequests > 0 ? Math.round(stats.totalLatency / stats.totalRequests) : 0;
  document.getElementById('avgLatency').textContent = avg > 0 ? `${avg}ms` : '—';
  const rate = stats.totalRequests > 0 ? Math.round((stats.successfulRequests / stats.totalRequests) * 100) : 100;
  document.getElementById('successRate').textContent = stats.totalRequests > 0 ? `${rate}%` : '—';
}

// ==================== TRACE LIST ====================

function renderTraceList() {
  const list = document.getElementById('traceList');
  const filtered = state.currentFilter === 'all'
    ? state.traces
    : state.traces.filter(t => t.type === state.currentFilter);

  if (filtered.length === 0) {
    list.innerHTML = `<div class="empty-state-small"><p>No traces${state.currentFilter !== 'all' ? ` for ${state.currentFilter}` : ''} yet.</p><span>Send a message in Customer Chat.</span></div>`;
    return;
  }

  list.innerHTML = '';
  filtered.forEach(trace => {
    const item = document.createElement('div');
    item.className = 'trace-item' + (trace.id === state.activeTraceId ? ' active' : '');
    item.dataset.traceId = trace.id;

    const ok = trace.success;
    item.innerHTML = `
      <div class="trace-header">
        <span class="trace-id">${trace.id}</span>
        <span class="trace-badge ${trace.type}">${trace.type}</span>
      </div>
      <div class="trace-purpose">${esc(truncate(trace.purpose, 80))}</div>
      <div class="trace-meta">
        ${trace.orderIds?.length ? `<span class="trace-meta-item">📦 ${trace.orderIds[0]}</span>` : ''}
        ${trace.customerId ? `<span class="trace-meta-item">👤 ${trace.customerId}</span>` : ''}
        <span class="trace-status ${ok ? 'success' : 'error'}">${ok ? '✓' : '✗'} ${ok ? 'success' : 'error'}</span>
        <span class="trace-meta-item">⏱ ${trace.latency}ms</span>
      </div>`;

    item.addEventListener('click', () => selectTrace(trace));
    list.appendChild(item);
  });
}

function selectTrace(trace) {
  state.activeTraceId = trace.id;
  // Update active class without full re-render
  document.querySelectorAll('.trace-item').forEach(el => {
    el.classList.toggle('active', el.dataset.traceId === trace.id);
  });
  renderTraceDetail(trace);
}

// ==================== TRACE DETAIL ====================

function renderTraceDetail(trace) {
  const body = document.getElementById('traceDetailBody');
  const hint = document.getElementById('traceDetailHint');
  hint.textContent = `${trace.id} · ${trace.type} · ${formatTs(trace.timestamp)}`;

  const ok = trace.success;
  const tools = trace.toolCalls || [];

  body.innerHTML = `
    <div class="td-meta-grid">
      <div class="td-meta-card">
        <div class="td-meta-label">Journey</div>
        <div class="td-meta-value"><span class="trace-badge ${trace.type}">${trace.type}</span></div>
      </div>
      <div class="td-meta-card">
        <div class="td-meta-label">Latency</div>
        <div class="td-meta-value">${trace.latency}ms</div>
      </div>
      <div class="td-meta-card">
        <div class="td-meta-label">Status</div>
        <div class="td-meta-value"><span class="trace-status ${ok ? 'success' : 'error'}">${ok ? '✓ Success' : '✗ Failed'}</span></div>
      </div>
      ${trace.customerId ? `<div class="td-meta-card"><div class="td-meta-label">Customer</div><div class="td-meta-value">${trace.customerId}</div></div>` : ''}
      ${trace.orderIds?.length ? `<div class="td-meta-card"><div class="td-meta-label">Order IDs</div><div class="td-meta-value">${trace.orderIds.join(', ')}</div></div>` : ''}
      ${trace.amounts?.length ? `<div class="td-meta-card"><div class="td-meta-label">Amounts</div><div class="td-meta-value">₹${trace.amounts.join(', ₹')}</div></div>` : ''}
    </div>

    <div class="td-tools-heading">
      <svg width="13" height="13" fill="currentColor" viewBox="0 0 20 20">
        <path fill-rule="evenodd" d="M12.316 3.051a1 1 0 01.633 1.265l-4 12a1 1 0 11-1.898-.632l4-12a1 1 0 011.265-.633zM5.707 6.293a1 1 0 010 1.414L3.414 10l2.293 2.293a1 1 0 11-1.414 1.414l-3-3a1 1 0 010-1.414l3-3a1 1 0 011.414 0zm8.586 0a1 1 0 011.414 0l3 3a1 1 0 010 1.414l-3 3a1 1 0 11-1.414-1.414L16.586 10l-2.293-2.293a1 1 0 010-1.414z" clip-rule="evenodd"/>
      </svg>
      Called Functions (${tools.length})
    </div>

    ${tools.length === 0
      ? '<p style="color:var(--text-muted);font-size:0.82rem">No tool calls recorded for this trace.</p>'
      : tools.map((tc, i) => `
        <div class="td-tool-item ${tc.success ? 'success' : 'error'}">
          <div class="td-tool-header">
            <span class="td-tool-name">${i + 1}. ${tc.tool_name}</span>
            <span class="td-tool-latency">
              ${tc.latency_ms}ms
              <span class="trace-status ${tc.success ? 'success' : 'error'}" style="padding:0.1rem 0.35rem">${tc.success ? '✓' : '✗'}</span>
            </span>
          </div>
          <div class="td-io-label">Input</div>
          <pre class="td-pre">${esc(JSON.stringify(tc.input, null, 2))}</pre>
          <div class="td-io-label">Output</div>
          <pre class="td-pre">${esc(JSON.stringify(tc.output, null, 2))}</pre>
        </div>`).join('')}

    <p style="margin-top:1rem;font-size:0.78rem;color:var(--text-muted)">
      Message: <em>"${esc(trace.purpose)}"</em>
    </p>`;
}

// ==================== SERVER LOGS ====================

async function loadRecentLogs() {
  try {
    const res = await fetch(`${API_BASE}/logs/recent?n=200`);
    if (!res.ok) return;
    const data = await res.json();
    const body = document.getElementById('logBody');
    body.innerHTML = '';   // clear placeholder
    data.logs.forEach(entry => appendLogRow(entry));
    scrollLogsToBottom();
  } catch (err) {
    console.error('ops: failed to load recent logs', err);
  }
}

function startLogStream() {
  const badge = document.getElementById('logLiveBadge');
  const evtSource = new EventSource(`${API_BASE}/logs/stream`);

  evtSource.onopen = () => {
    badge.textContent = '● LIVE';
    badge.classList.remove('disconnected');
  };

  evtSource.onmessage = (e) => {
    let entry;
    try { entry = JSON.parse(e.data); } catch { entry = { message: e.data, level: 'INFO' }; }
    const body = document.getElementById('logBody');
    // Remove placeholder if still there
    const ph = body.querySelector('.ops-placeholder');
    if (ph) ph.remove();
    appendLogRow(entry);
    if (state.autoScroll) scrollLogsToBottom();
  };

  evtSource.onerror = () => {
    badge.textContent = '● DISCONNECTED';
    badge.classList.add('disconnected');
    // Auto-reconnect is handled by EventSource natively
  };
}

function appendLogRow(entry) {
  const level = (entry.level || 'INFO').toUpperCase();
  const filter = state.logFilter;
  if (filter !== 'all' && level !== filter) return;

  const body = document.getElementById('logBody');
  const row = document.createElement('div');
  row.className = `log-row level-${level}`;
  row.dataset.level = level;

  const ts = entry.timestamp ? fmtLogTs(entry.timestamp) : '';
  const event = entry.event || '';
  const msg = entry.message || '';
  const traceId = entry.trace_id || '';

  row.innerHTML = `
    <span class="log-ts">${ts}</span>
    <span class="log-level ${level}">${level}</span>
    ${event ? `<span class="log-event" title="${esc(event)}">${esc(event)}</span>` : ''}
    <span class="log-msg">${esc(buildLogMessage(entry, msg))}</span>
    ${traceId ? `<span class="log-trace-id">${traceId}</span>` : ''}`;

  body.appendChild(row);
}

function buildLogMessage(entry, baseMsg) {
  // Surface the most useful fields alongside the message
  const skip = new Set(['timestamp','level','message','event','service','trace_id','session_id']);
  const extras = Object.entries(entry)
    .filter(([k]) => !skip.has(k))
    .map(([k, v]) => `${k}=${typeof v === 'object' ? JSON.stringify(v) : v}`)
    .join('  ');
  return baseMsg ? (extras ? `${baseMsg}  ${extras}` : baseMsg) : extras;
}

function scrollLogsToBottom() {
  const body = document.getElementById('logBody');
  body.scrollTop = body.scrollHeight;
}

function reapplyLogFilter() {
  const body = document.getElementById('logBody');
  const filter = state.logFilter;
  body.querySelectorAll('.log-row').forEach(row => {
    row.style.display = (filter === 'all' || row.dataset.level === filter) ? '' : 'none';
  });
}

// ==================== FILTERS & ACTIONS ====================

function setupFilters() {
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.currentFilter = btn.dataset.filter;
      renderTraceList();
    });
  });

  document.getElementById('logLevelFilter').addEventListener('change', (e) => {
    state.logFilter = e.target.value;
    reapplyLogFilter();
  });

  // Pause auto-scroll when user scrolls up
  document.getElementById('logBody').addEventListener('scroll', (e) => {
    const el = e.target;
    state.autoScroll = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  });
}

function setupActions() {
  document.getElementById('clearTracesBtn').addEventListener('click', () => {
    if (!confirm('Clear all stored traces and stats? This cannot be undone.')) return;
    localStorage.removeItem('atlascare_traces');
    localStorage.removeItem('atlascare_stats');
    state.traces = [];
    state.activeTraceId = null;
    renderTraceList();
    renderGlobalStats(null);
    // Reset detail
    document.getElementById('traceDetailBody').innerHTML = `
      <div class="ops-placeholder">
        <svg width="40" height="40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/>
        </svg>
        <p>Click a trace to inspect tool calls, inputs &amp; outputs</p>
      </div>`;
    document.getElementById('traceDetailHint').textContent = 'Select a trace on the left';
  });

  document.getElementById('refreshBtn').addEventListener('click', () => {
    loadTraces();
    showToast('Traces refreshed', 'success');
  });

  document.getElementById('clearLogsBtn').addEventListener('click', () => {
    document.getElementById('logBody').innerHTML = '';
    showToast('Log display cleared (server logs still running)', 'info');
  });

  document.getElementById('scrollBottomBtn').addEventListener('click', () => {
    state.autoScroll = true;
    scrollLogsToBottom();
  });
}

// ==================== UTILS ====================

function truncate(str, len) {
  return str.length <= len ? str : str.slice(0, len) + '…';
}

function formatTs(iso) {
  return new Date(iso).toLocaleString('en-US', {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

function fmtLogTs(iso) {
  try {
    const d = new Date(iso);
    const hh = String(d.getHours()).padStart(2, '0');
    const mm = String(d.getMinutes()).padStart(2, '0');
    const ss = String(d.getSeconds()).padStart(2, '0');
    return `${hh}:${mm}:${ss}`;
  } catch { return ''; }
}

function esc(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 300); }, 2500);
}
