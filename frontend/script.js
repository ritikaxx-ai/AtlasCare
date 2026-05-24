// AtlasCare — Customer Chat
// Handles the chat UI and writes trace data to localStorage for the SRE dashboard (ops.html).

const API_BASE = window.location.origin;

const state = {
  sessionId: `session-${Date.now()}`,
  customerId: null,
  pendingMessage: null,
  // Per-session counters (this customer only, this browser tab)
  session: {
    messageCount: 0,
    totalLatency: 0,
  },
};

// ==================== CUSTOMER ID (from URL param) ====================

function initCustomerId() {
  const params = new URLSearchParams(window.location.search);
  const cid = params.get('cid');
  if (!cid || !/^CUST-\d{3}$/.test(cid)) {
    // No valid ID — bounce back to landing
    window.location.href = '/';
    return;
  }
  state.customerId = cid;
  addCustomerBadgeToHeader(cid);
  showToast(`Logged in as ${cid}`, 'success');
  loadOrders(cid);
}

function addCustomerBadgeToHeader(customerId) {
  const existing = document.getElementById('customerBadge');
  if (existing) existing.remove();

  const badge = document.createElement('div');
  badge.id = 'customerBadge';
  badge.className = 'customer-badge';
  badge.title = 'Change customer — return to home';
  badge.innerHTML = `<span>👤</span> <span class="badge-id">${customerId}</span>`;
  badge.addEventListener('click', () => { window.location.href = '/'; });

  document.querySelector('.header-stats').prepend(badge);
}

// ==================== PER-SESSION STATS (header) ====================

function updateSessionStats() {
  const msgCard = document.getElementById('sessionMsgCard');
  const latencyCard = document.getElementById('sessionLatencyCard');
  const msgCount = document.getElementById('sessionMsgCount');
  const avgLatency = document.getElementById('sessionAvgLatency');

  const count = state.session.messageCount;
  const avg = count > 0 ? Math.round(state.session.totalLatency / count) : 0;

  msgCount.textContent = count;
  avgLatency.textContent = avg > 0 ? `${avg}ms` : '—';

  // Show cards once there's data
  if (count > 0) {
    msgCard.style.display = '';
    latencyCard.style.display = '';
  }
}

// ==================== CONFIRMATION MODAL ====================

const _CONFIRM_KEYWORDS = ['cancel'];

function needsConfirmation(message) {
  const lower = message.toLowerCase();
  return _CONFIRM_KEYWORDS.some(k => lower.includes(k));
}

function showConfirmModal(message) {
  return new Promise((resolve) => {
    const modal = document.getElementById('confirmModal');
    const text = document.getElementById('confirmModalText');
    const okBtn = document.getElementById('confirmOkBtn');
    const cancelBtn = document.getElementById('confirmCancelBtn');

    text.textContent = `You're about to submit: "${message.length > 80 ? message.slice(0, 80) + '…' : message}". This action cannot be undone.`;
    modal.style.display = 'flex';

    const cleanup = (result) => {
      modal.style.display = 'none';
      okBtn.removeEventListener('click', onOk);
      cancelBtn.removeEventListener('click', onCancel);
      resolve(result);
    };

    const onOk = () => cleanup(true);
    const onCancel = () => cleanup(false);

    okBtn.addEventListener('click', onOk);
    cancelBtn.addEventListener('click', onCancel);
  });
}

// ==================== INIT ====================

document.addEventListener('DOMContentLoaded', () => {
  initCustomerId();
  setupEventListeners();
  setupSidebar();
  checkServerConnection();
});

function setupEventListeners() {
  document.getElementById('chatForm').addEventListener('submit', handleChatSubmit);

  // Prompt chips fill input; user sends manually
  document.getElementById('chatWindow').addEventListener('click', (e) => {
    const chip = e.target.closest('.prompt-chip');
    if (!chip) return;
    const input = document.getElementById('messageInput');
    input.value = chip.dataset.prompt;
    input.focus();
    input.selectionStart = input.selectionEnd = input.value.length;
  });
}

function clearChat() {
  document.getElementById('chatWindow').innerHTML = `
    <div class="welcome-message" id="welcomeMessage">
      <div class="welcome-icon">👋</div>
      <h3>Welcome to AtlasCare!</h3>
      <p>I'm your AI-powered customer support agent. Choose a topic or type your question below.</p>
      <div class="prompt-grid">
        <button class="prompt-chip" data-prompt="Where is my order ORD-?">
          <span class="chip-icon">📦</span><span class="chip-label">Track my order</span><span class="chip-sub">Order status &amp; tracking</span>
        </button>
        <button class="prompt-chip" data-prompt="I want to cancel my order ORD-">
          <span class="chip-icon">❌</span><span class="chip-label">Cancel my order</span><span class="chip-sub">Full cancellation</span>
        </button>
        <button class="prompt-chip" data-prompt="Please cancel item 1 from order ORD- and refund to HDFC_CREDIT">
          <span class="chip-icon">🔄</span><span class="chip-label">Cancel item &amp; refund</span><span class="chip-sub">Partial cancellation</span>
        </button>
        <button class="prompt-chip" data-prompt="I received a damaged product on order ORD- and want a refund">
          <span class="chip-icon">🚨</span><span class="chip-label">Damaged product</span><span class="chip-sub">Refund request</span>
        </button>
        <button class="prompt-chip" data-prompt="What was my last inquiry about?">
          <span class="chip-icon">🕐</span><span class="chip-label">Past interactions</span><span class="chip-sub">Support history</span>
        </button>
        <button class="prompt-chip" data-prompt="What is the status of my case CASE-">
          <span class="chip-icon">📋</span><span class="chip-label">Check case status</span><span class="chip-sub">Escalation update</span>
        </button>
        <button class="prompt-chip" data-prompt="What is the return policy and refund window?">
          <span class="chip-icon">📖</span><span class="chip-label">Return policy</span><span class="chip-sub">Policy questions</span>
        </button>
        <button class="prompt-chip" data-prompt="Can you update the shipping address for order ORD- to my office address?">
          <span class="chip-icon">📍</span><span class="chip-label">Update address</span><span class="chip-sub">Shipping update</span>
        </button>
      </div>
    </div>`;
}

// ==================== CHAT ====================

async function handleChatSubmit(e) {
  e.preventDefault();

  const input = document.getElementById('messageInput');
  const message = input.value.trim();
  if (!message) return;

  if (needsConfirmation(message)) {
    const confirmed = await showConfirmModal(message);
    if (!confirmed) {
      showToast('Action cancelled.', 'info');
      return;
    }
  }

  addMessageToChat(message, 'user');

  input.value = '';
  input.disabled = true;
  document.getElementById('sendBtn').disabled = true;

  showLoading();
  const bubbleEl = addMessageToChat('', 'agent', true);

  try {
    await sendQueryStreaming(message, bubbleEl);
    showToast('Request processed successfully', 'success');
  } catch (error) {
    console.error('Error:', error);
    bubbleEl.textContent = 'Sorry, I encountered an error processing your request. Please try again.';
    showToast('Failed to process request: ' + error.message, 'error');
  } finally {
    hideLoading();
    input.disabled = false;
    document.getElementById('sendBtn').disabled = false;
    input.focus();
    bubbleEl.classList.remove('streaming');
  }
}

async function sendQueryStreaming(message, bubbleEl) {
  const response = await fetch(`${API_BASE}/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      session_id: state.sessionId,
      customer_id: state.customerId || null,
    }),
  });

  if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let fullText = '';
  let loadingHidden = false;
  let firstChunkTime = null;
  const STREAM_TIMEOUT_MS = 8000; // if no token in 8s, proxy is buffering — fall back

  while (true) {
    // Apply timeout only while waiting for the first token
    let readPromise = reader.read();
    if (!loadingHidden) {
      const timeoutPromise = new Promise((_, reject) =>
        setTimeout(() => reject(new Error('STREAM_TIMEOUT')), STREAM_TIMEOUT_MS)
      );
      try {
        var { done, value } = await Promise.race([readPromise, timeoutPromise]);
      } catch (err) {
        if (err.message === 'STREAM_TIMEOUT') {
          // Proxy is buffering — fall back to non-streaming
          reader.cancel();
          await sendQueryFallback(message, bubbleEl);
          return;
        }
        throw err;
      }
    } else {
      var { done, value } = await readPromise;
    }

    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split('\n\n');
    buffer = events.pop();

    for (const raw of events) {
      const line = raw.trim();
      if (!line.startsWith('data: ')) continue;

      let evt;
      try { evt = JSON.parse(line.slice(6)); } catch { continue; }

      if (evt.type === 'token') {
        if (!loadingHidden) { hideLoading(); loadingHidden = true; }
        fullText += evt.content;
        bubbleEl.innerHTML = renderMarkdown(fullText) + '<span class="cursor">▋</span>';
        scrollChatToBottom();

      } else if (evt.type === 'done') {
        bubbleEl.innerHTML = renderMarkdown(fullText);
        if (!loadingHidden) hideLoading();
        saveTrace(evt.trace, message);
        maybeRefreshOrders(evt.trace);

      } else if (evt.type === 'error') {
        throw new Error(evt.message);
      }
    }
  }
}

async function sendQueryFallback(message, bubbleEl) {
  // Non-streaming fallback for environments where SSE is buffered (e.g. Render free proxy)
  const response = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      session_id: state.sessionId,
      customer_id: state.customerId || null,
    }),
  });

  if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);

  const data = await response.json();
  hideLoading();
  bubbleEl.innerHTML = renderMarkdown(data.response);
  saveTrace(data.trace, message);
  maybeRefreshOrders(data.trace);
}

// ==================== TRACE (write-only — SRE dashboard reads this) ====================

function saveTrace(trace, message) {
  if (!trace) return;

  const journeyType = classifyJourney(message, trace.tool_calls.length);
  const success = trace.tool_calls.every(tc => tc.success);

  const traceData = {
    id: trace.trace_id,
    sessionId: trace.session_id,
    customerId: state.customerId,
    type: journeyType,
    purpose: message,
    timestamp: new Date().toISOString(),
    latency: trace.latency_ms,
    toolCalls: trace.tool_calls,
    success,
    orderIds: extractOrderIds(message, trace.tool_calls),
    trackingNumbers: extractTrackingNumbers(trace.tool_calls),
    amounts: extractAmounts(trace.tool_calls),
  };

  // Persist for SRE dashboard
  try {
    const existing = JSON.parse(localStorage.getItem('atlascare_traces') || '[]');
    existing.unshift(traceData);
    localStorage.setItem('atlascare_traces', JSON.stringify(existing.slice(0, 200)));

    const stats = JSON.parse(localStorage.getItem('atlascare_stats') || '{"totalRequests":0,"successfulRequests":0,"totalLatency":0}');
    stats.totalRequests++;
    stats.totalLatency += trace.latency_ms;
    if (success) stats.successfulRequests++;
    localStorage.setItem('atlascare_stats', JSON.stringify(stats));
  } catch (err) {
    console.error('saveTrace: localStorage write failed', err);
  }

  // Update per-session header stats (this customer only)
  state.session.messageCount++;
  state.session.totalLatency += trace.latency_ms;
  updateSessionStats();
}

function classifyJourney(message, toolCount) {
  const m = message.toLowerCase();
  if (m.includes('where is') || m.includes('track') || m.includes('status')) return 'J1';
  if (toolCount >= 3 || (m.includes('cancel') && m.includes('refund'))) return 'J2';
  if (m.includes('escalat') || m.includes('case')) return 'J3';
  return 'other';
}

function extractOrderIds(message, toolCalls) {
  const ids = new Set((message.match(/ORD-\d{5}/g) || []));
  toolCalls.forEach(tc => {
    if (tc.input?.order_id) ids.add(tc.input.order_id);
    if (tc.output?.order_id) ids.add(tc.output.order_id);
  });
  return [...ids];
}

function extractTrackingNumbers(toolCalls) {
  const nums = new Set();
  toolCalls.forEach(tc => { if (tc.output?.tracking_number) nums.add(tc.output.tracking_number); });
  return [...nums];
}

function extractAmounts(toolCalls) {
  const amounts = [];
  toolCalls.forEach(tc => {
    if (tc.input?.amount_inr) amounts.push(tc.input.amount_inr);
    if (tc.output?.amount_refunded) amounts.push(tc.output.amount_refunded);
  });
  return amounts;
}

// ==================== UI HELPERS ====================

function addMessageToChat(text, type, streaming = false) {
  const chatWindow = document.getElementById('chatWindow');
  const welcomeMsg = chatWindow.querySelector('.welcome-message');
  if (welcomeMsg) welcomeMsg.remove();

  const div = document.createElement('div');
  div.className = `message ${type}`;

  if (streaming) {
    div.classList.add('streaming');
    div.innerHTML = '<span class="cursor">▋</span>';
  } else {
    div.innerHTML = renderMarkdown(text);
  }

  // column-reverse: prepend = visual bottom
  chatWindow.insertBefore(div, chatWindow.firstChild);
  chatWindow.scrollTop = 0;
  return div;
}

function renderMarkdown(text) {
  return text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');
}

function scrollChatToBottom() {
  document.getElementById('chatWindow').scrollTop = 0;
}

function showLoading() {
  document.getElementById('loadingOverlay').classList.add('active');
}

function hideLoading() {
  document.getElementById('loadingOverlay').classList.remove('active');
}

function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

async function checkServerConnection() {
  try {
    const response = await fetch(`${API_BASE}/health`);
    updateConnectionStatus(response.ok);
  } catch {
    updateConnectionStatus(false);
  }
}

function updateConnectionStatus(isConnected) {
  const dot = document.querySelector('.status-dot');
  const text = document.querySelector('.status-text');
  dot.style.background = isConnected ? 'var(--success)' : 'var(--error)';
  text.textContent = isConnected ? 'Connected' : 'Disconnected';
  if (!isConnected) showToast('Cannot connect to server. Please ensure the server is running.', 'error');
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
    e.preventDefault();
    document.getElementById('messageInput').focus();
  }
});

// Debug
window.atlasCareState = state;

// ==================== LIVE SIDEBAR REFRESH ====================

function maybeRefreshOrders(trace) {
  if (!trace || !state.customerId) return;
  const cancelTools = ['cancel_order_item', 'cancel_full_order'];
  const didCancel = (trace.tool_calls || []).some(
    tc => cancelTools.includes(tc.tool_name) && tc.success
  );
  if (didCancel) loadOrders(state.customerId);
}

// ==================== ORDERS SIDEBAR ====================

async function loadOrders(customerId) {
  try {
    const res = await fetch(`${API_BASE}/customers/${customerId}/orders`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderOrders(data.orders || []);
  } catch (err) {
    console.error('loadOrders:', err);
    document.getElementById('ordersLoading').innerHTML =
      '<span style="color:var(--error);font-size:0.8rem">Could not load orders.</span>';
  }
}

function renderOrders(orders) {
  const list = document.getElementById('ordersList');
  list.innerHTML = '';

  if (!orders.length) {
    list.innerHTML = '<div style="text-align:center;padding:2rem;color:var(--text-muted);font-size:0.82rem">No orders found.</div>';
    return;
  }

  orders.forEach(order => list.appendChild(buildOrderCard(order)));
}

function buildOrderCard(order) {
  const card = document.createElement('div');
  card.className = 'order-card';

  const status = order.status || 'unknown';
  const total = order.total_amount > 0 ? `₹${order.total_amount.toLocaleString('en-IN')}` : '—';
  const payMethod = order.payment_method || '';
  const date = order.created_at ? new Date(order.created_at).toLocaleDateString('en-IN', { day:'numeric', month:'short', year:'numeric' }) : '';
  const activeItems = (order.items || []).filter(i => i.status === 'active');

  // Build action chips based on order status
  const actions = buildActionChips(order);

  card.innerHTML = `
    <div class="order-card-header">
      <button class="order-id-btn" data-order-id="${order.order_id}" title="Click to insert order ID into chat">
        ${order.order_id}
        <span class="copy-icon">📋</span>
      </button>
      <span class="order-status-badge ${status}">${status}</span>
    </div>
    <div class="order-card-body">
      <div class="order-meta-row">
        <span>${date}</span>
        <span class="order-amount">${total} · ${payMethod}</span>
      </div>

      <div class="order-items">
        ${(order.items || []).map(item => `
          <div class="order-item-row">
            <div class="order-item-line">${item.line_id}</div>
            <span class="order-item-name" title="${item.name}">${item.name}</span>
            <span class="order-item-status ${item.status}">${item.status === 'active' ? '✓' : '✗'}</span>
          </div>`).join('')}
      </div>

      ${actions.length ? `
      <div class="order-actions">
        ${actions.map(a => `
          <button class="order-action-chip ${a.danger ? 'danger' : ''}"
                  data-prompt="${escAttr(a.prompt)}"
                  title="${escAttr(a.label)}">
            ${a.label}
          </button>`).join('')}
      </div>` : ''}
    </div>`;

  // Order ID click → insert into input
  card.querySelector('.order-id-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    insertIntoInput(order.order_id);
  });

  // Action chip clicks → set full prompt
  card.querySelectorAll('.order-action-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      setInput(chip.dataset.prompt);
    });
  });

  return card;
}

function buildActionChips(order) {
  const id = order.order_id;
  const pay = order.payment_method || 'HDFC_CREDIT';
  const chips = [];

  chips.push({ label: '📦 Track', prompt: `Where is my order ${id}?` });

  if (order.status === 'shipped' || order.status === 'processing') {
    chips.push({ label: '📍 Update address', prompt: `Can you update the shipping address for order ${id} to my home address?` });
    chips.push({ label: '❌ Cancel', prompt: `I want to cancel my order ${id}`, danger: true });
  }

  if (order.status === 'delivered') {
    chips.push({ label: '🔄 Refund', prompt: `I received a damaged product on order ${id} and want a refund`, danger: true });
    // Add per-item cancel chips for active items
    (order.items || []).filter(i => i.status === 'active').forEach(item => {
      chips.push({
        label: `↩ Return item ${item.line_id}`,
        prompt: `Please cancel item ${item.line_id} from order ${id} and refund to ${pay}`,
        danger: true,
      });
    });
  }

  if (order.status === 'cancelled') {
    chips.push({ label: '🕐 History', prompt: `What is the status of order ${id}?` });
  }

  return chips;
}

function insertIntoInput(text) {
  const input = document.getElementById('messageInput');
  // If there's existing text that ends with a placeholder like "ORD-", replace it
  // Otherwise append / insert the order ID at cursor
  const val = input.value;
  if (val.includes('ORD-') && !val.match(/ORD-\d{5}/)) {
    input.value = val.replace(/ORD-[^\s]*/, text);
  } else {
    input.value = val ? `${val} ${text}` : text;
  }
  input.focus();
  input.selectionStart = input.selectionEnd = input.value.length;
}

function setInput(prompt) {
  const input = document.getElementById('messageInput');
  input.value = prompt;
  input.focus();
  input.selectionStart = input.selectionEnd = input.value.length;
}

function escAttr(str) {
  return str.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// ==================== SIDEBAR COLLAPSE ====================

function setupSidebar() {
  const main = document.querySelector('.chat-main');
  const collapseBtn = document.getElementById('sidebarCollapseBtn');
  const expandTab = document.getElementById('sidebarExpandTab');

  collapseBtn.addEventListener('click', () => main.classList.add('sidebar-collapsed'));
  expandTab.addEventListener('click', () => main.classList.remove('sidebar-collapsed'));
}
