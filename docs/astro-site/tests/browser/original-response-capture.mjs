import { artifactPath, sha256 } from '../../scripts/browser-contract.mjs';

function purpose(headers) {
  const values = Object.entries(headers ?? {}).filter(([name]) => ['purpose', 'sec-purpose'].includes(name.toLowerCase()));
  return values.map(([name, value]) => [name.toLowerCase(), String(value).trim().toLowerCase()])
    .sort(([left], [right]) => left.localeCompare(right));
}

/** Recognize both the current and legacy browser prefetch request headers. */
export function isSpeculativeRequest(headers) {
  return purpose(headers).some(([, value]) => /(?:^|[\s,;])prefetch(?:$|[\s,;])/u.test(value));
}

function descriptor(request, status) {
  return JSON.stringify([new URL(request.url).href, request.method, status, purpose(request.headers)]);
}

function decodeBody(result) {
  if (typeof result?.body !== 'string' || typeof result.base64Encoded !== 'boolean') {
    throw new Error('Original response capture has no complete protocol body');
  }
  const bytes = Buffer.from(result.body, result.base64Encoded ? 'base64' : 'utf8');
  if (result.base64Encoded && bytes.toString('base64') !== result.body) {
    throw new Error('Original response capture contains invalid base64');
  }
  if (bytes.length === 0) throw new Error('Original speculative response capture is empty');
  return bytes;
}

/**
 * Observe speculative bytes before Chromium's prefetch loader drains them.
 * Install before the first navigation: prefetch clones the active loader bundle.
 * Fetch.getResponseBody completes at body EOF; only then is that same response
 * continued with no status/header/body overrides. This pauses delivery and may
 * affect timing/cache behavior; it is not a performance benchmark or a refetch.
 *
 * Protocol: https://github.com/ChromeDevTools/devtools-protocol/blob/master/pdl/domains/Fetch.pdl
 * Original-body buffering and unchanged continuation:
 * https://github.com/chromium/chromium/blob/159.0.8154.6/content/browser/devtools/devtools_url_loader_interceptor.cc
 * Renderer prefetch bodies cannot establish this provenance:
 * https://github.com/chromium/chromium/blob/159.0.8154.6/content/browser/loader/prefetch_url_loader.cc
 */
export class OriginalResponseCapture {
  constructor(session, { origin, terminalTimeout = 15_000 }) {
    this.session = session;
    this.origin = origin;
    this.terminalTimeout = terminalTimeout;
    this.phase = 'new';
    this.records = [];
    this.issues = [];
    this.network = new Map();
    this.tasks = [];
    this.fetchIds = new Set();
    this.handlers = {
      'Network.requestWillBeSent': (event) => this._network(event.requestId).requests.push(event.request),
      'Network.responseReceived': (event) => this._network(event.requestId).responses.push(event.response),
      'Network.loadingFinished': (event) => this._terminal(event.requestId, { finished: true }),
      'Network.loadingFailed': (event) => this._terminal(event.requestId, { error: event.errorText || 'Network loading failed' }),
      'Fetch.requestPaused': (event) => {
        // Event emitters do not await async callbacks. Every task handles its
        // rejection and remains visible to settle()/close().
        const task = this._paused(event).catch((error) => this._issue(event.request?.url, error));
        this.tasks.push(task);
      },
      close: () => {
        if (this.phase === 'active') {
          this._issue(this.origin, new Error('Capture session closed before verification'));
          for (const state of this.network.values()) {
            if (!state.terminal) this._terminal(state.id, { error: 'Capture session closed' });
          }
        }
      },
    };
  }

  _issue(url, error) {
    this.issues.push({ url: url || this.origin, error: String(error) });
  }

  _network(id) {
    if (!this.network.has(id)) this.network.set(id, { id, requests: [], responses: [], terminal: null, waiters: new Set() });
    return this.network.get(id);
  }

  _terminal(id, terminal) {
    const state = this._network(id);
    if (state.terminal) {
      state.terminal = { error: 'Ambiguous duplicate network terminal events' };
    } else {
      state.terminal = terminal;
    }
    for (const resolve of state.waiters) resolve();
  }

  /** Await installation before page.goto(), including on a new empty page. */
  async start() {
    if (this.phase !== 'new') throw new Error('Capture can only be started once');
    for (const [name, handler] of Object.entries(this.handlers)) this.session.on(name, handler);
    this.phase = 'active';
    try {
      await this.session.send('Network.enable');
      await this.session.send('Fetch.enable', { patterns: [{ urlPattern: `${this.origin}/*`, requestStage: 'Response' }] });
    } catch (error) {
      this.phase = 'closed';
      for (const [name, handler] of Object.entries(this.handlers)) this.session.off(name, handler);
      await this.session.detach();
      throw error;
    }
  }

  async _paused(event) {
    if (typeof event.requestId !== 'string' || !event.requestId.trim()) throw new Error('Missing Fetch request identity');
    if (this.fetchIds.has(event.requestId)) throw new Error(`Duplicate Fetch request ID: ${event.requestId}`);
    this.fetchIds.add(event.requestId);
    const successful = Number.isInteger(event.responseStatusCode) && event.responseStatusCode >= 200 && event.responseStatusCode < 300;
    const speculative = isSpeculativeRequest(event.request?.headers);
    let record;
    try {
      if (this.phase !== 'active') throw new Error('Response arrived after capture shutdown began');
      if (new URL(event.request.url).origin !== this.origin) throw new Error('Capture escaped the internal origin');
      if (!successful || !speculative) return;
      record = {
        fetchId: event.requestId, networkId: event.networkId, request: event.request,
        status: event.responseStatusCode, key: descriptor(event.request, event.responseStatusCode),
        claimed: false, continued: false, bodyComplete: false,
      };
      this.records.push(record);
      if (typeof event.networkId !== 'string' || !event.networkId.trim()) throw new Error('Original capture has no Network request identity');
      if (event.request.method !== 'GET') throw new Error('Unexpected speculative request method');
      // Do not race this command with a local timeout/continuation/disable. A
      // non-returning body read fails at the enclosing Playwright test deadline.
      const result = await this.session.send('Fetch.getResponseBody', { requestId: event.requestId });
      record.bytes = decodeBody(result);
      record.bodyComplete = true;
    } catch (error) {
      if (record) record.error = String(error);
      this._issue(event.request?.url, error);
    } finally {
      // No independent fetch, fulfillment, header override, or body replacement.
      try {
        await this.session.send('Fetch.continueResponse', { requestId: event.requestId });
        if (record) record.continued = true;
      } catch (error) {
        if (record) record.error = String(error);
        this._issue(event.request?.url, error);
      }
    }
  }

  /** Drain protocol callbacks without changing a response still being read. */
  async settle() {
    let count;
    do {
      count = this.tasks.length;
      await Promise.all(this.tasks);
    } while (count !== this.tasks.length);
  }

  async _completed(record) {
    if (record.error) throw new Error(record.error);
    if (!record.bodyComplete || !record.continued) throw new Error('Original capture or unchanged continuation is incomplete');
    const state = this._network(record.networkId);
    if (!state.terminal) {
      await new Promise((resolve, reject) => {
        const ready = () => { clearTimeout(timer); state.waiters.delete(ready); resolve(); };
        const timer = setTimeout(() => {
          state.waiters.delete(ready);
          reject(new Error('Original capture has no successful network completion'));
        }, this.terminalTimeout);
        state.waiters.add(ready);
      });
    }
    if (!state.terminal?.finished || state.terminal.error) throw new Error(state.terminal?.error || 'Original request did not finish');
    const requests = state.requests.filter((request) => request.url === record.request.url && request.method === record.request.method);
    const responses = state.responses.filter((response) => response.url === record.request.url && response.status === record.status);
    if (requests.length !== 1 || responses.length !== 1) throw new Error('Missing or ambiguous original Network request/response identity');
  }

  /**
   * Bridge Playwright to the original protocol IDs only when the descriptor has
   * exactly one candidate. Repeated same-URL requests are never guessed by order.
   */
  async read({ url, method, status, headers }) {
    if (!['active', 'closing'].includes(this.phase)) throw new Error('Capture is not active');
    const matches = this.records.filter((record) => record.key === descriptor({ url, method, headers }, status));
    if (matches.length !== 1) throw new Error('Missing or ambiguous original speculative response capture');
    const record = matches[0];
    if (record.claimed) throw new Error('Original response capture was claimed more than once');
    record.claimed = true;
    await this.settle();
    await this._completed(record);
    return {
      bytes: record.bytes,
      evidence: {
        capture_method: 'cdp-fetch-original-response', fetch_request_id: record.fetchId,
        network_request_id: record.networkId, capture_complete: true,
        continued_unchanged: true, network_loading_finished: true,
      },
    };
  }

  /** Reject unused/incomplete captures and remove listeners after body EOF. */
  close() {
    if (this.closing) return this.closing;
    this.phase = 'closing';
    this.closing = (async () => {
      await this.settle();
      for (const record of this.records) {
        try {
          await this._completed(record);
          if (!record.claimed) throw new Error('Original speculative response capture was never observed by Playwright');
        } catch (error) { this._issue(record.request.url, error); }
      }
      // All getResponseBody commands have settled before disabling interception.
      await this.settle();
      try { await this.session.send('Fetch.disable'); } catch (error) { this._issue(this.origin, error); }
      for (const [name, handler] of Object.entries(this.handlers)) this.session.off(name, handler);
      this.phase = 'closed';
      try { await this.session.detach(); } catch (error) { this._issue(this.origin, error); }
      return this.issues;
    })();
    return this.closing;
  }
}

/** Compare one successful internal response with the unchanged artifact. */
export async function verifiedArtifactResponse(response, capture, artifactHashes) {
  const url = new URL(response.url());
  const status = response.status();
  if (status >= 400) throw new Error(`HTTP ${status}`);
  if (status >= 300) return null;
  const file = artifactPath(url.pathname);
  const request = response.request();
  const headers = await request.allHeaders();
  let bytes;
  let evidence = {};
  if (isSpeculativeRequest(headers)) {
    // Playwright's prefetch body fallback may independently reload the URL.
    // Never call it for speculative provenance, even if it returns nonempty data.
    ({ bytes, evidence } = await capture.read({ url: response.url(), method: request.method(), status, headers }));
  } else {
    bytes = await response.body();
  }
  const digest = sha256(bytes);
  if (!artifactHashes[file]) throw new Error(`Response is not in the qualified artifact: ${file}`);
  if (digest !== artifactHashes[file]) throw new Error(`Response differs from artifact: ${file}`);
  return { url: url.pathname + url.search, file, status, bytes: bytes.length, sha256: digest, ...evidence };
}
