import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { test } from 'node:test';
import { sha256 } from './browser-contract.mjs';
import { OriginalResponseCapture, isSpeculativeRequest, verifiedArtifactResponse } from '../tests/browser/original-response-capture.mjs';

const original = Buffer.from('<!doctype html><h1>Original server response</h1>');
const hashes = { 'index.html': sha256(original) };

function browserResponse({ speculative = true, body = original } = {}) {
  return {
    url: () => 'http://127.0.0.1:4321/innovate/',
    status: () => 200,
    request: () => ({ method: () => 'GET', allHeaders: async () => speculative ? { 'sec-purpose': 'prefetch' } : {} }),
    body: async () => body,
  };
}

test('a speculative response uses captured original bytes despite an empty renderer body', async () => {
  const response = browserResponse({ body: Buffer.alloc(0) });
  const capture = { read: async () => ({ bytes: original, evidence: { capture_complete: true } }) };
  const result = await verifiedArtifactResponse(response, capture, hashes);
  assert.equal(result.sha256, hashes['index.html']);
  assert.equal(result.bytes, original.length);
});

test('ordinary browser response bytes retain the artifact equality check', async () => {
  const result = await verifiedArtifactResponse(browserResponse({ speculative: false }), null, hashes);
  assert.equal(result.sha256, hashes['index.html']);
});

const origin = 'http://127.0.0.1:4321';
const request = { url: `${origin}/innovate/`, method: 'GET', headers: { 'Sec-Purpose': 'prefetch' } };
const observed = { ...request, status: 200 };
const bodyResult = (bytes = original) => ({ body: bytes.toString('base64'), base64Encoded: true });
const deferred = () => {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};

class Session extends EventEmitter {
  constructor() {
    super();
    this.calls = [];
    this.requests = new Map();
    this.body = async () => bodyResult();
    this.continue = async () => {};
    this.detached = false;
  }

  async send(method, params) {
    this.calls.push({ method, params });
    if (method === 'Fetch.getResponseBody') return this.body(params.requestId);
    if (method === 'Fetch.continueResponse') {
      await this.continue(params.requestId);
      const { event, options } = this.requests.get(params.requestId);
      if (!options.noResponse) this.emit('Network.responseReceived', {
        requestId: event.networkId, response: { url: options.responseUrl ?? event.request.url, status: options.responseStatus ?? event.responseStatusCode },
      });
      if (options.failed) this.emit('Network.loadingFailed', { requestId: event.networkId, errorText: 'net::ERR_FAILED' });
      else if (!options.unfinished) this.emit('Network.loadingFinished', { requestId: event.networkId });
      if (options.duplicateFinish) this.emit('Network.loadingFinished', { requestId: event.networkId });
    }
    return {};
  }

  async detach() { this.detached = true; this.emit('close'); }

  paused(options = {}) {
    const event = {
      requestId: options.fetchId ?? 'fetch-1', networkId: options.networkId ?? 'network-1',
      request: options.request ?? request, responseStatusCode: options.status ?? 200,
    };
    if (options.missingNetworkId) delete event.networkId;
    this.requests.set(event.requestId, { event, options });
    if (!options.noRequest) this.emit('Network.requestWillBeSent', { requestId: event.networkId, request: event.request });
    if (options.duplicateRequest) this.emit('Network.requestWillBeSent', { requestId: event.networkId, request: event.request });
    this.emit('Fetch.requestPaused', event);
    return event;
  }
}

async function setup(t, options = {}) {
  const session = new Session();
  const capture = new OriginalResponseCapture(session, { origin, terminalTimeout: 10, ...options });
  await capture.start();
  t.after(() => capture.close());
  return { session, capture };
}

test('installs response interception before use and returns original IDs with complete bytes', async (t) => {
  const { session, capture } = await setup(t);
  assert.deepEqual(session.calls, [
    { method: 'Network.enable', params: undefined },
    { method: 'Fetch.enable', params: { patterns: [{ urlPattern: `${origin}/*`, requestStage: 'Response' }] } },
  ]);
  session.paused();
  const result = await capture.read(observed);
  assert.deepEqual(result.bytes, original);
  assert.deepEqual(result.evidence, {
    capture_method: 'cdp-fetch-original-response', fetch_request_id: 'fetch-1', network_request_id: 'network-1',
    capture_complete: true, continued_unchanged: true, network_loading_finished: true,
  });
  assert.deepEqual(session.calls.find((call) => call.method === 'Fetch.continueResponse'), {
    method: 'Fetch.continueResponse', params: { requestId: 'fetch-1' },
  });
  assert.equal(session.calls.some((call) => /fulfill|loadNetworkResource|replay|takeResponseBody/u.test(call.method)), false);
  assert.deepEqual(await capture.close(), []);
  assert.equal(session.detached, true);
  assert.equal(session.eventNames().length, 0);
  await assert.rejects(capture.read(observed), /not active/u);
});

test('body EOF and unchanged-continuation acknowledgement both precede a successful claim', async (t) => {
  const { session, capture } = await setup(t);
  const body = deferred();
  const continuation = deferred();
  session.body = () => body.promise;
  session.continue = () => continuation.promise;
  session.paused();
  let completed = false;
  const reading = capture.read(observed).then((result) => { completed = true; return result; });
  const closing = capture.close();
  await Promise.resolve();
  assert.equal(session.calls.some((call) => call.method === 'Fetch.continueResponse'), false);
  assert.equal(session.calls.some((call) => call.method === 'Fetch.disable'), false);
  body.resolve(bodyResult());
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(completed, false);
  assert.equal(session.calls.some((call) => call.method === 'Fetch.disable'), false);
  continuation.resolve();
  assert.deepEqual((await reading).bytes, original);
  assert.deepEqual(await closing, []);
  assert.deepEqual(session.calls.slice(-3).map((call) => call.method), ['Fetch.getResponseBody', 'Fetch.continueResponse', 'Fetch.disable']);
});

test('missing and ambiguous captures never fall back to a second fetch or renderer body', async (t) => {
  const { session, capture } = await setup(t);
  await assert.rejects(capture.read(observed), /Missing or ambiguous/u);
  session.paused();
  session.paused({ fetchId: 'fetch-2', networkId: 'network-2' });
  await assert.rejects(capture.read(observed), /Missing or ambiguous/u);
  await capture.settle();
  assert.equal(session.calls.filter((call) => call.method === 'Fetch.getResponseBody').length, 2);
  assert((await capture.close()).some((issue) => /never observed/u.test(issue.error)));
});

test('each capture can be claimed only once', async (t) => {
  const { session, capture } = await setup(t);
  session.paused();
  await capture.read(observed);
  await assert.rejects(capture.read(observed), /more than once/u);
});

test('network identity must be present, unique, and match the actual response', async (t) => {
  for (const options of [
    { missingNetworkId: true }, { noRequest: true }, { duplicateRequest: true },
    { noResponse: true }, { responseUrl: `${origin}/innovate/wrong/` }, { responseStatus: 201 },
  ]) await t.test(JSON.stringify(options), async (subtest) => {
    const { session, capture } = await setup(subtest);
    session.paused(options);
    await assert.rejects(capture.read(observed), /identity/u);
  });
});

test('failed, missing, or duplicated terminal events reject incomplete capture', async (t) => {
  for (const options of [{ failed: true }, { unfinished: true }, { duplicateFinish: true }]) {
    await t.test(JSON.stringify(options), async (subtest) => {
      const { session, capture } = await setup(subtest);
      session.paused(options);
      await assert.rejects(capture.read(observed), /ERR_FAILED|completion|terminal/u);
    });
  }
});

test('a later successful terminal event releases a completed original body', async (t) => {
  const { session, capture } = await setup(t);
  session.paused({ unfinished: true });
  const reading = capture.read(observed);
  await new Promise((resolve) => setImmediate(resolve));
  session.emit('Network.loadingFinished', { requestId: 'network-1' });
  assert.deepEqual((await reading).bytes, original);
  assert.deepEqual(await capture.close(), []);
});

test('a missing Fetch identity cannot produce a captured response', async (t) => {
  const { session, capture } = await setup(t);
  session.paused({ fetchId: '' });
  await capture.settle();
  await assert.rejects(capture.read(observed), /Missing or ambiguous/u);
  assert.equal(session.calls.some((call) => call.method === 'Fetch.getResponseBody'), false);
  assert((await capture.close()).some((issue) => /Missing Fetch request identity/u.test(issue.error)));
});

test('empty, malformed, or absent protocol bodies fail closed', async (t) => {
  for (const result of [{}, { body: 'abc' }, { body: '', base64Encoded: true }, { body: 'bad!', base64Encoded: true }]) {
    await t.test(JSON.stringify(result), async (subtest) => {
      const { session, capture } = await setup(subtest);
      session.body = async () => result;
      session.paused();
      await assert.rejects(capture.read(observed), /body|empty|base64/u);
      assert((await capture.close()).length > 0);
    });
  }
});

test('UTF-8 protocol text is counted as decoded bytes, not characters', async (t) => {
  const { session, capture } = await setup(t);
  session.body = async () => ({ body: '£ café', base64Encoded: false });
  session.paused();
  assert.deepEqual((await capture.read(observed)).bytes, Buffer.from('£ café'));
});

test('protocol and continuation rejections are observed and retained during cleanup', async (t) => {
  for (const stage of ['body', 'continue']) await t.test(stage, async (subtest) => {
    const { session, capture } = await setup(subtest);
    session[stage] = async () => { throw new Error(`Rejected ${stage}`); };
    session.paused();
    await assert.rejects(capture.read(observed), new RegExp(`Rejected ${stage}`, 'u'));
    assert((await capture.close()).some((issue) => issue.error.includes(`Rejected ${stage}`)));
  });
});

test('an early session close cannot qualify a response with no terminal completion', async (t) => {
  const { session, capture } = await setup(t);
  session.paused({ unfinished: true });
  await capture.settle();
  session.emit('close');
  await assert.rejects(capture.read(observed), /session closed/u);
  assert((await capture.close()).some((issue) => /session closed/u.test(issue.error)));
});

test('ordinary responses and redirects continue unchanged without speculative-body capture', async (t) => {
  const { session, capture } = await setup(t);
  session.paused({ request: { ...request, headers: {} } });
  session.paused({ fetchId: 'redirect', networkId: 'redirect', status: 302 });
  await capture.settle();
  assert.equal(session.calls.some((call) => call.method === 'Fetch.getResponseBody'), false);
  assert.deepEqual(session.calls.filter((call) => call.method === 'Fetch.continueResponse').map((call) => call.params), [
    { requestId: 'fetch-1' }, { requestId: 'redirect' },
  ]);
  assert.deepEqual(await capture.close(), []);
});

test('duplicate Fetch IDs and late responses remain failures, with no unhandled callback rejection', async (t) => {
  const { session, capture } = await setup(t);
  const event = session.paused();
  session.emit('Fetch.requestPaused', event);
  await capture.read(observed);
  const closing = capture.close();
  session.paused({ fetchId: 'late', networkId: 'late' });
  const issues = await closing;
  assert(issues.some((issue) => /Duplicate Fetch/u.test(issue.error)));
  assert(issues.some((issue) => /shutdown/u.test(issue.error)));
});

test('setup errors detach the session and do not leave event listeners', async () => {
  const session = new Session();
  session.send = async () => { throw new Error('CDP unavailable'); };
  const capture = new OriginalResponseCapture(session, { origin });
  await assert.rejects(capture.start(), /CDP unavailable/u);
  assert.equal(session.detached, true);
  assert.equal(session.eventNames().length, 0);
});

test('speculative detection recognizes header names and tokens without broad substring matches', () => {
  for (const headers of [{ 'Sec-Purpose': 'prefetch' }, { PURPOSE: 'PREFETCH' }, { 'sec-purpose': 'prefetch;anonymous-client-ip' }]) {
    assert.equal(isSpeculativeRequest(headers), true);
  }
  for (const headers of [{}, { 'sec-purpose': 'not-prefetch' }, { Accept: 'prefetch' }]) assert.equal(isSpeculativeRequest(headers), false);
});

test('prefetch provenance never invokes Playwright body fallback, even when it could return data', async () => {
  const response = browserResponse();
  response.body = async () => { throw new Error('Renderer fallback must not be called'); };
  const capture = { read: async () => ({ bytes: original, evidence: {} }) };
  assert.equal((await verifiedArtifactResponse(response, capture, hashes)).bytes, original.length);
  await assert.rejects(verifiedArtifactResponse(response, { read: async () => { throw new Error('Missing capture'); } }, hashes), /Missing capture/u);
});

test('original captured bytes still reject empty, mismatched, or unknown artifacts', async () => {
  for (const bytes of [Buffer.alloc(0), Buffer.from('different server body')]) {
    const capture = { read: async () => ({ bytes, evidence: {} }) };
    await assert.rejects(verifiedArtifactResponse(browserResponse(), capture, hashes), /differs from artifact/u);
  }
  const capture = { read: async () => ({ bytes: original, evidence: {} }) };
  await assert.rejects(verifiedArtifactResponse(browserResponse(), capture, {}), /not in the qualified artifact/u);
  await assert.rejects(verifiedArtifactResponse(browserResponse({ speculative: false, body: Buffer.from('wrong') }), null, hashes), /differs from artifact/u);
});

test('the original HTTP failure and redirect treatment remains intact', async () => {
  const response = browserResponse();
  response.status = () => 404;
  await assert.rejects(verifiedArtifactResponse(response, null, hashes), /HTTP 404/u);
  response.status = () => 302;
  assert.equal(await verifiedArtifactResponse(response, null, hashes), null);
});
