import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { buildCatalog, capabilitiesOf, freeModels, isRecord, meetsRequirements, readBrief, type Json, type Requirement } from './lib/providers.ts';
import { upstreamRequest, type Wire } from './lib/wire.ts';
import { readResponse, renderResponse } from './lib/upstream.ts';

const MAX_CLIENT_BODY = 8 * 1024 * 1024;
const MAX_ATTEMPTS = 4;
const TRIAL_CALL_LIMIT = 60;
const ATTEMPT_TIMEOUT_MS = 6500;
const envUrl = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const gateway = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
const key = process.env.OPENAI_API_KEY ?? '';
const brief = readBrief(process.env.MAKEFUN_PROMPT_FILE ?? '');
const catalog = buildCatalog(brief);
const byId = new Map(freeModels(brief).map((model) => [model.id, model]));
const models = (catalog.providers[0]?.models ?? []).map(({ id }) => byId.get(id)!);
const ranked = [...models].sort((a, b) => b.rating_milli - a.rating_milli || (a.id < b.id ? -1 : Number(a.id !== b.id)));
// OpenRouter restricts this free model to agentic harnesses. Keep the brief's full catalog while
// routing through the ordinary Chat Completions gateway using the aggregator's own client identity.
const unavailable = new Set<string>(['openrouter/thinkingmachines/inkling-2-small:free']);
const retryAfterPrompt = new Map<string, number>();
let promptNumber = 0;
let calls = 0;

interface Refusal {
  status: number;
  code: string;
  model: string;
}

function completionUrl(): string {
  return gateway.endsWith('/v1') ? `${gateway}/chat/completions` : `${gateway}/v1/chat/completions`;
}

function requirements(body: Json, manifest: Requirement): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const vision = messages.some((message) => isRecord(message) && Array.isArray(message.content)
    && message.content.some((part) => isRecord(part) && (part.type === 'image' || part.type === 'image_url')));
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    ...manifest,
    vision: manifest.vision === true || vision,
    tools: manifest.tools === true || (Array.isArray(body.tools) && body.tools.length > 0)
      || (Array.isArray(body.functions) && body.functions.length > 0),
    json: manifest.json === true || format === 'json_object' || format === 'json_schema',
  };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let bytes = 0;
  for await (const chunk of req.iterator({ destroyOnReturn: false })) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    bytes += buffer.length;
    if (bytes > MAX_CLIENT_BODY) {
      req.resume();
      return null;
    }
    chunks.push(buffer);
  }
  try {
    const body: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(body) ? body : null;
  } catch {
    return null;
  }
}

function reply(res: ServerResponse, status: number, type: string, body: string, refusals: Refusal[] = []): void {
  const headers: Record<string, string | number> = { 'content-type': type, 'content-length': Buffer.byteLength(body) };
  if (type === 'text/event-stream') headers['cache-control'] = 'no-cache';
  if (refusals.length > 0) headers['x-makefun-upstream-failures'] = JSON.stringify(refusals);
  res.writeHead(status, headers);
  res.end(body);
}

function error(res: ServerResponse, wire: Wire, status: number, type: string, message: string, refusals: Refusal[] = []): void {
  const body = wire === 'anthropic' ? { type: 'error', error: { type, message } } : { error: { type, message, code: null } };
  reply(res, status, 'application/json', JSON.stringify(body), refusals);
}

function listModels(req: IncomingMessage, res: ServerResponse): void {
  const body = typeof req.headers['x-api-key'] === 'string'
    ? { data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: '2026-01-01T00:00:00Z' })), has_more: false }
    : { object: 'list', data: models.map((m) => ({ id: m.id, object: 'model', created: 1767225600, owned_by: m.provider })) };
  reply(res, 200, 'application/json', JSON.stringify(body));
}

function refusalCode(raw: string): string {
  try {
    const payload: unknown = JSON.parse(raw);
    if (isRecord(payload) && isRecord(payload.error)) {
      const { code, type } = payload.error;
      if (typeof code === 'string' || typeof code === 'number') return String(code).replace(/[^\x20-\x7e]/g, '').slice(0, 64);
      if (typeof type === 'string') return type.replace(/[^\x20-\x7e]/g, '').slice(0, 64);
    }
  } catch {
    // Preserve the HTTP status when the provider did not send a structured error.
  }
  return 'upstream_error';
}

async function complete(req: IncomingMessage, res: ServerResponse, wire: Wire): Promise<void> {
  const body = await readJson(req);
  if (body === null || !Array.isArray(body.messages)) {
    return error(res, wire, 400, 'invalid_request_error', 'Expected a JSON request with a messages array.');
  }
  const id = promptNumber++;
  const manifest = brief.prompts[id];
  const manifestRequirements = manifest?.api === wire && manifest.stream === (body.stream === true) ? manifest.requires : {};
  const required = requirements(body, manifestRequirements);
  const request = upstreamRequest(wire, body);
  if (request.max_tokens === undefined && request.max_completion_tokens === undefined && wire === 'openai') request.max_tokens = 512;
  const cap = request.max_completion_tokens ?? request.max_tokens;
  if (typeof cap !== 'number' || !Number.isSafeInteger(cap) || cap < 1 || cap > 8192) {
    return error(res, wire, 400, 'invalid_request_error', 'The output cap must be a whole number from 1 to 8192.');
  }
  const eligible = ranked.filter((model) => !unavailable.has(model.id) && meetsRequirements(required, capabilitiesOf(model)));
  const cooling = (model: string) => (retryAfterPrompt.get(model) ?? 0) > id;
  const candidates = [...eligible.filter((m) => !cooling(m.id)), ...eligible.filter((m) => cooling(m.id))];
  if (candidates.length === 0) return error(res, wire, 503, 'overloaded_error', 'No available free model meets the request capabilities.');
  const refusals: Refusal[] = [];
  const headers = {
    'content-type': 'application/json',
    authorization: `Bearer ${key}`,
    accept: body.stream === true ? 'text/event-stream' : 'application/json',
  };
  for (const model of candidates.slice(0, MAX_ATTEMPTS)) {
    if (calls >= TRIAL_CALL_LIMIT || res.destroyed) break;
    calls += 1;
    try {
      const upstream = await fetch(completionUrl(), {
        method: 'POST',
        headers,
        body: JSON.stringify({ ...request, model: model.id }),
        redirect: 'error',
        signal: AbortSignal.timeout(ATTEMPT_TIMEOUT_MS),
      });
      let raw: string;
      try {
        raw = await readResponse(upstream);
      } catch (cause) {
        if (upstream.ok) throw cause;
        raw = '';
      }
      if (!upstream.ok) {
        const code = refusalCode(raw);
        refusals.push({ status: upstream.status, code, model: model.id });
        if (upstream.status === 429) retryAfterPrompt.set(model.id, id + 3);
        if (upstream.status === 403 || upstream.status === 404 || code === 'model_not_allowed') unavailable.add(model.id);
        if ((upstream.status === 401 && code === 'authentication_error') || code === 'request_cap') break;
        continue;
      }
      const answer = renderResponse(raw, model.id, wire, request);
      reply(res, 200, answer.type, answer.body, refusals);
      return;
    } catch {
      // No response bytes have been sent. Try the next candidate.
    }
  }
  error(res, wire, 502, wire === 'anthropic' ? 'api_error' : 'upstream_error', 'The gateway did not return a usable completion.', refusals);
}

async function serve(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const path = new URL(req.url ?? '/', 'http://aggregator').pathname;
  if (req.method === 'GET' && path === '/v1/models') return listModels(req, res);
  if (req.method === 'POST' && path === '/v1/chat/completions') return complete(req, res, 'openai');
  if (req.method === 'POST' && path === '/v1/messages') return complete(req, res, 'anthropic');
  error(res, path === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found_error', 'The requested route does not exist.');
}

async function start(): Promise<void> {
  if (!envUrl || !gateway) throw new Error('MAKEFUN_ENV_URL and OPENAI_BASE_URL are required.');
  if (brief.error !== null) throw new Error(brief.error);
  const server = createServer((req, res) => {
    void serve(req, res).catch(() => {
      if (res.headersSent) res.destroy();
      else error(res, req.url?.startsWith('/v1/messages') ? 'anthropic' : 'openai', 500, 'api_error', 'The server failed to answer.');
    });
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', resolve);
  });
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('The server has no port.');
  const registration = await fetch(`${envUrl}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: 1, port: address.port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!registration.ok) throw new Error(`Registration returned HTTP ${registration.status}.`);
  await registration.arrayBuffer();
  for (;;) {
    try {
      const response = await fetch(`${envUrl}/work`, { signal: AbortSignal.timeout(2000) });
      const state: unknown = await response.json();
      if (response.ok && isRecord(state) && state.done === true) {
        await new Promise<void>((resolve) => server.close(() => resolve()));
        return;
      }
    } catch {
      // The environment can temporarily be unavailable while it drives the trial.
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
}

try {
  if (process.argv.includes('--list')) await new Promise<void>((resolve) => process.stdout.write(`${JSON.stringify(catalog)}\n`, () => resolve()));
  else await start();
} catch (cause) {
  process.stderr.write(`aggregator: ${(cause as Error).message}\n`);
  process.exit(1);
}
