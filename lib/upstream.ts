import { isRecord, type Json } from './providers.ts';
import { anthropicEvents, anthropicJson, sse, type Choice, type Completion, type Tool, type Wire } from './wire.ts';

const STOP_REASONS = new Set(['stop', 'length', 'tool_calls', 'function_call', 'content_filter']);
const RESPONSE_LIMIT = 2 * 1024 * 1024;

export async function readResponse(response: Response): Promise<string> {
  if (response.body === null) return '';
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const pieces: string[] = [];
  let bytes = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes += value.byteLength;
      if (bytes > RESPONSE_LIMIT) {
        await reader.cancel();
        throw new Error('provider response exceeds 2 MiB');
      }
      pieces.push(decoder.decode(value, { stream: true }));
    }
  } finally {
    reader.releaseLock();
  }
  pieces.push(decoder.decode());
  return pieces.join('');
}

interface ChoiceState extends Omit<Choice, 'tools'> {
  tools: Map<number, Tool>;
}

function emptyChoice(): ChoiceState {
  return { text: [], tools: new Map(), finish: '', stopSequence: null };
}

function addTool(choice: ChoiceState, index: number, id: unknown, fn: Json): void {
  const tool = choice.tools.get(index) ?? { id: '', name: '', arguments: [] };
  if (typeof id === 'string' && id !== '') tool.id = id;
  if (typeof fn.name === 'string') tool.name += fn.name;
  if (typeof fn.arguments === 'string' && fn.arguments !== '') tool.arguments.push(fn.arguments);
  choice.tools.set(index, tool);
}

function addMessage(choice: ChoiceState, message: Json, legacyId: string): void {
  if (typeof message.content === 'string' && message.content !== '') choice.text.push(message.content);
  if (Array.isArray(message.content)) {
    for (const part of message.content) {
      if (isRecord(part) && part.type === 'text' && typeof part.text === 'string' && part.text !== '') choice.text.push(part.text);
    }
  }
  if (Array.isArray(message.tool_calls)) {
    for (const [index, call] of message.tool_calls.entries()) {
      if (!isRecord(call) || !isRecord(call.function)) throw new Error('malformed tool call');
      addTool(choice, Number.isInteger(call.index) ? Number(call.index) : index, call.id, call.function);
    }
  }
  if (isRecord(message.function_call)) addTool(choice, 0, legacyId, message.function_call);
}

function finishChoice(choice: ChoiceState, raw: Json): void {
  if (typeof raw.finish_reason === 'string') choice.finish = raw.finish_reason;
  if (typeof raw.stop_sequence === 'string') choice.stopSequence = raw.stop_sequence;
}

function outputChoice(choice: ChoiceState): Choice {
  return { ...choice, tools: [...choice.tools.entries()].sort(([left], [right]) => left - right).map(([, tool]) => tool) };
}

function usageOf(raw: unknown): Completion['usage'] {
  if (!isRecord(raw) || typeof raw.prompt_tokens !== 'number' || typeof raw.completion_tokens !== 'number'
      || !Number.isSafeInteger(raw.prompt_tokens) || !Number.isSafeInteger(raw.completion_tokens)
      || raw.prompt_tokens < 0 || raw.completion_tokens < 1) {
    throw new Error('missing or invalid provider usage');
  }
  return { input_tokens: raw.prompt_tokens, output_tokens: raw.completion_tokens };
}

function expectedEcho(request: Json): string | undefined {
  if (!Array.isArray(request.messages)) return undefined;
  const last = request.messages.at(-1);
  if (!isRecord(last) || last.role !== 'user' || typeof last.content !== 'string') return undefined;
  return last.content.match(/(?:^|\. )Reply with exactly the word ([A-Z]+-\d{4}) and nothing else\.$/)?.[1]
    ?? last.content.match(/^Reply with exactly one JSON object of the form \{"token": "([A-Z]+-\d{4})"\} and nothing else\.$/)?.[1];
}

function validateChoice(choice: Choice, request: Json): void {
  if (!STOP_REASONS.has(choice.finish)) throw new Error('missing or invalid provider finish reason');
  const text = choice.text.join('');
  if (text === '' && choice.tools.length === 0) throw new Error('provider returned no answer content');
  const expected = expectedEcho(request);
  if (expected !== undefined && !text.includes(expected)) throw new Error('provider response omitted the requested token');
  if (isRecord(request.response_format)) {
    const format = request.response_format.type;
    if (format === 'json_object' && !isRecord(JSON.parse(text))) throw new Error('provider response is not a JSON object');
    if (format === 'json_schema') JSON.parse(text);
  }
  const tools = Array.isArray(request.tools) ? request.tools : [];
  const functions = Array.isArray(request.functions) ? request.functions : [];
  const declared = new Set(tools.flatMap((tool) => isRecord(tool) && isRecord(tool.function) ? [tool.function.name] : []));
  for (const fn of functions) if (isRecord(fn)) declared.add(fn.name);
  for (const tool of choice.tools) {
    if (tool.id === '' || tool.name === '' || !declared.has(tool.name)) throw new Error('invalid or undeclared provider tool call');
  }
}

function renderJson(raw: unknown, model: string, wire: Wire, request: Json): Json {
  if (!isRecord(raw) || raw.error !== undefined || typeof raw.id !== 'string' || raw.id === ''
      || !Array.isArray(raw.choices) || raw.choices.length === 0) throw new Error('malformed provider completion');
  const usage = usageOf(raw.usage);
  const choices: Choice[] = [];
  for (const item of raw.choices) {
    if (!isRecord(item) || !isRecord(item.message) || item.message.role !== 'assistant') throw new Error('missing assistant completion');
    const state = emptyChoice();
    addMessage(state, item.message, raw.id);
    finishChoice(state, item);
    const choice = outputChoice(state);
    validateChoice(choice, request);
    choices.push(choice);
  }
  return wire === 'openai' ? { ...raw, model } : anthropicJson({ id: raw.id, choice: choices[0], usage }, model);
}

function ssePayloads(raw: string): string[] {
  return raw.replace(/\r\n?/g, '\n').split('\n\n').flatMap((frame) => {
    const fields = frame.split('\n').filter((line) => line.startsWith('data:') || line === 'data');
    if (fields.length === 0) return [];
    return [fields.map((line) => line.slice(5).replace(/^ /, '')).join('\n').trim()];
  });
}

function finishIsRepeated(raw: Json, delta: Json, choice: ChoiceState, usage: unknown): boolean {
  if (choice.finish === '' || !isRecord(usage) || raw.finish_reason !== choice.finish) return false;
  return Object.entries(delta).every(([field, value]) =>
    (field === 'role' && value === 'assistant') || value === '' || value === null || (Array.isArray(value) && value.length === 0));
}

function renderStream(raw: string, model: string, wire: Wire, request: Json): string {
  const payloads = ssePayloads(raw);
  if (payloads.pop() !== '[DONE]') throw new Error('unterminated provider stream');
  const chunks: Json[] = [];
  const states = new Map<number, ChoiceState>();
  let id = '';
  let usage: unknown;
  for (const payload of payloads) {
    const chunk: unknown = JSON.parse(payload);
    if (!isRecord(chunk) || chunk.error !== undefined || !Array.isArray(chunk.choices)) throw new Error('malformed provider stream chunk');
    if (id === '' && typeof chunk.id === 'string') id = chunk.id;
    if (isRecord(chunk.usage)) usage = chunk.usage;
    const choices: Json[] = [];
    for (const [position, rawChoice] of chunk.choices.entries()) {
      if (!isRecord(rawChoice)) throw new Error('malformed provider stream choice');
      const index = Number.isInteger(rawChoice.index) ? Number(rawChoice.index) : position;
      const state = states.get(index) ?? emptyChoice();
      const delta = isRecord(rawChoice.delta) ? rawChoice.delta : {};
      // OpenRouter can repeat the finish choice on the final usage frame.
      if (finishIsRepeated(rawChoice, delta, state, chunk.usage)) continue;
      if (state.finish !== '') throw new Error('provider emitted another choice after its finish');
      addMessage(state, delta, id);
      finishChoice(state, rawChoice);
      states.set(index, state);
      choices.push(rawChoice);
    }
    chunks.push({ ...chunk, choices, model });
  }
  if (id === '' || states.size === 0) throw new Error('provider stream held no completion');
  const choices = [...states.values()].map(outputChoice);
  for (const choice of choices) validateChoice(choice, request);
  const checkedUsage = usageOf(usage);
  if (wire === 'anthropic') return anthropicEvents({ id, choice: choices[0], usage: checkedUsage }, model);
  return chunks.map((chunk) => sse(chunk)).join('') + 'data: [DONE]\n\n';
}

export function renderResponse(raw: string, model: string, wire: Wire, request: Json): { type: string; body: string } {
  const stream = request.stream === true;
  const body = stream ? renderStream(raw, model, wire, request) : JSON.stringify(renderJson(JSON.parse(raw), model, wire, request));
  if (Buffer.byteLength(body) > RESPONSE_LIMIT) throw new Error('translated response exceeds 2 MiB');
  return { type: stream ? 'text/event-stream' : 'application/json', body };
}
