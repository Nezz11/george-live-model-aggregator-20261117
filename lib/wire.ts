import { isRecord, type Json } from './providers.ts';

export type Wire = 'openai' | 'anthropic';

function records(value: unknown): Json[] {
  return Array.isArray(value) ? value.filter(isRecord) : [];
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  return records(content).map((part) => part.type === 'text' && typeof part.text === 'string' ? part.text : '').join('');
}

function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type !== 'image' || !isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  if (source.type === 'base64' && typeof source.data === 'string' && typeof source.media_type === 'string') {
    return { type: 'image_url', image_url: { url: `data:${source.media_type};base64,${source.data}` } };
  }
  return null;
}

function translateMessages(body: Json): Json[] {
  const messages: Json[] = [];
  if (typeof body.system === 'string' || Array.isArray(body.system)) {
    messages.push({ role: 'system', content: textOf(body.system) });
  }
  for (const message of records(body.messages)) {
    const { role, content } = message;
    if (typeof content === 'string') {
      messages.push({ role, content });
      continue;
    }
    if (role === 'assistant') {
      const tools = records(content).filter((block) => block.type === 'tool_use').map((block) => ({
        id: block.id,
        type: 'function',
        function: { name: block.name, arguments: JSON.stringify(block.input) },
      }));
      messages.push({ role, content: textOf(content) || null, ...(tools.length > 0 ? { tool_calls: tools } : {}) });
      continue;
    }
    let parts: Json[] = [];
    for (const block of records(content)) {
      if (block.type === 'tool_result') {
        if (parts.length > 0) messages.push({ role, content: parts });
        parts = [];
        const result = typeof block.content === 'string' ? block.content : records(block.content).map(partOf).filter((part) => part !== null);
        messages.push({ role: 'tool', tool_call_id: block.tool_use_id, content: result });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role, content: parts });
  }
  return messages;
}

export function upstreamRequest(wire: Wire, body: Json): Json {
  if (wire === 'openai') {
    if (body.stream !== true) return body;
    return { ...body, stream_options: { ...(isRecord(body.stream_options) ? body.stream_options : {}), include_usage: true } };
  }
  const request: Json = { messages: translateMessages(body), max_tokens: body.max_tokens };
  for (const key of ['temperature', 'top_p']) {
    if (typeof body[key] === 'number') request[key] = body[key];
  }
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) request.stop = body.stop_sequences;
  const tools = records(body.tools).map((tool) => ({
    type: 'function',
    function: { name: tool.name, description: tool.description, parameters: tool.input_schema },
  }));
  if (tools.length > 0) request.tools = tools;
  if (isRecord(body.tool_choice)) {
    const choice = body.tool_choice;
    if (choice.type === 'auto' || choice.type === 'none') request.tool_choice = choice.type;
    if (choice.type === 'any') request.tool_choice = 'required';
    if (choice.type === 'tool') request.tool_choice = { type: 'function', function: { name: choice.name } };
    if (typeof choice.disable_parallel_tool_use === 'boolean') request.parallel_tool_calls = !choice.disable_parallel_tool_use;
  }
  if (body.stream === true) {
    request.stream = true;
    request.stream_options = { include_usage: true };
  }
  return request;
}

export interface Tool {
  id: string;
  name: string;
  arguments: string[];
}

export interface Choice {
  text: string[];
  tools: Tool[];
  finish: string;
  stopSequence: string | null;
}

export interface Completion {
  id: string;
  choice: Choice;
  usage: { input_tokens: number; output_tokens: number };
}

function stopReason(choice: Choice): string {
  if (choice.finish === 'length') return 'max_tokens';
  if (choice.finish === 'tool_calls' || choice.finish === 'function_call') return 'tool_use';
  if (choice.finish === 'content_filter') return 'refusal';
  return choice.stopSequence === null ? 'end_turn' : 'stop_sequence';
}

function toolInput(tool: Tool): Json {
  const input: unknown = JSON.parse(tool.arguments.join(''));
  if (!isRecord(input)) throw new Error('tool arguments are not a JSON object');
  return input;
}

export function anthropicJson({ id, choice, usage }: Completion, model: string): Json {
  const content: Json[] = [];
  const text = choice.text.join('');
  if (text !== '') content.push({ type: 'text', text });
  for (const tool of choice.tools) {
    content.push({ type: 'tool_use', id: tool.id, name: tool.name, input: toolInput(tool) });
  }
  return {
    id,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReason(choice),
    stop_sequence: choice.stopSequence,
    usage,
  };
}

export function sse(data: unknown, event?: string): string {
  return `${event === undefined ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function namedEvent(type: string, payload: Json = {}): string {
  return sse({ type, ...payload }, type);
}

export function anthropicEvents({ id, choice, usage }: Completion, model: string): string {
  const events = [namedEvent('message_start', {
    message: { id, type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null,
      usage: { input_tokens: usage.input_tokens, output_tokens: 0 } },
  })];
  let index = 0;
  if (choice.text.length > 0) {
    events.push(namedEvent('content_block_start', { index, content_block: { type: 'text', text: '' } }));
    for (const text of choice.text) {
      events.push(namedEvent('content_block_delta', { index, delta: { type: 'text_delta', text } }));
    }
    events.push(namedEvent('content_block_stop', { index }));
    index += 1;
  }
  for (const tool of choice.tools) {
    toolInput(tool);
    events.push(namedEvent('content_block_start', { index, content_block: { type: 'tool_use', id: tool.id, name: tool.name, input: {} } }));
    for (const partial_json of tool.arguments) {
      events.push(namedEvent('content_block_delta', { index, delta: { type: 'input_json_delta', partial_json } }));
    }
    events.push(namedEvent('content_block_stop', { index }));
    index += 1;
  }
  events.push(namedEvent('message_delta', { delta: { stop_reason: stopReason(choice), stop_sequence: choice.stopSequence },
    usage: { output_tokens: usage.output_tokens } }));
  events.push(namedEvent('message_stop'));
  return events.join('');
}
