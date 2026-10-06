import OpenAI, { APIError } from 'openai';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { z } from 'zod';

const apiKey = process.env.GEMINI_API_KEY;
export const llmAvailable = Boolean(apiKey);

const client = apiKey
  ? new OpenAI({
      apiKey,
      baseURL: 'https://generativelanguage.googleapis.com/v1beta/openai/',
      timeout: 45_000,
      maxRetries: 0,
    })
  : null;
const model = process.env.GEMINI_MODEL || 'gemini-3.5-flash-lite';

/** Бесплатный лимит Gemini узкий — не чаще одного запроса в 15 с. */
const MIN_GAP_MS = 15_000;
let nextSlot = 0;
let chain: Promise<void> = Promise.resolve();

function headerRetryAfter(err: APIError): number | undefined {
  const raw = err.headers?.get?.('retry-after') ?? (err.headers as { 'retry-after'?: string } | undefined)?.['retry-after'];
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? Math.min(n * 1000, 60_000) : undefined;
}

async function throttle() {
  const run = chain.then(async () => {
    const wait = nextSlot - Date.now();
    if (wait > 0) await sleep(wait);
    nextSlot = Date.now() + MIN_GAP_MS;
  });
  chain = run.catch(() => undefined);
  await run;
}

export function loadPrompt(name: string): string {
  return readFileSync(join(process.cwd(), 'prompts', `${name}.md`), 'utf8');
}

/** Gemini иногда оборачивает JSON в markdown — оставляем только объект. */
export function extractJson(text: string): string {
  let s = text.trim();
  const fence = /^```(?:json)?\s*([\s\S]*?)\s*```$/i.exec(s);
  if (fence?.[1]) s = fence[1].trim();
  const start = s.indexOf('{');
  const end = s.lastIndexOf('}');
  if (start === -1 || end <= start) throw new Error('В ответе нет JSON-объекта');
  return s.slice(start, end + 1);
}

function friendlyLlmError(err: unknown): Error {
  const raw = err instanceof Error ? err.message : String(err);
  if (/quota|RESOURCE_EXHAUSTED|per day|daily/i.test(raw)) {
    return new Error('Дневной лимит Gemini исчерпан. Подожди несколько часов или возьми другой ключ в AI Studio.');
  }
  if (err instanceof APIError) {
    if (err.status === 503) return new Error('Gemini сейчас перегружен. Подожди минуту и нажми «Повторить».');
    if (err.status === 401 || err.status === 403) return new Error('Ключ Gemini отклонён. Проверь GEMINI_API_KEY в .env.');
    if (err.status === 404) return new Error('Модель Gemini недоступна. Поставь в .env GEMINI_MODEL=gemini-3.5-flash-lite');
    if (err.status === 429) {
      return new Error('Лимит Gemini. Не жми повторно — подожди 1–2 минуты, затем «Повторить».');
    }
  }
  if (/high demand|UNAVAILABLE|503/i.test(raw)) {
    return new Error('Gemini сейчас перегружен. Подожди минуту и нажми «Повторить».');
  }
  return err instanceof Error ? err : new Error(raw);
}

async function sleep(ms: number) {
  await new Promise((r) => setTimeout(r, ms));
}

async function complete(
  messages: OpenAI.ChatCompletionMessageParam[],
  temperature: number,
  useJsonFormat: boolean,
): Promise<string> {
  if (!client) throw new Error('Нет ключа Gemini');
  let jsonMode = useJsonFormat;
  for (let i = 0; i < 5; i++) {
    try {
      await throttle();
      const res = await client.chat.completions.create({
        model,
        messages,
        temperature,
        ...(jsonMode ? { response_format: { type: 'json_object' as const } } : {}),
      });
      return res.choices[0]?.message?.content ?? '';
    } catch (err) {
      const status = err instanceof APIError ? err.status : undefined;
      if (jsonMode && status === 400) {
        jsonMode = false;
        continue;
      }
      if ((status === 429 || status === 503) && i < 3) {
        jsonMode = status === 429 ? jsonMode : false;
        const wait =
          (err instanceof APIError ? headerRetryAfter(err) : undefined) ??
          (status === 429 ? 25_000 * (i + 1) : 3_000 * (i + 1));
        await sleep(wait);
        continue;
      }
      throw friendlyLlmError(err);
    }
  }
  throw new Error('Gemini сейчас перегружен. Подожди минуту и нажми «Повторить».');
}

/** Вызов LLM с system-промптом из prompts/<name>.md и валидацией JSON (1 повтор при невалидном ответе). */
export async function callJson<T>(
  promptName: string,
  input: unknown,
  schema: z.ZodType<T>,
  opts?: { temperature?: number },
): Promise<T> {
  if (!client) throw new Error('Нет ключа Gemini');
  const messages: OpenAI.ChatCompletionMessageParam[] = [
    { role: 'system', content: loadPrompt(promptName) },
    { role: 'user', content: JSON.stringify(input, null, 2) },
  ];
  const temperature = opts?.temperature ?? (promptName === 'qa' ? 0 : 0.8);
  for (let attempt = 0; attempt < 2; attempt++) {
    const text = await complete(messages, temperature, true);
    try {
      return schema.parse(JSON.parse(extractJson(text)));
    } catch (err) {
      messages.push({ role: 'assistant', content: text });
      messages.push({
        role: 'user',
        content: `Ответ невалиден: ${String(err)}. Верни исправленный JSON строго по формату.`,
      });
    }
  }
  throw new Error(`Модель вернула невалидный JSON на шаге ${promptName}`);
}

/** Вызов LLM с кастомным system-промптом (для персонажей и вариаций) */
export async function callJsonWithCustomPrompt<T>(
  systemPrompt: string,
  input: unknown,
  schema: z.ZodType<T>,
  opts?: { temperature?: number },
): Promise<T> {
  if (!client) throw new Error('Нет ключа Gemini');
  const messages: OpenAI.ChatCompletionMessageParam[] = [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: JSON.stringify(input, null, 2) },
  ];
  const temperature = opts?.temperature ?? 0.8;
  for (let attempt = 0; attempt < 2; attempt++) {
    const text = await complete(messages, temperature, true);
    try {
      return schema.parse(JSON.parse(extractJson(text)));
    } catch (err) {
      messages.push({ role: 'assistant', content: text });
      messages.push({
        role: 'user',
        content: `Ответ невалиден: ${String(err)}. Верни исправленный JSON строго по формату.`,
      });
    }
  }
  throw new Error('Модель вернула невалидный JSON');
}
