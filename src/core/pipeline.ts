import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, extname, join } from 'node:path';
import { callJson, callJsonWithCustomPrompt, llmAvailable } from './llm.js';
import { renderSlides } from './render.js';
import { ruleQa } from './steps/qa.js';
import { z } from 'zod';
import { BriefSchema, CopySchema, type Brief, type Copy, type Qa } from './schema.js';
import { PERSONAS, getPersonaPrompt } from './personas.js';

export const ROOT = join(import.meta.dirname, '../..');
export const OUTPUT_DIR = join(ROOT, 'output');

export const DEFAULT_BRAND: Brief['brand'] = {
  name: process.env.BRAND_NAME || '',
  colors: (process.env.BRAND_COLORS || '#0F172A,#F97316,#6366F1').split(',').map((c) => c.trim()),
  style: process.env.BRAND_STYLE || 'минимализм, мягкий свет',
};

export interface RunOptions {
  /** Пересчитать все шаги, игнорируя кэш. */
  force?: boolean;
  /** Прогонять LLM-редактора и правки. false — только правила (для ручных правок текста). */
  llmQa?: boolean;
  log?: (step: string, msg: string) => void;
}

export interface RunResult {
  name: string;
  dir: string;
  slides: string[];
  caption: string;
  qa: Qa;
}

/**
 * ТЗ → [0] разбор ТЗ → [1] копирайтер → [2] QA (+ до 2 правок) → [3] вёрстка PNG.
 *
 * Каждый шаг сохраняет результат в output/<name>/. Если файл шага уже есть, шаг пропускается —
 * так можно поправить текст в 01_copy.json и перерендерить без повторных вызовов LLM.
 */
export async function runPipeline(name: string, briefText: string, opts: RunOptions = {}): Promise<RunResult> {
  const { force = false, llmQa = true } = opts;
  const log = opts.log ?? ((step, msg) => console.log(`[${step}] ${msg}`));
  const outDir = join(OUTPUT_DIR, name);
  const dirs = { slides: join(outDir, 'slides') };
  Object.values(dirs).forEach((d) => mkdirSync(d, { recursive: true }));

  async function step<T>(file: string, stepName: string, schema: { parse(x: unknown): T }, run: () => Promise<T>): Promise<T> {
    const path = join(outDir, file);
    if (!force && existsSync(path)) {
      log(stepName, `взят из ${file}`);
      return schema.parse(JSON.parse(readFileSync(path, 'utf8')));
    }
    if (!llmAvailable) throw new Error(`[${stepName}] нет ${file} и не задан GEMINI_API_KEY — нечем сгенерировать`);
    log(stepName, 'генерация…');
    const result = await run();
    writeFileSync(path, JSON.stringify(result, null, 2));
    return result;
  }

  // [0] ТЗ: JSON берём как есть, свободный текст разбирает LLM.
  let brief: Brief;
  const asJson = tryJson(briefText);
  if (asJson) {
    brief = BriefSchema.parse(asJson);
    writeFileSync(join(outDir, '00_brief.json'), JSON.stringify(brief, null, 2));
  } else {
    brief = await step('00_brief.json', 'brief', BriefSchema, () =>
      callJson('brief-parser', { DEFAULT_BRAND, brief_text: briefText }, BriefSchema, { temperature: 0.95 }),
    );
  }
  log('brief', `${brief.topic} — ${brief.slides.length} слайдов`);

  // [1] Копирайтер
  let copy: Copy = await step('01_copy.json', 'copywriter', CopySchema, () =>
    callJson('copywriter', brief, CopySchema, { temperature: 1.1 }),
  );

  // [2] QA: жёсткие правила, затем LLM-редактор. До 2 раундов правок.
  let qa: Qa = ruleQa(brief, copy);
  for (let round = 1; llmAvailable && llmQa && round <= 2; round++) {
    if (qa.pass) break;
    log('qa', `раунд ${round}: ${qa.issues.length} замечаний → правка`);
    copy = await callJson('revise', { brief, carousel: copy, issues: qa.issues }, CopySchema);
    writeFileSync(join(outDir, '01_copy.json'), JSON.stringify(copy, null, 2));
    qa = ruleQa(brief, copy);
  }
  writeFileSync(join(outDir, '03_qa.json'), JSON.stringify(qa, null, 2));
  log('qa', qa.pass ? 'OK' : `есть замечания: ${qa.issues.map((i) => `#${i.n} ${i.problem}`).join('; ')}`);

  // [3] Вёрстка
  log('render', 'вёрстка слайдов…');
  const slideFile = (n: number) => join(dirs.slides, `slide_${pad(n)}.png`);
  await renderSlides(brief, copy, slideFile);
  const caption = `${copy.caption}\n\n${copy.hashtags.map((h) => `#${h.replace(/^#/, '')}`).join(' ')}\n`;
  writeFileSync(join(outDir, 'caption.txt'), caption);
  log('done', `${copy.slides.length} слайдов готово`);
  return { name, dir: outDir, slides: copy.slides.map((s) => slideFile(s.n)), caption, qa };
}

/** Имя проекта из имени файла ТЗ. */
export const projectName = (file: string) => basename(file, extname(file));

const pad = (n: number) => String(n).padStart(2, '0');

function tryJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

export const VariationSchema = z.object({
  seed: z.number().int().optional(),
  angle: z.string().min(1).optional(),
  avoid_headlines: z.array(z.string()).optional(),
  avoid_ideas: z.array(z.string()).optional(),
});
export type Variation = z.infer<typeof VariationSchema>;

const briefMemo = new Map<string, Brief>();

export function formatCaption(copy: Copy): string {
  const tags = copy.hashtags.map((h) => `#${h.replace(/^#/, '')}`).join(' ');
  return `${copy.caption}\n\n${tags}\n`;
}

export async function parseBrief(text: string, variation?: Variation): Promise<Brief> {
  const asJson = tryJson(text);
  if (asJson) return BriefSchema.parse(asJson);
  const key = `${text.trim()}::${JSON.stringify(variation ?? {})}`;
  const cached = briefMemo.get(key);
  if (cached) return cached;
  if (!llmAvailable) throw new Error('Нет ключа Gemini и ТЗ не JSON — нечем разобрать');
  const brief = await callJson(
    'brief-parser',
    { DEFAULT_BRAND, brief_text: text, VARIATION: variation ?? null },
    BriefSchema,
    { temperature: 0.95 },
  );
  briefMemo.set(key, brief);
  return brief;
}

function isRateLimit(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err);
  return /Слишком много запросов|перегружен|Лимит Gemini|Дневной лимит/i.test(msg);
}

export async function writeCopy(brief: Brief, variation?: Variation): Promise<{ copy: Copy; qa: Qa }> {
  if (!llmAvailable) throw new Error('Нет ключа Gemini — нечем написать тексты');
  let copy = await callJson(
    'copywriter',
    variation ? { ...brief, VARIATION: variation } : brief,
    CopySchema,
    { temperature: 1.1 },
  );
  let qa = ruleQa(brief, copy);
  if (qa.pass) return { copy, qa };
  try {
    for (let round = 1; round <= 2; round++) {
      copy = await callJson('revise', { brief, carousel: copy, issues: qa.issues }, CopySchema);
      qa = ruleQa(brief, copy);
      if (qa.pass) break;
    }
  } catch (err) {
    if (isRateLimit(err)) return { copy, qa };
    throw err;
  }
  return { copy, qa };
}

export interface PersonaCopyResult {
  personaId: string;
  personaName: string;
  personaEmoji: string;
  copy: Copy;
  qa: Qa;
}

export async function writeAllPersonasCopy(brief: Brief, variation?: Variation): Promise<PersonaCopyResult[]> {
  if (!llmAvailable) throw new Error('Нет ключа Gemini — нечем написать тексты');
  const results: PersonaCopyResult[] = [];

  for (const persona of PERSONAS) {
    const systemPrompt = getPersonaPrompt(persona);
    let copy = await callJsonWithCustomPrompt(
      systemPrompt,
      variation ? { ...brief, VARIATION: variation } : brief,
      CopySchema,
      { temperature: 1.1 },
    );
    let qa = ruleQa(brief, copy);
    if (!qa.pass) {
      try {
        for (let round = 1; round <= 2; round++) {
          copy = await callJsonWithCustomPrompt(
            systemPrompt,
            { brief, carousel: copy, issues: qa.issues },
            CopySchema,
            { temperature: 1.1 },
          );
          qa = ruleQa(brief, copy);
          if (qa.pass) break;
        }
      } catch (err) {
        if (!isRateLimit(err)) throw err;
      }
    }
    results.push({
      personaId: persona.id,
      personaName: persona.name,
      personaEmoji: persona.emoji,
      copy,
      qa,
    });
  }

  return results;
}
