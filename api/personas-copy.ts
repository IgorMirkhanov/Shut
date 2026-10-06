import { BriefSchema } from '../src/core/schema.js';
import { VariationSchema, writeAllPersonasCopy } from '../src/core/pipeline.js';
import { fail, guard, readBody } from '../src/core/http.js';
import { z } from 'zod';

export const config = { maxDuration: 300 };

const Body = z.object({
  brief: BriefSchema,
  variation: VariationSchema.optional(),
});

export async function POST(request: Request): Promise<Response> {
  const blocked = guard(request);
  if (blocked) return blocked;
  try {
    const { brief, variation } = Body.parse(await readBody(request));
    const results = await writeAllPersonasCopy(brief, variation);
    return Response.json({ personas: results });
  } catch (err) {
    return fail(err);
  }
}
