/**
 * repair — Tool-call repair pipeline for DeepSeek.
 *
 * Harvested from reasonix Pillar 2 (Tool-Call Repair).
 *
 * DeepSeek has known failure modes:
 * 1. Tool-call JSON emitted inside `<think>` blocks, missing from tool_calls
 * 2. Arguments dropped when schema has >10 params or deeply nested objects
 * 3. Same tool called repeatedly with identical args (call-storm)
 * 4. Truncated JSON due to max_tokens hit mid-structure
 *
 * This module provides four passes to address each one.
 */

import type { ToolCallRepairInput } from "./types.js";

/**
 * Port of original closeTruncatedJSON (internal/provider/provider.go:524-574).
 * String-aware stack, tracks inStr/escape, closes unterminated strings,
 * trims trailing ',' / ':' with null fill, '{}' fallback.
 */
function closeTruncatedJSON(s: string): string {
  const stack: string[] = [];
  let inStr = false;
  let esc = false;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (inStr) {
      if (esc) {
        esc = false;
      } else if (c === "\\") {
        esc = true;
      } else if (c === '"') {
        inStr = false;
      }
      continue;
    }
    if (c === '"') {
      inStr = true;
    } else if (c === "{") {
      stack.push("}");
    } else if (c === "[") {
      stack.push("]");
    } else if (c === "}" || c === "]") {
      if (stack.length > 0) stack.pop();
    }
  }
  let out = s;
  if (esc) out = out.slice(0, -1);
  if (inStr) out += '"';
  const trimmed = out.trimEnd();
  if (trimmed.endsWith(",")) {
    out = trimmed.slice(0, -1);
  } else if (trimmed.endsWith(":")) {
    out = trimmed + "null";
  }
  for (let i = stack.length - 1; i >= 0; i--) out += stack[i];
  try {
    JSON.parse(out);
    return out;
  } catch {
    return "{}";
  }
}

/**
 * Estimate whether tool-call arguments JSON is truncated.
 */
export function isTruncatedJSON(text: string): boolean {
  if (!text) return false;
  try {
    JSON.parse(text);
    return false;
  } catch {
    // Use string-aware closer to determine if repairable truncation
    const repaired = closeTruncatedJSON(text);
    if (repaired === text) return false;
    try {
      JSON.parse(repaired);
      return true;
    } catch {
      return false;
    }
  }
}

/**
 * Attempt to repair truncated JSON by closing open braces/brackets.
 * Handles unterminated strings, dangling comma/colon, fallback to {}.
 */
export function repairTruncatedJSON(text: string): {
  repaired: string;
  fixed: boolean;
} {
  if (!text) return { repaired: text, fixed: false };
  try {
    JSON.parse(text);
    return { repaired: text, fixed: false };
  } catch {
    // Try robust closer
  }
  const repaired = closeTruncatedJSON(text);
  if (repaired === text) return { repaired: text, fixed: false };
  try {
    JSON.parse(repaired);
    return { repaired, fixed: true };
  } catch {
    return { repaired: text, fixed: false };
  }
}

/* ------------------------------------------------------------------ */
/*  Pass 2: scavenge — find tool calls in reasoning / text            */
/* ------------------------------------------------------------------ */

/**
 * Options for scavengeToolCalls.
 */
export interface ScavengeOptions {
  /**
   * Tool names the session actually exposes (e.g. the tools list seen at
   * `before_provider_request`). When provided, only calls whose name is
   * present are kept. When omitted or empty, extraction keeps every
   * structurally valid call — strict JSON validation still applies.
   */
  knownTools?: Iterable<string> | null | undefined;
}

/**
 * Scavenge tool calls that DeepSeek "leaked" into reasoning_content or
 * message content (outside the tool_calls array).
 *
 * A candidate is only emitted when it is structurally valid: a string
 * `name` and `arguments` that parse as valid JSON (object arguments are
 * serialized). Malformed candidates are dropped, never emitted.
 */
export function scavengeToolCalls(
  content: string | null | undefined,
  options: ScavengeOptions = {},
): ToolCallRepairInput[] {
  if (!content) return [];

  const known = options.knownTools ? new Set(options.knownTools) : null;

  // Pattern 1: tool_use blocks inside <think> tags.
  // DeepSeek often emits tool call JSON inside reasoning blocks.
  const thinkContents: string[] = [];
  for (const m of content.matchAll(/<think>([\s\S]*?)<\/think>/g)) {
    thinkContents.push(m[1]);
  }

  // Pattern 2: tool calls embedded in markdown code fences.
  const fenceContents: string[] = [];
  for (const m of content.matchAll(/```(?:json)?\s*\n?([\s\S]*?)```/g)) {
    fenceContents.push(m[1]);
  }

  // Pattern 3: bare tool calls inline in the response text — only when no
  // <think> block matched, mirroring the original pass.
  const regions: string[] = [...thinkContents, ...fenceContents];
  if (thinkContents.length === 0) regions.push(content);

  const found: ToolCallRepairInput[] = [];
  const seen = new Set<string>();
  for (const region of regions) {
    for (const obj of extractJsonObjects(region)) {
      const call = normalizeScavengedCall(obj);
      if (!call) continue;
      if (known && !known.has(call.function.name)) continue;
      const key = callKey(call.function.name, call.function.arguments);
      if (seen.has(key)) continue;
      seen.add(key);
      found.push({
        ...call,
        id: call.id || `scavenged-${found.length}`,
      });
    }
  }

  return found;
}

/**
 * Deterministic key for a (name, serialized arguments) tool-call tuple.
 * Shared by scavenge de-duplication and storm detection.
 */
export function callKey(name: string, argumentsStr: string): string {
  return `${name}|${argumentsStr}`;
}

/**
 * Find every complete JSON object in `text` that looks like a tool call
 * (`{"function": {...}}` or `{"name": ..., "arguments": ...}`), returned as
 * the parsed object. Objects are matched by balanced braces (string-aware),
 * so nested-argument calls and string-form arguments are handled — unlike
 * the old non-greedy regex which cut on the first `}`.
 */
function extractJsonObjects(text: string): unknown[] {
  const objects: unknown[] = [];
  let pos = 0;
  while (pos < text.length) {
    const start = text.indexOf("{", pos);
    if (start === -1) break;
    const end = findJsonObjectEnd(text, start);
    if (end === -1) break; // unbalanced — no complete object left
    const raw = text.slice(start, end + 1);
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      pos = end + 1; // malformed object — drop and move on
      continue;
    }
    if (isToolCallShape(parsed)) {
      objects.push(parsed);
      pos = end + 1; // skip the matched object's interior so nested
      // bare-form look-alikes are not scavenged twice
    } else {
      // A non-tool object may still wrap a tool call — scan its interior.
      objects.push(...extractJsonObjects(raw.slice(1, -1)));
      pos = end + 1;
    }
  }
  return objects;
}

/**
 * Index of the closing brace for the object that opens at `start`,
 * respecting string literals and escapes. -1 when unbalanced.
 */
function findJsonObjectEnd(text: string, start: number): number {
  let depth = 0;
  let inStr = false;
  let esc = false;
  for (let i = start; i < text.length; i++) {
    const c = text[i];
    if (inStr) {
      if (esc) {
        esc = false;
      } else if (c === "\\") {
        esc = true;
      } else if (c === '"') {
        inStr = false;
      }
      continue;
    }
    if (c === '"') {
      inStr = true;
    } else if (c === "{") {
      depth++;
    } else if (c === "}") {
      depth--;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** True when the parsed object carries a tool-call shape. */
function isToolCallShape(obj: unknown): boolean {
  if (!obj || typeof obj !== "object" || Array.isArray(obj)) return false;
  const o = obj as Record<string, unknown>;
  const f = (o.function as Record<string, unknown> | undefined) ?? o;
  return (
    typeof f?.name === "string" &&
    f.name.length > 0 &&
    "arguments" in f
  );
}

/**
 * Convert a parsed tool-call object into a repair input, or null when it is
 * malformed: non-string/empty name, or arguments that are neither a valid
 * JSON string nor an object.
 */
function normalizeScavengedCall(obj: unknown): ToolCallRepairInput | null {
  if (!obj || typeof obj !== "object" || Array.isArray(obj)) return null;
  const o = obj as Record<string, unknown>;
  const f = (o.function as Record<string, unknown> | undefined) ?? o;
  const name = f?.name;
  if (typeof name !== "string" || name.length === 0) return null;
  const args = f?.arguments;
  let argumentsStr: string;
  if (typeof args === "string") {
    try {
      JSON.parse(args); // must parse as valid JSON
    } catch {
      return null;
    }
    argumentsStr = args;
  } else if (args !== null && typeof args === "object") {
    argumentsStr = JSON.stringify(args);
  } else {
    return null;
  }
  const id = typeof o.id === "string" && o.id.length > 0 ? o.id : "";
  return {
    id,
    type: "function",
    function: { name, arguments: argumentsStr },
  };
}

/* ------------------------------------------------------------------ */
/*  Pass 3: storm — detect and break identical call repeats            */
/* ------------------------------------------------------------------ */

/**
 * How many times the same call repeat may be suppressed before a
 * `[loop guard]` notice is appended to the message so the suppression is
 * visible to the model instead of silent (mirrors the original's
 * stormBreakThreshold = 3, two natural self-corrections then escalate).
 */
export const STORM_BREAK_THRESHOLD = 3;

export interface StormResult {
  clean: ToolCallRepairInput[];
  stormCount: number;
  /** Keys (name|args) of the calls that were suppressed. */
  suppressedKeys: string[];
}

/**
 * Detect call-storms: same (tool, args) tuple within a sliding window.
 * Returns the calls to keep, the number of suppressed calls, and the keys
 * of the suppressed calls.
 *
 * Keying is exact (name, serialized arguments). A (name, errorClass)
 * dimension is not added because pi does not expose tool-execution error
 * class at the available hook: `message_end` assistant messages carry only
 * `name` + `arguments` on toolCall blocks, and tool results arrive later as
 * separate `toolResult` messages with just a boolean `isError`. Re-worded-
 * argument loops on a failing tool are therefore not catchable here.
 */
export function detectCallStorm(
  calls: ToolCallRepairInput[],
  windowSize = 5,
): StormResult {
  if (calls.length < 2) return { clean: calls, stormCount: 0, suppressedKeys: [] };

  const clean: ToolCallRepairInput[] = [calls[0]];
  const suppressedKeys: string[] = [];
  let stormCount = 0;

  for (let i = 1; i < calls.length; i++) {
    const prev = calls.slice(Math.max(0, i - windowSize), i);
    const isDuplicate = prev.some(
      (p) =>
        p.function.name === calls[i].function.name &&
        p.function.arguments === calls[i].function.arguments,
    );

    if (isDuplicate) {
      stormCount++;
      suppressedKeys.push(
        callKey(calls[i].function.name, calls[i].function.arguments),
      );
      // Suppress this call — don't add to clean
    } else {
      clean.push(calls[i]);
    }
  }

  return { clean, stormCount, suppressedKeys };
}

/**
 * Track per-key suppression counts and report which keys crossed the
 * loop-guard threshold on this pass. `counts` is mutated and persists across
 * calls so a death-spiral spanning several messages is still caught.
 */
export function escalateLoopGuards(
  suppressedKeys: string[],
  counts: Map<string, number>,
  threshold = STORM_BREAK_THRESHOLD,
): string[] {
  const flagged: string[] = [];
  for (const key of suppressedKeys) {
    const next = (counts.get(key) ?? 0) + 1;
    counts.set(key, next);
    if (next === threshold) flagged.push(key);
  }
  return flagged;
}

/* ------------------------------------------------------------------ */
/*  Pipeline                                                            */
/* ------------------------------------------------------------------ */

export interface RepairOptions {
  /** Max params before schema flattening triggers (default: 10). */
  maxParams?: number;
  /** Max nesting depth before flattening triggers (default: 2). */
  maxDepth?: number;
  /** Sliding window for call-storm detection (default: 5). */
  stormWindow?: number;
}

export interface RepairResult {
  repaired: ToolCallRepairInput[];
  scavenged: ToolCallRepairInput[];
  stormCount: number;
  truncatedFixed: number;
}

/**
 * Full repair pipeline for a set of tool calls.
 *
 * 1. Scavenge — recover tool calls leaked into reasoning_content
 * 2. Repair truncated JSON in arguments
 * 3. Detect and break call-storms
 */
export function repairToolCalls(
  calls: ToolCallRepairInput[],
  reasoningContent?: string | null,
  options: RepairOptions = {},
): RepairResult {
  const { stormWindow = 5 } = options;

  const scavenged: ToolCallRepairInput[] = reasoningContent
    ? scavengeToolCalls(reasoningContent)
    : [];

  const allCalls = [...calls, ...scavenged];

  // Repair truncated JSON in arguments
  let truncatedFixed = 0;
  for (const call of allCalls) {
    const { repaired, fixed } = repairTruncatedJSON(call.function.arguments);
    if (fixed) {
      call.function.arguments = repaired;
      truncatedFixed++;
    }
  }

  // Detect call-storms
  const { clean, stormCount } = detectCallStorm(allCalls, stormWindow);

  return {
    repaired: clean,
    scavenged,
    stormCount,
    truncatedFixed,
  };
}
