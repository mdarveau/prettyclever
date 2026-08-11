/**
 * Hindsight (clairvoyant) benchmark: how many points does a seed hold for a
 * player who knows every die roll in advance?
 *
 *   npx tsx scripts/hindsight.ts --variant twice-as-clever --games 20 --beam 64
 *
 * Determinization: a per-seed face table F[k][die] (plus a tie-break vector
 * for the passive-turn platter assignment) fixes the outcome of the k-th
 * chance event for every die, independent of the player's actions — the
 * standard "possible world" construction. Rerolls, returns and +1s remain
 * legal: under the table they are deterministic, so the clairvoyant player
 * prices them exactly.
 *
 * The determinized game is a pure deterministic search problem. It is solved
 * with layered beam search (--beam states per decision depth) ordered by the
 * variant's committed TD net value, so the reported optimum is a LOWER bound
 * on the true clairvoyant maximum — already far above what any honest policy
 * can reach, which is what makes it useful as a ceiling instrument.
 *
 * For each seed the same determinized world is also played by --strategy
 * (default td-net, no clairvoyance — it just experiences the table as normal
 * rolls). Efficiency = policy score / hindsight score on the identical world.
 */
import {
  applyActionMut,
  cloneState,
  getPending,
  getVariant,
  mulberry32,
  newGame,
  scoreState,
  type GameState,
  type VariantDef,
} from '../src/engine';
import { makeStrategy } from '../src/strategies';
import {
  createEvalCtx,
  extractFeatures,
  forward,
  netFromParams,
  type EvalCtx,
} from '../src/strategies/tdnet';
import { TDNET_WEIGHTS } from '../src/strategies/tdnet-weights';
import { extractFeaturesTwice, TDT_SCALE } from '../src/strategies/tdnet-twice';
import { TDNET_TWICE_WEIGHTS } from '../src/strategies/tdnet-twice-weights';
import { extractFeaturesBonus } from '../src/strategies/tdnet-bonus';
import { TDNET_BONUS_WEIGHTS } from '../src/strategies/tdnet-bonus-weights';
import { extractFeaturesTwiceBonus } from '../src/strategies/tdnet-twice-bonus';
import { TDNET_TWICE_BONUS_WEIGHTS } from '../src/strategies/tdnet-twice-bonus-weights';

// ---------------------------------------------------------------------------
// Per-variant value heuristic (beam ordering only — correctness-neutral)
// ---------------------------------------------------------------------------

interface Heuristic {
  ctx: EvalCtx;
  extract: (s: GameState, v: VariantDef, x: Float64Array) => void;
  scale: number;
}

function heuristicFor(variantId: string, guide = 'td-net'): Heuristic {
  const bonus = guide === 'td-net-bonus';
  switch (variantId) {
    case 'thats-pretty-clever':
      return bonus
        ? { ctx: createEvalCtx(netFromParams(TDNET_BONUS_WEIGHTS)), extract: extractFeaturesBonus, scale: 300 }
        : { ctx: createEvalCtx(netFromParams(TDNET_WEIGHTS)), extract: extractFeatures, scale: 300 };
    case 'twice-as-clever':
      return {
        ctx: createEvalCtx(netFromParams(bonus ? TDNET_TWICE_BONUS_WEIGHTS : TDNET_TWICE_WEIGHTS)),
        extract: bonus ? extractFeaturesTwiceBonus : extractFeaturesTwice,
        scale: TDT_SCALE,
      };
    default:
      throw new Error(`no committed value net for variant '${variantId}'`);
  }
}

// ---------------------------------------------------------------------------
// Determinized chance: the face table
// ---------------------------------------------------------------------------

class FaceTable {
  private rows: { faces: number[]; tie: number[] }[] = [];
  constructor(
    private rand: () => number,
    private nDice: number,
  ) {}

  row(k: number): { faces: number[]; tie: number[] } {
    while (this.rows.length <= k) {
      const faces: number[] = [];
      const tie: number[] = [];
      for (let i = 0; i < this.nDice; i++) {
        faces.push(1 + Math.floor(this.rand() * 6));
        tie.push(this.rand());
      }
      this.rows.push({ faces, tie });
    }
    return this.rows[k];
  }
}

/** resolveChanceMut with the k-th predetermined row instead of an RNG. */
function resolveChanceTable(s: GameState, t: FaceTable, k: number): void {
  const { faces, tie } = t.row(k);
  if (s.phase === 'roll') {
    for (let i = 0; i < s.loc.length; i++) if (s.loc[i] === 'pool') s.faces[i] = faces[i];
    s.phase = 'pick';
  } else if (s.phase === 'passiveRoll') {
    for (let i = 0; i < s.loc.length; i++) {
      s.loc[i] = 'pool';
      s.faces[i] = faces[i];
    }
    const order = s.faces
      .map((f, i) => ({ f, i, r: tie[i] }))
      .sort((a, b) => a.f - b.f || a.r - b.r);
    for (let j = 0; j < 3; j++) s.loc[order[j].i] = 'platter';
    s.phase = 'passivePick';
  } else {
    throw new Error(`resolveChanceTable in phase ${s.phase}`);
  }
}

// ---------------------------------------------------------------------------
// Beam search over the determinized game
// ---------------------------------------------------------------------------

interface Item {
  s: GameState;
  /** Chance events this line has consumed (its private index into the table). */
  k: number;
  val: number;
}

/** Advance through chance nodes until the next decision or the end of the game. */
function advanceMut(it: Item, v: VariantDef, t: FaceTable): void {
  for (;;) {
    const node = getPending(it.s, v);
    if (node.kind !== 'chance') return;
    resolveChanceTable(it.s, t, it.k);
    it.k++;
  }
}

interface Line {
  score: number;
  areas: Record<string, number>;
  foxPoints: number;
  minArea: number;
}

function lineOf(s: GameState, v: VariantDef): Line {
  const br = scoreState(s, v);
  return { score: br.total, areas: br.areas, foxPoints: br.foxPoints, minArea: br.minArea };
}

function hindsightLine(v: VariantDef, h: Heuristic, table: FaceTable, beam: number): Line {
  const value = (s: GameState): number => {
    if (s.phase === 'over') return scoreState(s, v).total / h.scale;
    h.extract(s, v, h.ctx.x);
    return forward(h.ctx.net, h.ctx.x, h.ctx.h1, h.ctx.h2);
  };

  const root: Item = { s: newGame(v), k: 0, val: 0 };
  advanceMut(root, v, table);
  let frontier: Item[] = [root];
  let best: Line | null = null;

  while (frontier.length > 0) {
    const next: Item[] = [];
    for (const it of frontier) {
      const node = getPending(it.s, v);
      if (node.kind !== 'decision') throw new Error(`beam item in phase ${it.s.phase}`);
      for (const a of node.actions) {
        const ni: Item = { s: cloneState(it.s), k: it.k, val: 0 };
        applyActionMut(ni.s, v, a);
        advanceMut(ni, v, table);
        if (ni.s.phase === 'over') {
          const line = lineOf(ni.s, v);
          if (!best || line.score > best.score) best = line;
        } else {
          ni.val = value(ni.s);
          next.push(ni);
        }
      }
    }
    next.sort((a, b) => b.val - a.val);
    frontier = next.slice(0, beam);
  }
  if (!best) throw new Error('beam found no terminal state');
  return best;
}

/** Play `strategy` (no clairvoyance) through the same determinized world. */
function policyLine(v: VariantDef, strategyName: string, table: FaceTable, seed: number): Line {
  const strategy = makeStrategy(v.id, strategyName);
  const s = newGame(v);
  const ctx = { variant: v, rng: mulberry32(seed) };
  let k = 0;
  for (;;) {
    const node = getPending(s, v);
    if (node.kind === 'over') break;
    if (node.kind === 'chance') {
      resolveChanceTable(s, table, k);
      k++;
      continue;
    }
    applyActionMut(s, v, strategy.choose(s, node.actions, ctx));
  }
  return lineOf(s, v);
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

function parseArgs(argv: string[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (let i = 0; i < argv.length; i++) {
    const m = argv[i].match(/^--([a-z0-9-]+)(?:=(.*))?$/i);
    if (!m) continue;
    out[m[1]] = m[2] ?? (argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[++i] : 'true');
  }
  return out;
}

const args = parseArgs(process.argv.slice(2));
const v = getVariant(args.variant ?? 'thats-pretty-clever');
const games = Number(args.games ?? 20);
const seed0 = Number(args.seed ?? 5000);
const beam = Number(args.beam ?? 64);
const strategyName = args.strategy ?? 'td-net';
const h = heuristicFor(v.id, args.guide ?? 'td-net');

console.log(`hindsight — ${v.id}, ${games} worlds from seed ${seed0}, beam ${beam}, policy ${strategyName}, guide ${args.guide ?? 'td-net'}`);
console.log('seed     policy  hindsight  efficiency');

const meanLine = { policy: null as Line | null, hindsight: null as Line | null };
const addLine = (key: 'policy' | 'hindsight', l: Line) => {
  const acc = meanLine[key];
  if (!acc) {
    meanLine[key] = { ...l, areas: { ...l.areas } };
    return;
  }
  acc.score += l.score;
  acc.foxPoints += l.foxPoints;
  acc.minArea += l.minArea;
  for (const k of Object.keys(l.areas)) acc.areas[k] += l.areas[k];
};
const fmtLine = (l: Line, n: number) =>
  `${(l.score / n).toFixed(1)}  [${Object.entries(l.areas)
    .map(([k, x]) => `${k.slice(0, 2)} ${(x / n).toFixed(1)}`)
    .join(' ')}]  fox ${(l.foxPoints / n).toFixed(1)} (min ${(l.minArea / n).toFixed(1)})`;

let effSum = 0;
const t0 = performance.now();
for (let i = 0; i < games; i++) {
  const seed = seed0 + i;
  const mkTable = () => new FaceTable(mulberry32((seed ^ 0x9e3779b9) >>> 0), v.colors.length);
  const pol = policyLine(v, strategyName, mkTable(), seed);
  // The policy's line is itself a legal clairvoyant line in this world, so it
  // is a witness for the lower bound — a narrow beam can never report below it.
  const beamLine = hindsightLine(v, h, mkTable(), beam);
  const hin = beamLine.score >= pol.score ? beamLine : pol;
  const eff = pol.score / hin.score;
  addLine('policy', pol);
  addLine('hindsight', hin);
  effSum += eff;
  console.log(
    `${String(seed).padEnd(8)} ${String(pol.score).padStart(5)}  ${String(hin.score).padStart(8)}  ${(100 * eff).toFixed(1).padStart(9)}%`,
  );
}
const dt = (performance.now() - t0) / 1000;
console.log(`\nmean policy    ${fmtLine(meanLine.policy!, games)}`);
console.log(`mean hindsight ${fmtLine(meanLine.hindsight!, games)}`);
console.log(
  `mean efficiency ${((100 * effSum) / games).toFixed(1)}%  (${(dt / games).toFixed(1)}s/world)`,
);
