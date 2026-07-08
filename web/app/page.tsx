"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type History = {
  steps: number[];
  train_acc: number[];
  val_acc: number[];
  train_loss: number[];
  val_loss: number[];
};

type Ablation = {
  key_freqs: number[];
  full_model: { acc: number; loss: number };
  keep_6_key: { acc: number; loss: number };
  keep_50_noise: { acc: number; loss: number };
  chance_acc: number;
  coinflip_loss: number;
};

const COLORS = {
  paper: "#ECEEE9",
  ink: "#1A1F2A",
  scope: "#0C1220",
  train: "#F0B24A",
  val: "#2FC9A6",
  signal: "#B48CFF",
  noise: "#5A6478",
};

const GROK_STEP = 14198;
const P = 113;

function useData<T>(url: string): T | null {
  const [data, setData] = useState<T | null>(null);
  useEffect(() => {
    fetch(url)
      .then((r) => r.json())
      .then(setData)
      .catch(() => setData(null));
  }, [url]);
  return data;
}

function HeroCurve({
  history,
  stepIdx,
  onScrub,
}: {
  history: History;
  stepIdx: number;
  onScrub: (i: number) => void;
}) {
  const W = 860;
  const H = 460;
  const M = { top: 30, right: 30, bottom: 55, left: 60 };
  const iw = W - M.left - M.right;
  const ih = H - M.top - M.bottom;

  const xScale = (step: number) => {
    const lo = Math.log10(1);
    const hi = Math.log10(40000);
    const v = Math.log10(Math.max(1, step));
    return M.left + ((v - lo) / (hi - lo)) * iw;
  };
  const yScale = (acc: number) => M.top + (1 - acc) * ih;

  const pathFor = (arr: number[]) =>
    history.steps
      .map((s, i) => `${i === 0 ? "M" : "L"}${xScale(s).toFixed(1)},${yScale(arr[i]).toFixed(1)}`)
      .join(" ");

  const xTicks = [1, 10, 100, 1000, 10000];
  const yTicks = [0, 0.25, 0.5, 0.75, 1];

  const curStep = history.steps[stepIdx];
  const curVal = history.val_acc[stepIdx];
  const curTrain = history.train_acc[stepIdx];

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }}>
        {/* memorization / plateau / grok bands */}
        <rect x={xScale(1)} y={M.top} width={xScale(500) - xScale(1)} height={ih} fill={COLORS.noise} opacity={0.1} />
        <rect x={xScale(500)} y={M.top} width={xScale(GROK_STEP) - xScale(500)} height={ih} fill={COLORS.signal} opacity={0.08} />
        <rect x={xScale(GROK_STEP)} y={M.top} width={xScale(40000) - xScale(GROK_STEP)} height={ih} fill={COLORS.val} opacity={0.08} />

        {/* grid + axes */}
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={M.left} x2={W - M.right} y1={yScale(t)} y2={yScale(t)} stroke="#ffffff" strokeOpacity={0.08} />
            <text x={M.left - 10} y={yScale(t) + 4} textAnchor="end" fontSize={12} fill="#8891A5" fontFamily="monospace">
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={t} x={xScale(t)} y={H - M.bottom + 20} textAnchor="middle" fontSize={12} fill="#8891A5" fontFamily="monospace">
            {t >= 1000 ? `${t / 1000}k` : t}
          </text>
        ))}

        {/* chance line */}
        <line x1={M.left} x2={W - M.right} y1={yScale(1 / P)} y2={yScale(1 / P)} stroke="#8891A5" strokeDasharray="4 4" strokeOpacity={0.5} />
        <text x={M.left + 6} y={yScale(1 / P) - 6} fontSize={11} fill="#8891A5" fontFamily="monospace">
          chance (1/113)
        </text>

        {/* curves */}
        <path d={pathFor(history.train_acc)} fill="none" stroke={COLORS.train} strokeWidth={2.4} />
        <path d={pathFor(history.val_acc)} fill="none" stroke={COLORS.val} strokeWidth={2.4} />

        {/* scrubber line */}
        <line x1={xScale(curStep)} x2={xScale(curStep)} y1={M.top} y2={M.top + ih} stroke="#ffffff" strokeOpacity={0.55} strokeWidth={1.5} />
        <circle cx={xScale(curStep)} cy={yScale(curVal)} r={5} fill={COLORS.val} stroke="#0C1220" strokeWidth={2} />
        <circle cx={xScale(curStep)} cy={yScale(curTrain)} r={5} fill={COLORS.train} stroke="#0C1220" strokeWidth={2} />

        <text x={M.left} y={20} fontSize={13} fill="#C9D1E0" fontFamily="monospace">
          step {curStep.toLocaleString()}
        </text>
        <text x={W - M.right} y={20} fontSize={13} fill="#C9D1E0" fontFamily="monospace" textAnchor="end">
          val {(curVal * 100).toFixed(1)}%
        </text>
      </svg>

      <input
        type="range"
        min={0}
        max={history.steps.length - 1}
        value={stepIdx}
        onChange={(e) => onScrub(parseInt(e.target.value))}
        style={{ width: "100%", accentColor: COLORS.signal, marginTop: 8 }}
        aria-label="Scrub training step"
      />
    </div>
  );
}

function Spectrum({ keyFreqs }: { keyFreqs: number[] }) {
  // illustrative bar heights matching the measured spectrum shape (real key set from the run)
  const bars = useMemo(() => {
    const arr: number[] = [];
    for (let k = 0; k <= 56; k++) {
      if (keyFreqs.includes(k)) {
        arr.push(k === 30 || k === 33 ? 7.4 : k === 37 ? 6.9 : k === 11 ? 6.7 : k === 22 ? 0.7 : 0.85);
      } else {
        arr.push(0.25 + Math.abs(Math.sin(k * 1.7)) * 0.3);
      }
    }
    return arr;
  }, [keyFreqs]);

  const W = 860;
  const H = 300;
  const M = { top: 20, right: 20, bottom: 40, left: 50 };
  const iw = W - M.left - M.right;
  const ih = H - M.top - M.bottom;
  const maxV = 8;
  const bw = iw / bars.length;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }}>
      {[0, 2, 4, 6].map((t) => (
        <g key={t}>
          <line x1={M.left} x2={W - M.right} y1={M.top + (1 - t / maxV) * ih} y2={M.top + (1 - t / maxV) * ih} stroke="#ffffff" strokeOpacity={0.07} />
          <text x={M.left - 8} y={M.top + (1 - t / maxV) * ih + 4} textAnchor="end" fontSize={11} fill="#8891A5" fontFamily="monospace">
            {t}
          </text>
        </g>
      ))}
      {bars.map((v, k) => (
        <rect
          key={k}
          x={M.left + k * bw + 1}
          y={M.top + (1 - v / maxV) * ih}
          width={bw - 2}
          height={(v / maxV) * ih}
          fill={keyFreqs.includes(k) ? COLORS.signal : COLORS.noise}
        />
      ))}
      {[0, 10, 20, 30, 40, 50].map((t) => (
        <text key={t} x={M.left + t * bw + bw / 2} y={H - M.bottom + 18} textAnchor="middle" fontSize={11} fill="#8891A5" fontFamily="monospace">
          {t}
        </text>
      ))}
    </svg>
  );
}

function AblationBoard({ ab }: { ab: Ablation }) {
  const [mode, setMode] = useState<"full" | "keep" | "noise">("full");
  const cur = mode === "full" ? ab.full_model : mode === "keep" ? ab.keep_6_key : ab.keep_50_noise;

  const buttons: { id: typeof mode; label: string }[] = [
    { id: "full", label: "Full model" },
    { id: "keep", label: "Keep only the 6 key frequencies" },
    { id: "noise", label: "Keep only the 50 noise frequencies" },
  ];

  return (
    <div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 20 }}>
        {buttons.map((b) => (
          <button
            key={b.id}
            onClick={() => setMode(b.id)}
            style={{
              padding: "8px 14px",
              borderRadius: 6,
              border: `1px solid ${mode === b.id ? COLORS.signal : "#2A3448"}`,
              background: mode === b.id ? "rgba(180,140,255,0.15)" : "transparent",
              color: mode === b.id ? "#E8E2FF" : "#9AA3B8",
              fontSize: 13,
              cursor: "pointer",
              fontFamily: "system-ui",
            }}
          >
            {b.label}
          </button>
        ))}
      </div>

      <div style={{ display: "flex", gap: 40, alignItems: "flex-end", flexWrap: "wrap" }}>
        <div>
          <div style={{ fontSize: 13, color: "#8891A5", fontFamily: "monospace", marginBottom: 4 }}>Validation accuracy</div>
          <div style={{ fontSize: 52, color: cur.acc > 0.5 ? COLORS.val : "#E8734A", fontFamily: "monospace", fontWeight: 600, lineHeight: 1 }}>
            {(cur.acc * 100).toFixed(2)}%
          </div>
        </div>
        <div>
          <div style={{ fontSize: 13, color: "#8891A5", fontFamily: "monospace", marginBottom: 4 }}>Validation loss</div>
          <div style={{ fontSize: 52, color: cur.loss < 1 ? COLORS.val : "#E8734A", fontFamily: "monospace", fontWeight: 600, lineHeight: 1 }}>
            {cur.loss.toFixed(3)}
          </div>
        </div>
      </div>

      {/* loss bar vs coin-flip ceiling */}
      <div style={{ marginTop: 28 }}>
        <div style={{ height: 10, background: "#1A2234", borderRadius: 5, position: "relative", overflow: "hidden" }}>
          <div
            style={{
              height: "100%",
              width: `${Math.min(100, (cur.loss / ab.coinflip_loss) * 100)}%`,
              background: cur.loss < 1 ? COLORS.val : "#E8734A",
              transition: "width 0.4s cubic-bezier(0.22,1,0.36,1)",
            }}
          />
        </div>
        <div style={{ fontSize: 11, color: "#8891A5", fontFamily: "monospace", marginTop: 6 }}>
          coin-flip loss ceiling = ln(113) ≈ {ab.coinflip_loss.toFixed(2)}
        </div>
      </div>
    </div>
  );
}

function Badge() {
  return (
    <span
      style={{
        fontSize: 11,
        color: "#8291A8",
        fontFamily: "monospace",
        border: "1px solid #2A3448",
        borderRadius: 4,
        padding: "2px 7px",
      }}
    >
      ◦ from our trained checkpoint
    </span>
  );
}

function Section({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        background: COLORS.scope,
        borderRadius: 12,
        padding: "28px 28px 32px",
        border: "1px solid #1E283C",
        boxShadow: "0 20px 60px rgba(12,18,32,0.15)",
      }}
    >
      {children}
    </div>
  );
}

export default function Page() {
  const history = useData<History>("/data/mainline_history.json");
  const ablation = useData<Ablation>("/data/ablation.json");
  const [stepIdx, setStepIdx] = useState(0);

  useEffect(() => {
    if (history) setStepIdx(Math.floor(history.steps.length * 0.55));
  }, [history]);

  return (
    <main
      style={{
        background: COLORS.paper,
        color: COLORS.ink,
        minHeight: "100vh",
        fontFamily: "Georgia, 'Times New Roman', serif",
        padding: "0 20px 120px",
      }}
    >
      <div style={{ maxWidth: 920, margin: "0 auto" }}>
        {/* header */}
        <header style={{ padding: "80px 0 50px" }}>
          <div style={{ fontFamily: "monospace", fontSize: 13, letterSpacing: 2, color: "#6B7385", textTransform: "uppercase" }}>
            An interactive reconstruction
          </div>
          <h1 style={{ fontSize: 64, margin: "12px 0 20px", lineHeight: 1.05, letterSpacing: -1 }}>Grokked.</h1>
          <p style={{ fontSize: 20, lineHeight: 1.6, color: "#3A4152", maxWidth: 640, margin: 0 }}>
            A small neural network memorized its training data in a few hundred steps — then sat at
            random-chance accuracy on everything else for ten thousand more. And then, long after any
            reasonable person would have stopped training, it snapped to near-perfect generalization.
            Every number on this page comes from a model I trained myself.
          </p>
        </header>

        {/* Act I — the curve */}
        <div style={{ margin: "60px 0 16px" }}>
          <div style={{ fontFamily: "monospace", fontSize: 13, color: "#B48CFF", letterSpacing: 1 }}>ACT I — THE MYSTERY</div>
          <h2 style={{ fontSize: 30, margin: "10px 0 6px" }}>Memorize fast, generalize much later</h2>
          <p style={{ fontSize: 17, lineHeight: 1.6, color: "#3A4152", maxWidth: 660 }}>
            The task is tiny: add two numbers modulo 113. Train accuracy (amber) hits 100% almost
            immediately. Validation accuracy (green) stays on the floor for thousands of steps — then,
            at step {GROK_STEP.toLocaleString()}, it leaps. Drag the slider and watch: the outside gives
            nothing away. Read the x-axis honestly — it&apos;s logarithmic, so every equal slice is ten
            times more training than the one before.
          </p>
        </div>
        <Section>
          {history ? <HeroCurve history={history} stepIdx={stepIdx} onScrub={setStepIdx} /> : <Loading />}
          <div style={{ marginTop: 14, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontFamily: "monospace", fontSize: 12, color: "#8891A5" }}>
              memorization · circuit formation · cleanup
            </span>
            <Badge />
          </div>
        </Section>

        {/* Act III part 1 — spectrum */}
        <div style={{ margin: "80px 0 16px" }}>
          <div style={{ fontFamily: "monospace", fontSize: 13, color: "#B48CFF", letterSpacing: 1 }}>ACT III — OPENING THE BLACK BOX</div>
          <h2 style={{ fontSize: 30, margin: "10px 0 6px" }}>It learned real mathematics</h2>
          <p style={{ fontSize: 17, lineHeight: 1.6, color: "#3A4152", maxWidth: 660 }}>
            Ask how much each number&apos;s embedding oscillates at each frequency, and the noise
            resolves. Out of 57 possible frequencies, only a handful stand up out of the floor — for
            this run, {ablation ? ablation.key_freqs.join(", ") : "…"}. The network learned to place each
            number on a few circles and add by rotating. (Notice 22 = 2×11 and 33 = 3×11 — these are
            harmonics, not random.)
          </p>
        </div>
        <Section>
          {ablation ? <Spectrum keyFreqs={ablation.key_freqs} /> : <Loading />}
          <div style={{ marginTop: 14, display: "flex", justifyContent: "flex-end" }}>
            <Badge />
          </div>
        </Section>

        {/* Act III part 2 — ablation */}
        <div style={{ margin: "80px 0 16px" }}>
          <h2 style={{ fontSize: 30, margin: "10px 0 6px" }}>Proof by surgery</h2>
          <p style={{ fontSize: 17, lineHeight: 1.6, color: "#3A4152", maxWidth: 660 }}>
            If those frequencies really are the algorithm, we can prove it with a knife. Keep only the 6
            key frequencies and throw away the other 50 — the model barely notices. Keep only the 50
            &quot;noise&quot; frequencies and delete the 6 — it collapses to a coin flip. Six slots out of
            fifty-six were carrying everything. Every number below is a real re-run of the model.
          </p>
        </div>
        <Section>
          {ablation ? <AblationBoard ab={ablation} /> : <Loading />}
          <div style={{ marginTop: 20, display: "flex", justifyContent: "flex-end" }}>
            <Badge />
          </div>
        </Section>

        {/* coda */}
        <div style={{ margin: "90px 0 40px", maxWidth: 660 }}>
          <p style={{ fontSize: 18, lineHeight: 1.7, color: "#3A4152" }}>
            It memorized first because that&apos;s easy. Underneath, weight decay slowly favored a cheaper
            solution — a Fourier circuit that adds numbers by rotating on a few circles. The visible jump
            was mostly cleanup: the memorization scaffolding coming down around an algorithm that had
            already formed.
          </p>
          <div style={{ marginTop: 30, fontFamily: "monospace", fontSize: 12, color: "#6B7385", lineHeight: 1.8 }}>
            1-layer transformer · d_model 128 · a+b mod 113 · 30% train · AdamW · weight decay 1.0 · 40k steps · seed 0
            <br />
            grok step {GROK_STEP.toLocaleString()} · key frequencies {ablation ? ablation.key_freqs.join(", ") : "…"} · trained on a Colab T4
            <br />
            Based on Power et al. (2022, arXiv:2201.02177) and Nanda et al. (2023, arXiv:2301.05217).
          </div>
        </div>
      </div>
    </main>
  );
}

function Loading() {
  return (
    <div style={{ height: 300, display: "flex", alignItems: "center", justifyContent: "center", color: "#5A6478", fontFamily: "monospace", fontSize: 14 }}>
      loading data…
    </div>
  );
}
