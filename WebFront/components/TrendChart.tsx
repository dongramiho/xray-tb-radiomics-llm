"use client";

import { useState } from "react";
import styles from "./TrendChart.module.css";
import { labelKo, RISK_KO } from "./AnalysisCard";
import { formatDate, formatPercent } from "@/utils/format";
import type { RiskLevel, TimelinePoint } from "@/types";

interface TrendChartProps {
  /** Oldest first (as returned by GET /patients/{id}/timeline). */
  points: TimelinePoint[];
}

// Same status colours as the risk badges in AnalysisCard.module.css.
const RISK_COLOR: Record<RiskLevel, string> = { low: "#4ade80", moderate: "#fbbf24", high: "#f87171" };
const LINE_COLOR = "#60a5fa";
const SURFACE = "#18181b";
const GRID = "#27272a";
const TEXT_MUTED = "#71717a";
const TEXT = "#e4e4e7";

const WIDTH = 640;
const HEIGHT = 170;
const PAD = { top: 16, right: 20, bottom: 26, left: 44 };
const PLOT_W = WIDTH - PAD.left - PAD.right;
const PLOT_H = HEIGHT - PAD.top - PAD.bottom;
const MARKER = 6;

function yOf(prob: number): number {
  return PAD.top + PLOT_H * (1 - Math.min(1, Math.max(0, prob)));
}

function xOf(index: number, count: number): number {
  if (count <= 1) return PAD.left + PLOT_W / 2;
  return PAD.left + (PLOT_W * index) / (count - 1);
}

// Marker shape doubles as a secondary encoding so risk is never colour-alone (amber / green sit close under CVD).
function Marker({ x, y, risk, hollow }: { x: number; y: number; risk: RiskLevel; hollow: boolean }) {
  const common = {
    fill: hollow ? SURFACE : RISK_COLOR[risk],
    stroke: hollow ? RISK_COLOR[risk] : SURFACE,
    strokeWidth: 2,
  };
  if (risk === "high") {
    return <polygon points={`${x},${y - MARKER} ${x + MARKER},${y + MARKER * 0.8} ${x - MARKER},${y + MARKER * 0.8}`} {...common} />;
  }
  if (risk === "moderate") {
    return <polygon points={`${x},${y - MARKER} ${x + MARKER},${y} ${x},${y + MARKER} ${x - MARKER},${y}`} {...common} />;
  }
  return <circle cx={x} cy={y} r={MARKER - 1} {...common} />;
}

const LEGEND: RiskLevel[] = ["low", "moderate", "high"];

function describe(point: TimelinePoint): string {
  const prob = point.tb_probability != null ? formatPercent(point.tb_probability) : "확률 없음";
  return `${formatDate(point.date)} · ${labelKo(point.primary_label)} · ${prob} · 위험도 ${RISK_KO[point.risk_level] ?? point.risk_level}`;
}

// Rough text width for the tooltip box (Korean glyphs are about twice as wide as Latin ones at 11px).
function textWidth(text: string): number {
  return Array.from(text).reduce((w, ch) => w + (ch.charCodeAt(0) > 255 ? 11 : 6.5), 0);
}

// Inline SVG line chart of tb_probability over time; dots coloured (and shaped) by risk level.
export default function TrendChart({ points }: TrendChartProps) {
  const [hovered, setHovered] = useState<number | null>(null);

  if (points.length === 0) return <p className={styles.empty}>아직 분석 결과가 없습니다.</p>;

  const count = points.length;
  const plotted = points.map((point, i) => {
    const hasValue = point.tb_probability != null;
    return { point, x: xOf(i, count), y: hasValue ? yOf(point.tb_probability as number) : PAD.top + PLOT_H, hasValue };
  });
  const linePath = plotted
    .filter((p) => p.hasValue)
    .map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`)
    .join(" ");
  const last = [...plotted].reverse().find((p) => p.hasValue) ?? null;
  const xLabelIdx = count <= 2 ? plotted.map((_, i) => i) : [0, Math.floor((count - 1) / 2), count - 1];
  const hoveredPoint = hovered != null ? plotted[hovered] : null;

  let tooltip: { x: number; y: number; w: number; text: string } | null = null;
  if (hoveredPoint) {
    const text = describe(hoveredPoint.point);
    const w = textWidth(text) + 16;
    const x = Math.min(Math.max(hoveredPoint.x - w / 2, PAD.left), WIDTH - PAD.right - w);
    const y = hoveredPoint.y - 34 < 2 ? hoveredPoint.y + 14 : hoveredPoint.y - 34;
    tooltip = { x, y, w, text };
  }

  return (
    <div className={styles.wrapper}>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className={styles.svg}
        role="img"
        aria-label={`결핵 확률 변화 추이, 분석 ${count}건`}
        onMouseLeave={() => setHovered(null)}
      >
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <line x1={PAD.left} x2={WIDTH - PAD.right} y1={yOf(v)} y2={yOf(v)} stroke={GRID} strokeWidth={1} />
            <text x={PAD.left - 8} y={yOf(v)} textAnchor="end" dominantBaseline="middle" fill={TEXT_MUTED} fontSize={10}>
              {Math.round(v * 100)}%
            </text>
          </g>
        ))}

        {xLabelIdx.map((i) => (
          <text
            key={i}
            x={plotted[i].x}
            y={HEIGHT - 8}
            textAnchor={count === 1 ? "middle" : i === 0 ? "start" : i === count - 1 ? "end" : "middle"}
            fill={TEXT_MUTED}
            fontSize={10}
          >
            {formatDate(points[i].date)}
          </text>
        ))}

        {linePath && (
          <path d={linePath} fill="none" stroke={LINE_COLOR} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        )}

        {/* Endpoint label (the only direct label); hidden while a tooltip is showing. */}
        {last && hovered === null && (
          <text
            x={last.x}
            y={last.y - 12}
            textAnchor={last.x > WIDTH - PAD.right - 40 ? "end" : "middle"}
            fill={TEXT}
            fontSize={11}
            fontWeight={600}
          >
            {formatPercent(last.point.tb_probability)}
          </text>
        )}

        {plotted.map((p, i) => (
          <g
            key={`${p.point.diagnosis_id}-${i}`}
            className={styles.markerGroup}
            onMouseEnter={() => setHovered(i)}
            onFocus={() => setHovered(i)}
            onBlur={() => setHovered(null)}
            tabIndex={0}
          >
            <title>{describe(p.point)}</title>
            <Marker x={p.x} y={p.y} risk={p.point.risk_level} hollow={!p.hasValue} />
            {/* Hit target larger than the mark. */}
            <circle cx={p.x} cy={p.y} r={12} fill="transparent" />
          </g>
        ))}

        {tooltip && (
          <g pointerEvents="none">
            <rect x={tooltip.x} y={tooltip.y} width={tooltip.w} height={22} rx={5} fill="#27272a" stroke="#3f3f46" />
            <text x={tooltip.x + 8} y={tooltip.y + 15} fill={TEXT} fontSize={11}>
              {tooltip.text}
            </text>
          </g>
        )}
      </svg>

      <div className={styles.legend} aria-hidden="true">
        {LEGEND.map((risk) => (
          <span key={risk} className={styles.legendItem}>
            <svg width={14} height={14} viewBox="0 0 14 14">
              <Marker x={7} y={7} risk={risk} hollow={false} />
            </svg>
            위험도 {RISK_KO[risk]}
          </span>
        ))}
        <span className={styles.legendItem}>
          <svg width={14} height={14} viewBox="0 0 14 14">
            <Marker x={7} y={7} risk="low" hollow />
          </svg>
          확률 없음
        </span>
      </div>
    </div>
  );
}
