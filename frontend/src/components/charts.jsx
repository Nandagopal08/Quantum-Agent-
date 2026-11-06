import { ResponsiveContainer } from "recharts";

export const PALETTE = ["#4f46e5", "#0ea5e9", "#10b981", "#f59e0b", "#e11d48", "#8b5cf6", "#14b8a6", "#f97316"];

export function ChartBox({ height = 260, children }) {
  return <div style={{ width: "100%", height }}><ResponsiveContainer>{children}</ResponsiveContainer></div>;
}

export const tooltipStyle = { contentStyle: { fontSize: 12, borderRadius: 8, borderColor: "#e2e8f0" } };
