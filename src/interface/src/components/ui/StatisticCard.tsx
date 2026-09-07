export type StatisticTone = "stone" | "amber" | "green" | "red";

const COLORS: Record<StatisticTone, string> = {
  stone: "bg-stone-200",
  amber: "bg-amber-400",
  green: "bg-emerald-500",
  red: "bg-red-400",
};

interface StatisticCardProps {
  label: string;
  value: number | undefined;
  tone?: StatisticTone;
}

export default function StatisticCard({ label, value, tone = "stone" }: StatisticCardProps) {
  return (
    <div className="rounded-2xl border border-stone-200 bg-white p-5 shadow-[0_1px_2px_rgba(28,25,23,0.04)]">
      <div className="flex items-center gap-2 text-sm font-medium text-stone-500">
        <span className={`h-2 w-2 rounded-full ${COLORS[tone]}`} aria-hidden="true" />
        {label}
      </div>
      <p className="mt-3 text-3xl font-bold tracking-tight text-stone-950">{value ?? "..."}</p>
    </div>
  );
}
