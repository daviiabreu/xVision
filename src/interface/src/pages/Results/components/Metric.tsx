import type { VideoMetric } from "../../../services/videoService";

function texto(metric: VideoMetric): string {
  if (typeof metric.value === "boolean") return metric.value ? "Sim" : "Não";
  if (typeof metric.value === "string") return metric.value;
  if (metric.value === null) return "Indisponível";
  return Number.isInteger(metric.value)
    ? String(metric.value)
    : metric.value.toFixed(2).replace(".", ",");
}

function Metric({ metric }: { metric: VideoMetric }) {
  const indisponivel = metric.status === "unavailable" || metric.value === null;

  return (
    <div className="rounded-xl border border-stone-200 bg-stone-50 p-4">
      <p className="text-xs font-semibold tracking-wide text-stone-500 uppercase">{metric.name}</p>
      <p
        className={`mt-2 text-2xl font-bold ${indisponivel ? "text-stone-500" : "text-stone-950"}`}
      >
        {texto(metric)}
        {!indisponivel && metric.unit && (
          <span className="ml-1 text-sm font-medium text-stone-500">{metric.unit}</span>
        )}
      </p>
      {metric.status === "low_confidence" && !indisponivel && (
        <p className="mt-2 text-xs font-semibold text-amber-700">Baixa confiança</p>
      )}
    </div>
  );
}

export default Metric;
