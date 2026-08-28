import type { VideoMetric } from "../../../services/videoService";

function texto(metric: VideoMetric): string {
  if (typeof metric.value === "boolean") {
    return metric.value ? "Sim" : "Não";
  }

  if (typeof metric.value === "string") {
    return metric.value;
  }

  if (metric.value === null) {
    return "indisponível";
  }

  return Number.isInteger(metric.value) ? String(metric.value) : metric.value.toFixed(2);
}

function Metric({ metric }: { metric: VideoMetric }) {
  const indisponivel = metric.status === "unavailable" || metric.value === null;

  return (
    <div>
      <p className="text-sm text-gray-500">{metric.name.toUpperCase()}</p>

      <p
        className={`mt-1 text-lg font-semibold ${indisponivel ? "text-gray-500" : "text-gray-900"}`}
      >
        {texto(metric)}

        {!indisponivel && metric.unit && (
          <span className="ml-1 text-sm font-normal text-gray-500">{metric.unit}</span>
        )}
      </p>

      {metric.status === "low_confidence" && !indisponivel && (
        <p className="mt-1 text-xs font-medium text-amber-700">Medida de baixa confiança</p>
      )}
    </div>
  );
}

export default Metric;
