import type { VideoResult } from "../../../services/videoService";
import Metric from "./Metric";

interface MetricGridProps {
  result: VideoResult;
  vazio: string;
}

function MetricGrid({ result, vazio }: MetricGridProps) {
  if (result.metrics.length === 0) {
    return <p className="rounded-xl bg-stone-50 p-4 text-sm leading-6 text-stone-500">{vazio}</p>;
  }

  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {result.metrics.map((metric, index) => (
        <Metric key={`${result.name}-${metric.name}-${index}`} metric={metric} />
      ))}
    </div>
  );
}

export default MetricGrid;
