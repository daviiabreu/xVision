import type { VideoResult } from "../../../services/videoService";
import Metric from "./Metric";

interface MetricGridProps {
  result: VideoResult;
  vazio: string;
}

function MetricGrid({ result, vazio }: MetricGridProps) {
  if (result.metrics.length === 0) {
    return <p className="pt-4 text-sm text-gray-500">{vazio}</p>;
  }

  return (
    <div className="grid grid-cols-2 gap-x-12 gap-y-6 pt-4">
      {result.metrics.map((metric, indice) => (
        <Metric key={`${result.name}-${metric.name}-${indice}`} metric={metric} />
      ))}
    </div>
  );
}

export default MetricGrid;
