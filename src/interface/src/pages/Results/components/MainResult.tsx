import type { VideoResult } from "../../../services/videoService";
import { nomeDoProcesso } from "../../../utils/nomesDeProcesso";
import MetricGrid from "./MetricGrid";

function MainResult({ result }: { result: VideoResult }) {
  return (
    <section className="w-full text-left">
      <h2 className="text-xl font-semibold text-gray-900">{nomeDoProcesso(result.name)}</h2>

      <div className="mt-5 border-t border-gray-200 pt-6">
        <MetricGrid
          result={result}
          vazio="A análise rodou, mas não produziu nenhuma métrica para este vídeo."
        />
      </div>
    </section>
  );
}

export default MainResult;
