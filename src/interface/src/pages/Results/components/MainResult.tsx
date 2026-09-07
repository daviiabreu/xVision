import type { VideoResult } from "../../../services/videoService";
import MetricGrid from "./MetricGrid";

function MainResult({ result }: { result: VideoResult }) {
  return (
    <section className="rounded-2xl border border-stone-200 bg-white p-5 shadow-[0_1px_2px_rgba(28,25,23,0.04)] sm:p-6">
      <div className="flex items-center justify-between gap-4">
        <div>
          <p className="text-xs font-bold tracking-wide text-amber-700 uppercase">
            Resultado principal
          </p>
          <h2 className="mt-1 text-xl font-bold text-stone-950">Métricas da finalização</h2>
        </div>
        <span
          className={`rounded-full px-2.5 py-1 text-xs font-semibold ${result.ok ? "bg-emerald-50 text-emerald-700" : "bg-red-50 text-red-700"}`}
        >
          {result.ok ? "Medida" : "Não medida"}
        </span>
      </div>
      <div className="mt-5">
        <MetricGrid
          result={result}
          vazio="A análise rodou, mas não produziu métricas para este vídeo."
        />
      </div>
    </section>
  );
}

export default MainResult;
