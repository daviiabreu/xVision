import Icon from "../../../components/ui/Icon";
import type { VideoResult } from "../../../services/videoService";
import { nomeDoProcesso } from "../../../utils/nomesDeProcesso";
import MetricGrid from "./MetricGrid";

interface ResultAccordionProps {
  result: VideoResult;
  principalMediu: boolean;
}

function fonteDoFrame(frame: string): string {
  return frame.startsWith("data:") ? frame : `data:image/png;base64,${frame}`;
}

function semMetrica(result: VideoResult, principalMediu: boolean): string {
  if (result.state === "skipped") return "Esta etapa foi ignorada, então não há nada para mostrar.";
  if (!result.ok) return "Esta etapa não concluiu, então não há nada para mostrar.";
  return principalMediu
    ? "Etapa concluída. As métricas consolidadas aparecem em Métricas da finalização."
    : "Etapa concluída. Esta análise não produziu nenhuma métrica.";
}

function ResultAccordion({ result, principalMediu }: ResultAccordionProps) {
  const status = result.state === "skipped" ? "Ignorada" : result.ok ? "Concluída" : "Falhou";

  return (
    <details className="group py-1">
      <summary className="flex cursor-pointer list-none items-center gap-3 rounded-xl px-2 py-3 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#e7bf08]">
        <span
          className={`h-2 w-2 shrink-0 rounded-full ${result.ok ? "bg-emerald-500" : result.state === "skipped" ? "bg-stone-400" : "bg-red-500"}`}
        />
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm font-semibold text-stone-900">
            {nomeDoProcesso(result.name)}
          </span>
          <span className="mt-0.5 block text-xs text-stone-500">
            {status} · {formatarDuracao(result.durationMs)}
          </span>
        </span>
        <Icon
          name="chevron-down"
          className="h-4 w-4 shrink-0 text-stone-400 transition group-open:rotate-180"
        />
      </summary>

      <div className="px-2 pt-2 pb-5">
        {result.reason && (
          <p className="mb-4 rounded-xl bg-stone-50 p-3 text-xs leading-5 text-stone-600">
            {result.reason}
          </p>
        )}
        {result.frame && (
          <img
            src={fonteDoFrame(result.frame)}
            alt={`Quadro de demonstração de ${nomeDoProcesso(result.name)}`}
            className="mb-4 max-h-72 w-full rounded-xl bg-stone-950 object-contain"
          />
        )}
        <MetricGrid result={result} vazio={semMetrica(result, principalMediu)} />
      </div>
    </details>
  );
}

function formatarDuracao(milliseconds: number): string {
  return milliseconds < 1000
    ? `${Math.round(milliseconds)} ms`
    : `${(milliseconds / 1000).toFixed(1).replace(".", ",")} s`;
}

export default ResultAccordion;
