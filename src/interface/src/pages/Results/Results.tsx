import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import EmptyState from "../../components/ui/EmptyState";
import Icon from "../../components/ui/Icon";
import ProgressBar from "../../components/ui/ProgressBar";
import StatusBadge from "../../components/ui/StatusBadge";
import { useTituloDaPagina } from "../../hooks/useTituloDaPagina";
import { ErroHttp } from "../../services/api";
import {
  getProcessedVideo,
  getRenderVideo,
  getVideoStatus,
  type ProcessedVideo,
  type VideoProgress,
  type VideoResult,
} from "../../services/videoService";
import { ERROR_MESSAGES } from "../../utils/errorMessages";
import { nomeDoProcesso } from "../../utils/nomesDeProcesso";
import { segundosAteReconsultar } from "../../utils/polling";
import Loading from "../Loading/Loading";
import MainResult from "./components/MainResult";
import ResultAccordion from "./components/ResultAccordion";
import TopView from "./components/TopView";

const PRINCIPAL = "shot";
type Estado =
  | { fase: "carregando" }
  | { fase: "processando"; progresso: VideoProgress | null }
  | { fase: "ausente" }
  | { fase: "erro"; mensagem: string }
  | { fase: "pronto"; video: ProcessedVideo; render: string | null };

function watchResult(id: number, setState: (state: Estado) => void) {
  let alive = true;
  let objectUrl: string | null = null;
  let timer: ReturnType<typeof setTimeout> | undefined;
  async function load(first: boolean) {
    if (first) setState({ fase: "carregando" });
    try {
      const video = await getProcessedVideo(id);
      if (!alive) return;
      const render = video.renderUrl ? await baixarRender(video.renderUrl) : null;
      if (!alive) {
        if (render) URL.revokeObjectURL(render);
        return;
      }
      objectUrl = render;
      setState({ fase: "pronto", video, render });
    } catch (error) {
      if (!alive) return;
      if (error instanceof ErroHttp && error.status === 409) {
        const progress = await getVideoStatus(id).catch(() => null);
        if (!alive) return;
        setState({ fase: "processando", progresso: progress });
        timer = setTimeout(
          () => void load(false),
          segundosAteReconsultar(progress?.retryAfterSeconds ?? null) * 1000,
        );
      } else if (error instanceof ErroHttp && error.status === 404) setState({ fase: "ausente" });
      else
        setState({
          fase: "erro",
          mensagem: error instanceof Error ? error.message : ERROR_MESSAGES.SERVER_ERROR,
        });
    }
  }
  void load(true);
  return () => {
    alive = false;
    clearTimeout(timer);
    if (objectUrl) URL.revokeObjectURL(objectUrl);
  };
}

function useResult(id: number, valid: boolean) {
  const [state, setState] = useState<Estado>({ fase: "carregando" });
  useEffect(() => (valid ? watchResult(id, setState) : undefined), [id, valid]);
  return state;
}

function Results() {
  const rawId = useParams().id;
  const id = Number(rawId);
  const valid = rawId !== undefined && Number.isInteger(id) && id > 0;
  const state = useResult(id, valid);
  useTituloDaPagina(state.fase === "pronto" ? state.video.name : `Resultado ${rawId ?? ""}`);
  if (!valid || state.fase === "ausente")
    return (
      <EmptyState
        titulo="Não encontramos esse vídeo"
        descricao="Ele pode ter sido removido, ou o endereço está errado."
      />
    );
  if (state.fase === "carregando")
    return (
      <Loading
        titulo="Carregando o resultado"
        descricao="Buscando tudo o que a análise produziu para este vídeo."
      />
    );
  if (state.fase === "processando") return <ProcessingState progress={state.progresso} />;
  if (state.fase === "erro")
    return <EmptyState titulo="Não conseguimos abrir a análise" descricao={state.mensagem} />;
  return <ReadyResult video={state.video} render={state.render} />;
}

function ResultHeader({ video }: { video: ProcessedVideo }) {
  return (
    <>
      <Link
        to="/"
        className="inline-flex items-center gap-2 rounded-lg text-sm font-semibold text-stone-600 hover:text-stone-950 focus-visible:ring-2 focus-visible:ring-[#e7bf08]"
      >
        <Icon name="arrow-left" className="h-4 w-4" />
        Voltar para análises
      </Link>
      <header className="mt-7 flex flex-col gap-5 border-b border-stone-200 pb-7 sm:flex-row sm:items-end sm:justify-between">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="truncate text-3xl font-bold tracking-tight text-stone-950 sm:text-4xl">
              {video.name}
            </h1>
            <StatusBadge status={video.videoStatus} />
          </div>
          <p className="mt-3 text-sm text-stone-500">
            Análise atualizada em {formatarData(video.updatedAt)}
          </p>
        </div>
      </header>
    </>
  );
}

function ResultSummaries({ video }: { video: ProcessedVideo }) {
  const completed = video.results.filter((result) => result.ok).length;
  const metrics = video.results.reduce((total, result) => total + result.metrics.length, 0);
  const duration = video.results.reduce((total, result) => total + result.durationMs, 0);
  return (
    <section aria-label="Resumo do resultado" className="mt-7 grid gap-3 sm:grid-cols-3">
      <ResultSummary label="Etapas concluídas" value={`${completed} de ${video.results.length}`} />
      <ResultSummary label="Métricas produzidas" value={String(metrics)} />
      <ResultSummary label="Tempo do pipeline" value={formatarDuracao(duration)} />
    </section>
  );
}

function FailureNotice({ video, render }: { video: ProcessedVideo; render: string | null }) {
  if (video.videoStatus !== "Falhou") return null;
  return (
    <div className="mt-7 flex gap-3 rounded-2xl border border-amber-200 bg-amber-50 p-4 text-sm leading-6 text-amber-950">
      <Icon name="warning" className="mt-0.5 h-5 w-5 shrink-0 text-amber-700" />
      <p>
        {video.reason ?? "A análise não terminou."}
        {render && " O vídeo processado abaixo ajuda a entender o que foi detectado."}
      </p>
    </div>
  );
}

function PipelineSteps({
  results,
  principalMeasured,
}: {
  results: VideoResult[];
  principalMeasured: boolean;
}) {
  return (
    <section className="rounded-2xl border border-stone-200 bg-white p-5 shadow-[0_1px_2px_rgba(28,25,23,0.04)] sm:p-6">
      <h2 className="text-lg font-bold text-stone-950">Etapas do pipeline</h2>
      <p className="mt-1 text-sm leading-5 text-stone-500">
        Abra uma etapa para conferir sua evidência e seu diagnóstico.
      </p>
      <div className="mt-5 divide-y divide-stone-200">
        {results.map((result) => (
          <ResultAccordion key={result.name} result={result} principalMediu={principalMeasured} />
        ))}
      </div>
    </section>
  );
}

function ResultBody({ video, render }: { video: ProcessedVideo; render: string | null }) {
  const principal = video.results.find((result) => result.name === PRINCIPAL);
  const others = video.results.filter((result) => result.name !== PRINCIPAL);
  return (
    <div className="mt-8 grid items-start gap-8 xl:grid-cols-[minmax(0,1.3fr)_minmax(21rem,0.7fr)]">
      <div className="space-y-8">
        {render && <TopView src={render} />}
        {principal && <MainResult result={principal} />}
      </div>
      <PipelineSteps results={others} principalMeasured={(principal?.metrics.length ?? 0) > 0} />
    </div>
  );
}

function ReadyResult({ video, render }: { video: ProcessedVideo; render: string | null }) {
  return (
    <article className="mx-auto w-full max-w-7xl px-5 py-8 sm:px-8 lg:px-10 lg:py-10">
      <ResultHeader video={video} />
      <ResultSummaries video={video} />
      <FailureNotice video={video} render={render} />
      <ResultBody video={video} render={render} />
    </article>
  );
}

function ProcessingState({ progress }: { progress: VideoProgress | null }) {
  const percentage = progress?.progress ?? 0;
  const description = progress?.currentStage
    ? `Executando ${nomeDoProcesso(progress.currentStage).toLocaleLowerCase("pt-BR")}.`
    : progress?.queuePosition
      ? `Posição ${progress.queuePosition} na fila de processamento.`
      : "Preparando o pipeline para iniciar.";
  return (
    <div className="mx-auto flex min-h-[70vh] w-full max-w-xl items-center px-5 py-12">
      <section className="w-full rounded-3xl border border-stone-200 bg-white p-7 text-center shadow-[0_20px_60px_rgba(28,25,23,0.07)] sm:p-10">
        <span className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-[#fff8c5] text-amber-800">
          <Icon name="sparkles" className="h-7 w-7" />
        </span>
        <h1 className="mt-6 text-2xl font-bold text-stone-950">Estamos analisando seu vídeo</h1>
        <p className="mt-3 text-sm leading-6 text-stone-500">
          {description} Esta tela se atualiza sozinha.
        </p>
        <div className="mt-7 text-left">
          <div className="mb-2 flex justify-between text-xs font-semibold text-stone-600">
            <span>
              {progress
                ? `${progress.completedStages} de ${progress.totalStages} etapas`
                : "Aguardando progresso"}
            </span>
            <span>{Math.round(percentage)}%</span>
          </div>
          <ProgressBar value={percentage} label="Progresso da análise" />
        </div>
      </section>
    </div>
  );
}

function ResultSummary({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-stone-200 bg-white p-5">
      <p className="text-sm font-medium text-stone-500">{label}</p>
      <p className="mt-2 text-2xl font-bold text-stone-950">{value}</p>
    </div>
  );
}
function formatarData(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? "data indisponível"
    : new Intl.DateTimeFormat("pt-BR", { dateStyle: "medium", timeStyle: "short" }).format(date);
}
function formatarDuracao(milliseconds: number) {
  return milliseconds < 1000
    ? `${Math.round(milliseconds)} ms`
    : `${(milliseconds / 1000).toFixed(1).replace(".", ",")} s`;
}
async function baixarRender(url: string) {
  try {
    return await getRenderVideo(url);
  } catch {
    return null;
  }
}

export default Results;
