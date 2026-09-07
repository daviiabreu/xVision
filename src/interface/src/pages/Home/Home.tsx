import Button from "../../components/ui/Button";
import Icon from "../../components/ui/Icon";
import StatisticCard from "../../components/ui/StatisticCard";
import { useVideos } from "../../contexts/useVideos";
import { useTituloDaPagina } from "../../hooks/useTituloDaPagina";
import VideoProcessingList from "./components/VideoProcessingList";
import VideoUpload from "./components/VideoUpload";

function Home() {
  const { total, statistics, recarregar, loading } = useVideos();

  useTituloDaPagina("Análises");

  const emAndamento = statistics.pending + statistics.processing;

  return (
    <div className="mx-auto w-full max-w-7xl px-5 py-8 sm:px-8 lg:px-10 lg:py-10">
      <header className="flex flex-col gap-5 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-sm font-semibold text-amber-700">Visão geral</p>
          <h1 className="mt-2 text-3xl font-bold tracking-tight text-stone-950 sm:text-4xl">
            Análises de finalização
          </h1>
          <p className="mt-3 max-w-2xl text-sm leading-6 text-stone-600 sm:text-base">
            Envie um clipe e acompanhe cada etapa até as métricas da jogada.
          </p>
        </div>
        <Button
          variante="secundario"
          tamanho="compacto"
          disabled={loading}
          onClick={() => void recarregar()}
        >
          <Icon name="refresh" className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          Atualizar
        </Button>
      </header>

      <section
        aria-label="Resumo das análises"
        className="mt-8 grid gap-3 sm:grid-cols-2 xl:grid-cols-4"
      >
        <StatisticCard label="Total" value={total} />
        <StatisticCard label="Em andamento" value={emAndamento} tone="amber" />
        <StatisticCard label="Finalizadas" value={statistics.completed} tone="green" />
        <StatisticCard label="Com falha" value={statistics.failed} tone="red" />
      </section>

      <div className="mt-8 grid items-start gap-8 xl:grid-cols-[minmax(20rem,0.82fr)_minmax(34rem,1.5fr)]">
        <VideoUpload />
        <VideoProcessingList />
      </div>
    </div>
  );
}

export default Home;
