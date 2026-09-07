import { useMemo, useState } from "react";

import Button from "../../../components/ui/Button";
import Icon from "../../../components/ui/Icon";
import { useVideos } from "../../../contexts/useVideos";
import type { VideoStatus } from "../../../services/videoService";
import VideoProcessingCard from "./VideoProcessingCard";

type Filtro = "Todos" | VideoStatus;

const FILTROS: Filtro[] = ["Todos", "Pendente", "Em processamento", "Finalizado", "Falhou"];

function VideoProcessingList() {
  const { videos, loading, error, recarregar } = useVideos();
  const [busca, setBusca] = useState("");
  const [filtro, setFiltro] = useState<Filtro>("Todos");

  const visiveis = useMemo(() => {
    const termo = busca.trim().toLocaleLowerCase("pt-BR");
    return videos.filter(
      (video) =>
        (filtro === "Todos" || video.videoStatus === filtro) &&
        (termo.length === 0 || video.name.toLocaleLowerCase("pt-BR").includes(termo)),
    );
  }, [busca, filtro, videos]);

  return (
    <section className="min-w-0 rounded-2xl border border-stone-200 bg-white p-5 shadow-[0_1px_2px_rgba(28,25,23,0.04)] sm:p-6">
      <ListHeader busca={busca} setBusca={setBusca} />
      <Filters filtro={filtro} setFiltro={setFiltro} />
      <ListContent
        videos={videos}
        visiveis={visiveis}
        loading={loading}
        error={error}
        recarregar={recarregar}
      />
    </section>
  );
}

function ListHeader({ busca, setBusca }: { busca: string; setBusca: (value: string) => void }) {
  return (
    <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
      <div>
        <h2 className="text-lg font-bold text-stone-950">Análises recentes</h2>
        <p className="mt-1 text-sm text-stone-500">O progresso é atualizado automaticamente.</p>
      </div>
      <label className="relative block sm:w-52">
        <span className="sr-only">Buscar por arquivo</span>
        <Icon
          name="search"
          className="pointer-events-none absolute top-1/2 left-3 h-4 w-4 -translate-y-1/2 text-stone-400"
        />
        <input
          type="search"
          value={busca}
          onChange={(event) => setBusca(event.target.value)}
          placeholder="Buscar arquivo"
          className="h-10 w-full rounded-xl border border-stone-300 bg-white pr-3 pl-9 text-sm outline-none placeholder:text-stone-400 focus:border-amber-500 focus:ring-2 focus:ring-amber-100"
        />
      </label>
    </div>
  );
}

function Filters({ filtro, setFiltro }: { filtro: Filtro; setFiltro: (value: Filtro) => void }) {
  return (
    <div className="mt-5 flex gap-2 overflow-x-auto pb-1" aria-label="Filtrar análises">
      {FILTROS.map((opcao) => (
        <button
          key={opcao}
          type="button"
          aria-pressed={filtro === opcao}
          onClick={() => setFiltro(opcao)}
          className={`shrink-0 rounded-full px-3 py-1.5 text-xs font-semibold transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#e7bf08] ${filtro === opcao ? "bg-stone-950 text-white" : "bg-stone-100 text-stone-600 hover:bg-stone-200"}`}
        >
          {opcao}
        </button>
      ))}
    </div>
  );
}

interface ListContentProps {
  videos: ReturnType<typeof useVideos>["videos"];
  visiveis: ReturnType<typeof useVideos>["videos"];
  loading: boolean;
  error: string | null;
  recarregar: () => Promise<void>;
}

function ListContent({ videos, visiveis, loading, error, recarregar }: ListContentProps) {
  return (
    <div aria-live="polite" className="mt-5 space-y-3">
      {loading && videos.length === 0 && <LoadingRows />}
      {error && videos.length === 0 && <LoadError error={error} retry={recarregar} />}
      {!loading && !error && videos.length === 0 && <EmptyList />}
      {videos.length > 0 && visiveis.length === 0 && (
        <p className="rounded-xl bg-stone-50 px-4 py-8 text-center text-sm text-stone-500">
          Nenhuma análise corresponde a este filtro.
        </p>
      )}
      {visiveis.map((video) => (
        <VideoProcessingCard key={video.id} video={video} />
      ))}
    </div>
  );
}

function LoadError({ error, retry }: { error: string; retry: () => Promise<void> }) {
  return (
    <div role="alert" className="rounded-xl border border-red-200 bg-red-50 p-5 text-center">
      <Icon name="warning" className="mx-auto h-6 w-6 text-red-600" />
      <p className="mt-2 text-sm font-semibold text-red-800">
        Não foi possível carregar as análises
      </p>
      <p className="mt-1 text-xs text-red-700">{error}</p>
      <Button
        variante="secundario"
        tamanho="compacto"
        className="mt-4"
        onClick={() => void retry()}
      >
        Tentar de novo
      </Button>
    </div>
  );
}

function EmptyList() {
  return (
    <div className="rounded-xl bg-stone-50 px-5 py-10 text-center">
      <Icon name="sparkles" className="mx-auto h-7 w-7 text-stone-400" />
      <p className="mt-3 text-sm font-semibold text-stone-800">Nenhuma análise ainda</p>
      <p className="mt-1 text-xs leading-5 text-stone-500">
        Seu primeiro vídeo aparecerá aqui com o progresso de cada etapa.
      </p>
    </div>
  );
}

function LoadingRows() {
  return (
    <div role="status" aria-label="Carregando análises" className="space-y-3">
      {[0, 1, 2].map((item) => (
        <div key={item} className="h-20 animate-pulse rounded-xl bg-stone-100" />
      ))}
    </div>
  );
}

export default VideoProcessingList;
