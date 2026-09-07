import { Link } from "react-router-dom";

import Icon from "../../../components/ui/Icon";
import ProgressBar from "../../../components/ui/ProgressBar";
import StatusBadge from "../../../components/ui/StatusBadge";
import { terminou, type VideoSummary } from "../../../services/videoService";
import { nomeDoProcesso } from "../../../utils/nomesDeProcesso";

function VideoProcessingCard({ video }: { video: VideoSummary }) {
  const conteudo = (
    <>
      <CardHeader video={video} />
      <ProcessingProgress video={video} />
      <FailureReason video={video} />
    </>
  );

  const classes =
    "block rounded-xl border border-stone-200 bg-white p-4 transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#e7bf08] focus-visible:ring-offset-2";

  if (terminou(video.videoStatus)) {
    return (
      <Link
        to={`/results/${video.id}`}
        className={`${classes} hover:border-stone-300 hover:shadow-sm`}
      >
        {conteudo}
      </Link>
    );
  }

  return <div className={classes}>{conteudo}</div>;
}

function CardHeader({ video }: { video: VideoSummary }) {
  return (
    <div className="min-w-0 sm:flex sm:items-start sm:justify-between sm:gap-5">
      <div className="flex min-w-0 items-center gap-2.5">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-stone-100 text-stone-500">
          <Icon name="file-video" className="h-4.5 w-4.5" />
        </span>
        <div className="min-w-0">
          <h3 className="truncate text-sm font-semibold text-stone-950">{video.name}</h3>
          <p className="mt-0.5 text-xs text-stone-500">{formatarData(video.createdAt)}</p>
        </div>
      </div>
      <div className="mt-3 flex items-center justify-between gap-3 pl-11 sm:mt-0 sm:pl-0">
        <StatusBadge status={video.videoStatus} />
        {terminou(video.videoStatus) && (
          <Icon name="arrow-right" className="h-4 w-4 text-stone-400" />
        )}
      </div>
    </div>
  );
}

function ProcessingProgress({ video }: { video: VideoSummary }) {
  if (terminou(video.videoStatus)) return null;
  const stage =
    video.videoStatus === "Pendente"
      ? video.queuePosition
        ? `Posição ${video.queuePosition} na fila`
        : "Aguardando processamento"
      : video.currentStage
        ? nomeDoProcesso(video.currentStage)
        : "Preparando pipeline";
  return (
    <div className="mt-4 pl-11">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-xs text-stone-600">
        <span>{stage}</span>
        <span className="font-semibold text-stone-700">{Math.round(video.progress)}%</span>
      </div>
      <ProgressBar value={video.progress} label={`Progresso de ${video.name}`} />
    </div>
  );
}

function FailureReason({ video }: { video: VideoSummary }) {
  if (video.videoStatus !== "Falhou" || !video.reason) return null;
  return <p className="mt-3 line-clamp-2 pl-11 text-xs leading-5 text-red-700">{video.reason}</p>;
}

function formatarData(valor: string): string {
  const data = new Date(valor);
  if (Number.isNaN(data.getTime())) {
    return "Data indisponível";
  }

  return new Intl.DateTimeFormat("pt-BR", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  }).format(data);
}

export default VideoProcessingCard;
