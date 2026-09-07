import type { VideoStatus } from "../../services/videoService";

const CORES: Record<VideoStatus, string> = {
  Pendente: "bg-stone-100 text-stone-700 ring-stone-200",
  "Em processamento": "bg-amber-50 text-amber-800 ring-amber-200",
  Falhou: "bg-red-50 text-red-700 ring-red-200",
  Finalizado: "bg-emerald-50 text-emerald-700 ring-emerald-200",
};

function StatusBadge({ status }: { status: VideoStatus }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold ring-1 ring-inset ${CORES[status]}`}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current" aria-hidden="true" />
      {status}
    </span>
  );
}

export default StatusBadge;
