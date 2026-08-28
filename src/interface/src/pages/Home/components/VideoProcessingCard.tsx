import { Link } from "react-router-dom";

import Button from "../../../components/ui/Button";
import { terminou, type VideoStatus } from "../../../services/videoService";

interface VideoProcessingCardProps {
  id: number;
  name: string;
  status: VideoStatus | null;
  statusError: boolean;
  onTentarDeNovo: () => void;
  onRemover: () => void;
}

const CORES: Record<VideoStatus, string> = {
  Pendente: "bg-gray-400",
  "Em processamento": "bg-amber-400",
  Falhou: "bg-red-400",
  Finalizado: "bg-green-400",
};

const CARTAO = "flex w-full items-center justify-between rounded-xl px-6 py-5";

function VideoProcessingCard({
  id,
  name,
  status,
  statusError,
  onTentarDeNovo,
  onRemover,
}: VideoProcessingCardProps) {
  const semStatus = statusError || status === null;
  const pronto = terminou(status);

  const identificacao = (
    <div>
      <p className="text-sm font-medium text-gray-600">{name}</p>

      <div className="mt-3 flex items-center gap-2">
        <span className={`h-3 w-3 rounded-full ${semStatus ? "bg-gray-400" : CORES[status]}`} />

        <p className="text-base font-semibold text-gray-900">
          {semStatus ? "Não foi possível consultar o status" : status}
        </p>
      </div>
    </div>
  );

  if (semStatus) {
    return (
      <div className={`${CARTAO} bg-amber-50`}>
        {identificacao}

        <div className="flex gap-2">
          <Button variante="secundario" tamanho="compacto" onClick={onTentarDeNovo}>
            Tentar de novo
          </Button>

          <Button variante="secundario" tamanho="compacto" onClick={onRemover}>
            Remover da lista
          </Button>
        </div>
      </div>
    );
  }

  if (!pronto) {
    return <div className={`${CARTAO} bg-amber-50`}>{identificacao}</div>;
  }

  return (
    <Link to={`/results/${id}`} className={`${CARTAO} bg-amber-100 transition hover:bg-amber-200`}>
      {identificacao}

      <span aria-hidden="true" className="text-2xl font-light text-gray-700">
        &gt;
      </span>
    </Link>
  );
}

export default VideoProcessingCard;
