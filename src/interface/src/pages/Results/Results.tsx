import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";

import EmptyState from "../../components/ui/EmptyState";
import { useTituloDaPagina } from "../../hooks/useTituloDaPagina";
import { ErroHttp } from "../../services/api";
import { getCachedVideo } from "../../services/videoCache";
import {
  getProcessedVideo,
  getRenderVideo,
  type ProcessedVideo,
} from "../../services/videoService";
import { ERROR_MESSAGES } from "../../utils/errorMessages";
import Loading from "../Loading/Loading";
import MainResult from "./components/MainResult";
import ResultAccordion from "./components/ResultAccordion";
import TopView from "./components/TopView";

const INTERVALO_MS = 5000;
const PRINCIPAL = "shot";

type Estado =
  | { fase: "carregando" }
  | { fase: "processando" }
  | { fase: "ausente" }
  | { fase: "erro"; mensagem: string }
  | { fase: "pronto"; video: ProcessedVideo; render: string | null };

function Results() {
  const { id } = useParams();
  const [estado, setEstado] = useState<Estado>({ fase: "carregando" });

  const numero = Number(id);
  const valido = id !== undefined && Number.isInteger(numero) && numero > 0;

  const guardado = useMemo(() => (valido ? getCachedVideo(numero) : undefined), [valido, numero]);

  useTituloDaPagina(guardado?.name ?? `Resultado ${id ?? ""}`);

  useEffect(() => {
    if (!valido) {
      return;
    }

    let vivo = true;
    let objectUrl: string | null = null;
    let relogio: ReturnType<typeof setTimeout> | undefined;

    async function carregar(primeira: boolean) {
      if (primeira) {
        setEstado({ fase: "carregando" });
      }

      try {
        const video = await getProcessedVideo(numero);

        if (!vivo) {
          return;
        }

        const render = video.renderUrl ? await baixarRender(video.renderUrl) : null;

        if (!vivo) {
          if (render) {
            URL.revokeObjectURL(render);
          }
          return;
        }

        objectUrl = render;
        setEstado({ fase: "pronto", video, render });
      } catch (erro) {
        if (!vivo) {
          return;
        }

        if (erro instanceof ErroHttp && erro.status === 409) {
          setEstado({ fase: "processando" });
          relogio = setTimeout(() => void carregar(false), INTERVALO_MS);
          return;
        }

        if (erro instanceof ErroHttp && erro.status === 404) {
          setEstado({ fase: "ausente" });
          return;
        }

        setEstado({
          fase: "erro",
          mensagem: erro instanceof Error ? erro.message : ERROR_MESSAGES.SERVER_ERROR,
        });
      }
    }

    void carregar(true);

    return () => {
      vivo = false;
      clearTimeout(relogio);

      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [valido, numero]);

  if (!valido) {
    return (
      <EmptyState
        titulo="Não encontramos esse vídeo"
        descricao="Ele pode ter sido removido, ou o endereço está errado."
      />
    );
  }

  if (estado.fase === "carregando") {
    return (
      <Loading
        titulo="Carregando o resultado"
        descricao="Buscando o que a análise produziu para este vídeo."
      />
    );
  }

  if (estado.fase === "processando") {
    return (
      <Loading
        titulo="Estamos analisando seu vídeo"
        descricao="Isso pode levar alguns instantes. A tela se atualiza sozinha quando terminar."
      />
    );
  }

  if (estado.fase === "ausente") {
    return (
      <EmptyState
        titulo="Não encontramos esse vídeo"
        descricao="Ele pode ter sido removido, ou o endereço está errado."
      />
    );
  }

  if (estado.fase === "erro") {
    return <EmptyState titulo="Não conseguimos abrir a análise" descricao={estado.mensagem} />;
  }

  const { video, render } = estado;
  const motivo =
    video.videoStatus === "Falhou" ? (video.reason ?? "A análise não terminou.") : null;
  const principal = video.results.find((result) => result.name === PRINCIPAL);
  const demais = video.results.filter((result) => result.name !== PRINCIPAL);
  const principalMediu = (principal?.metrics.length ?? 0) > 0;

  return (
    <article className="mx-auto w-full max-w-[45rem] px-6 py-16">
      <h1 className="text-4xl font-bold tracking-tight text-gray-900">
        {motivo ? (
          <>
            Não conseguimos <span className="text-amber-700">medir esta finalização</span>
          </>
        ) : (
          <>
            Seu vídeo foi <span className="text-amber-700">analisado com sucesso</span>
          </>
        )}
      </h1>

      <dl className="mt-5 flex flex-wrap gap-x-8 gap-y-2 border-b border-gray-200 pb-5 text-sm">
        {guardado && (
          <div className="flex gap-2">
            <dt className="text-gray-500">Arquivo</dt>
            <dd className="text-gray-700">{guardado.name}</dd>
          </div>
        )}

        <div className="flex gap-2">
          <dt className="text-gray-500">Situação</dt>
          <dd className="text-gray-700">{video.videoStatus}</dd>
        </div>

        <div className="flex gap-2">
          <dt className="text-gray-500">Etapas</dt>
          <dd className="text-gray-700">{video.results.length}</dd>
        </div>
      </dl>

      {motivo && (
        <p className="mt-6 rounded-md border border-amber-200 bg-amber-50 px-4 py-3 text-sm leading-relaxed text-gray-700">
          {motivo}
          {render && " O vídeo abaixo mostra o que a detecção enxergou no clipe."}
        </p>
      )}

      {render && (
        <div className="mt-12">
          <TopView src={render} />
        </div>
      )}

      {principal && (
        <div className="mt-12">
          <MainResult result={principal} />
        </div>
      )}

      <div className="mt-12">
        {demais.map((result) => (
          <ResultAccordion key={result.name} result={result} principalMediu={principalMediu} />
        ))}
      </div>
    </article>
  );
}

async function baixarRender(url: string): Promise<string | null> {
  try {
    return await getRenderVideo(url);
  } catch {
    return null;
  }
}

export default Results;
