import { useCallback, useEffect, useRef, useState } from "react";

import { getCachedVideos, removeCachedVideo } from "../services/videoCache";
import { getVideoStatus, terminou } from "../services/videoService";

import { type Video, VideoContext } from "./videoContext";

const INTERVALO_MS = 5000;

export function VideoProvider({ children }: { children: React.ReactNode }) {
  const [videos, setVideos] = useState<Video[]>([]);
  const rodada = useRef(0);

  const buscar = useCallback(async (): Promise<Video[]> => {
    const guardados = getCachedVideos();
    const respostas = await Promise.allSettled(guardados.map((video) => getVideoStatus(video.id)));

    return respostas.map((resposta, indice) =>
      resposta.status === "fulfilled"
        ? { ...guardados[indice], status: resposta.value.video_status, statusError: false }
        : { ...guardados[indice], status: null, statusError: true },
    );
  }, []);

  const recarregar = useCallback(async () => {
    const minha = ++rodada.current;
    const lista = await buscar();

    if (minha === rodada.current) {
      setVideos(lista);
    }
  }, [buscar]);

  useEffect(() => {
    void recarregar();

    return () => {
      rodada.current += 1;
    };
  }, [recarregar]);

  const esperando = videos.some((video) => !video.statusError && !terminou(video.status));

  useEffect(() => {
    if (!esperando) {
      return;
    }

    const relogio = setInterval(() => void recarregar(), INTERVALO_MS);

    return () => clearInterval(relogio);
  }, [esperando, recarregar]);

  const remover = useCallback(
    (id: number) => {
      removeCachedVideo(id);
      void recarregar();
    },
    [recarregar],
  );

  return (
    <VideoContext.Provider value={{ videos, recarregar, remover }}>
      {children}
    </VideoContext.Provider>
  );
}
