import { lerSessao } from "./sessao";

export interface CachedVideo {
  id: number;
  name: string;
}

function chave(): string | null {
  const conta = lerSessao()?.conta;

  return typeof conta === "number" ? `videos:${conta}` : null;
}

function eVideo(valor: unknown): valor is CachedVideo {
  const video = valor as CachedVideo | null;

  return typeof video?.id === "number" && typeof video?.name === "string";
}

export function getCachedVideos(): CachedVideo[] {
  const guardados = chave();
  const stored = guardados === null ? null : localStorage.getItem(guardados);

  if (!stored) {
    return [];
  }

  try {
    const lidos: unknown = JSON.parse(stored);

    return Array.isArray(lidos) ? lidos.filter(eVideo) : [];
  } catch {
    return [];
  }
}

export function getCachedVideo(id: number): CachedVideo | undefined {
  return getCachedVideos().find((video) => video.id === id);
}

export function addCachedVideo(video: CachedVideo): void {
  const guardados = chave();

  if (guardados === null) {
    return;
  }

  localStorage.setItem(guardados, JSON.stringify([...getCachedVideos(), video]));
}

export function removeCachedVideo(id: number): void {
  const guardados = chave();

  if (guardados === null) {
    return;
  }

  const restantes = getCachedVideos().filter((video) => video.id !== id);

  localStorage.setItem(guardados, JSON.stringify(restantes));
}
