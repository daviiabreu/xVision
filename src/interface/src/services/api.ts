import { ERROR_MESSAGES } from "../utils/errorMessages";
import { apagarSessao, lerSessao } from "./sessao";

const API_URL = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");

export function apiPublicUrl(caminho: string): string {
  return `${API_URL}${caminho}`;
}

const MENSAGENS: Record<number, string> = {
  400: ERROR_MESSAGES.VIDEO_EMPTY,
  413: ERROR_MESSAGES.VIDEO_TOO_LARGE,
  415: ERROR_MESSAGES.VIDEO_UNREADABLE,
  422: ERROR_MESSAGES.INVALID_REQUEST,
  429: ERROR_MESSAGES.TOO_MANY_ATTEMPTS,
  503: ERROR_MESSAGES.QUEUE_FULL,
};

export class ErroHttp extends Error {
  readonly status: number;
  readonly detalhe: string;

  constructor(
    status: number,
    detalhe = "",
    mensagem: string = MENSAGENS[status] ?? ERROR_MESSAGES.SERVER_ERROR,
  ) {
    super(mensagem);
    this.status = status;
    this.detalhe = detalhe;
  }
}

export class NaoAutenticado extends ErroHttp {
  constructor(detalhe = "") {
    super(401, detalhe, ERROR_MESSAGES.SESSION_EXPIRED);
  }
}

function alvoDe(caminho: string): string {
  if (!caminho.startsWith("http")) {
    return apiPublicUrl(caminho);
  }

  const origemEsperada = API_URL ? new URL(API_URL).origin : window.location.origin;

  if (new URL(caminho).origin !== origemEsperada) {
    throw new ErroHttp(0, caminho, ERROR_MESSAGES.UNEXPECTED_ADDRESS);
  }

  return caminho;
}

function comoTexto(detail: unknown): string {
  if (typeof detail === "string") {
    return detail;
  }

  return detail === undefined || detail === null ? "" : JSON.stringify(detail);
}

async function detalheDe(resposta: Response): Promise<string> {
  try {
    return comoTexto(((await resposta.json()) as { detail?: unknown }).detail);
  } catch {
    return "";
  }
}

export async function pedir(caminho: string, init: RequestInit = {}): Promise<Response> {
  const sessao = lerSessao();
  const alvo = alvoDe(caminho);
  let resposta: Response;

  try {
    resposta = await fetch(alvo, {
      ...init,
      headers: {
        ...(sessao ? { Authorization: `Bearer ${sessao.token}` } : {}),
        ...init.headers,
      },
    });
  } catch {
    throw new Error(ERROR_MESSAGES.NO_CONNECTION);
  }

  if (resposta.status === 401) {
    apagarSessao();
    throw new NaoAutenticado(await detalheDe(resposta));
  }

  if (!resposta.ok) {
    throw new ErroHttp(resposta.status, await detalheDe(resposta));
  }

  return resposta;
}

export function apiFetch(caminho: string, init: RequestInit = {}): Promise<Response> {
  return pedir(caminho, {
    ...init,
    headers: init.body ? { "Content-Type": "application/json", ...init.headers } : init.headers,
  });
}

export async function apiBlob(caminho: string): Promise<Blob | null> {
  try {
    return await (await pedir(caminho)).blob();
  } catch (erro) {
    if (erro instanceof ErroHttp && erro.status === 404) {
      return null;
    }
    throw erro;
  }
}

export function apiUpload(
  caminho: string,
  arquivo: File,
  onProgress?: (percentage: number) => void,
): Promise<Response> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const sessao = lerSessao();
    xhr.open("PUT", alvoDe(caminho));
    xhr.setRequestHeader("Content-Type", arquivo.type);
    if (sessao) {
      xhr.setRequestHeader("Authorization", `Bearer ${sessao.token}`);
    }
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onProgress?.(Math.round((event.loaded * 100) / event.total));
      }
    };
    xhr.onerror = () => reject(new Error(ERROR_MESSAGES.NO_CONNECTION));
    xhr.onload = () => {
      const detalhe = (() => {
        try {
          return comoTexto((JSON.parse(xhr.responseText) as { detail?: unknown }).detail);
        } catch {
          return "";
        }
      })();
      if (xhr.status === 401) {
        apagarSessao();
        reject(new NaoAutenticado(detalhe));
        return;
      }
      if (xhr.status < 200 || xhr.status >= 300) {
        reject(new ErroHttp(xhr.status, detalhe));
        return;
      }
      onProgress?.(100);
      resolve(new Response(xhr.responseText || null, { status: xhr.status }));
    };
    xhr.send(arquivo);
  });
}
