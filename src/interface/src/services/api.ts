import { ERROR_MESSAGES } from "../utils/errorMessages";
import { apagarSessao, lerSessao } from "./sessao";

const API_URL = import.meta.env.VITE_API_URL;

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
  if (!API_URL) {
    throw new ErroHttp(0, "VITE_API_URL", ERROR_MESSAGES.API_URL_MISSING);
  }

  if (!caminho.startsWith("http")) {
    return `${API_URL}${caminho}`;
  }

  if (new URL(caminho).origin !== new URL(API_URL).origin) {
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

export function apiUpload(caminho: string, arquivo: File): Promise<Response> {
  return pedir(caminho, {
    method: "PUT",
    headers: { "Content-Type": arquivo.type },
    body: arquivo,
  });
}
