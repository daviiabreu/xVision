import { apiFetch } from "./api";
import { guardarSessao } from "./sessao";

export interface Principal {
  id: number;
  tipo: "usuario" | "servico";
  nome: string;
}

interface SessaoResponse {
  access_token: string;
  token_type: string;
  expira_em: number;
}

export async function registrar(email: string, senha: string): Promise<Principal> {
  const r = await apiFetch("/v1/auth/register", {
    method: "POST",
    body: JSON.stringify({ email, senha }),
  });

  return (await r.json()) as Principal;
}

export async function quemSou(token: string): Promise<Principal> {
  const r = await apiFetch("/v1/auth/me", {
    headers: { Authorization: `Bearer ${token}` },
  });

  return (await r.json()) as Principal;
}

export async function autenticar(email: string, senha: string): Promise<Principal> {
  const resposta = await apiFetch("/v1/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, senha }),
  });

  const sessao = (await resposta.json()) as SessaoResponse;
  const eu = await quemSou(sessao.access_token);

  guardarSessao(sessao.access_token, sessao.expira_em, eu.id);

  return eu;
}
