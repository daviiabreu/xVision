import { apiFetch } from "./api";
import { guardarSessao, lerSessao, renovarSessao } from "./sessao";

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

export interface AccountStatistics {
  total: number;
  pendentes: number;
  em_processamento: number;
  finalizadas: number;
  falhas: number;
}

export async function registrar(email: string, senha: string): Promise<Principal> {
  const r = await apiFetch("/v1/auth/register", {
    method: "POST",
    body: JSON.stringify({ email, senha }),
  });

  return (await r.json()) as Principal;
}

export async function solicitarRedefinicao(email: string): Promise<string> {
  const resposta = await apiFetch("/v1/auth/password/forgot", {
    method: "POST",
    body: JSON.stringify({ email }),
  });

  return ((await resposta.json()) as { mensagem: string }).mensagem;
}

export async function redefinirSenha(token: string, senha: string): Promise<string> {
  const resposta = await apiFetch("/v1/auth/password/reset", {
    method: "POST",
    body: JSON.stringify({ token, senha }),
  });

  return ((await resposta.json()) as { mensagem: string }).mensagem;
}

export async function quemSou(token: string): Promise<Principal> {
  const r = await apiFetch("/v1/auth/me", {
    headers: { Authorization: `Bearer ${token}` },
  });

  return (await r.json()) as Principal;
}

export async function atualizarEmail(email: string, senhaAtual: string): Promise<Principal> {
  const resposta = await apiFetch("/v1/auth/me", {
    method: "PATCH",
    body: JSON.stringify({ email, senha_atual: senhaAtual }),
  });

  return (await resposta.json()) as Principal;
}

export async function alterarSenha(senhaAtual: string, novaSenha: string): Promise<string> {
  const tokenDaRequisicao = lerSessao()?.token;
  const resposta = await apiFetch("/v1/auth/password/change", {
    method: "POST",
    body: JSON.stringify({ senha_atual: senhaAtual, nova_senha: novaSenha }),
  });

  const sessao = (await resposta.json()) as SessaoResponse & { mensagem: string };
  if (tokenDaRequisicao && lerSessao()?.token === tokenDaRequisicao) {
    renovarSessao(sessao.access_token, sessao.expira_em);
  }
  return sessao.mensagem;
}

export async function buscarEstatisticasConta(): Promise<AccountStatistics> {
  const resposta = await apiFetch("/v1/auth/me/statistics");
  return (await resposta.json()) as AccountStatistics;
}

export async function autenticar(email: string, senha: string): Promise<Principal> {
  const resposta = await apiFetch("/v1/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, senha }),
  });

  const sessao = (await resposta.json()) as SessaoResponse;
  const eu = await quemSou(sessao.access_token);

  guardarSessao(sessao.access_token, sessao.expira_em, eu.id, eu.nome);

  return eu;
}
