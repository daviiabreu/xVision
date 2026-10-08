export const CHAVE = "sessao";

export const SESSAO_PERDIDA = "sessao-perdida";

export interface Sessao {
  token: string;
  expiraEm: number;
  conta: number | null;
  nome: string | null;
}

export function guardarSessao(
  token: string,
  duracaoEmSegundos: number,
  conta: number | null = null,
  nome: string | null = null,
): void {
  const expiraEm = Math.floor(Date.now() / 1000) + duracaoEmSegundos;

  localStorage.setItem(CHAVE, JSON.stringify({ token, expiraEm, conta, nome }));
}

function eSessao(valor: unknown): valor is Sessao {
  const sessao = valor as Sessao | null;

  return (
    typeof sessao?.token === "string" &&
    sessao.token.length > 0 &&
    Number.isFinite(sessao.expiraEm) &&
    (sessao.conta === null || typeof sessao.conta === "number") &&
    (sessao.nome === undefined || sessao.nome === null || typeof sessao.nome === "string")
  );
}

export function lerSessao(): Sessao | null {
  const bruto = localStorage.getItem(CHAVE);

  if (!bruto) {
    return null;
  }

  try {
    const sessao: unknown = JSON.parse(bruto);

    if (!eSessao(sessao) || sessao.expiraEm * 1000 < Date.now()) {
      apagarSessao();
      return null;
    }

    return { ...sessao, nome: sessao.nome ?? null };
  } catch {
    apagarSessao();
    return null;
  }
}

export function apagarSessao(): void {
  localStorage.removeItem(CHAVE);
  window.dispatchEvent(new Event(SESSAO_PERDIDA));
}

export function atualizarNomeDaSessao(nome: string): Sessao | null {
  const sessao = lerSessao();

  if (!sessao) {
    return null;
  }

  const atualizada = { ...sessao, nome };
  localStorage.setItem(CHAVE, JSON.stringify(atualizada));
  return atualizada;
}

export function renovarSessao(token: string, duracaoEmSegundos: number): Sessao | null {
  const atual = lerSessao();

  if (!atual) {
    return null;
  }

  guardarSessao(token, duracaoEmSegundos, atual.conta, atual.nome);
  return lerSessao();
}
