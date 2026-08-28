const CHAVE = "sessao";

export const SESSAO_PERDIDA = "sessao-perdida";

export interface Sessao {
  token: string;
  expiraEm: number;
  conta: number | null;
}

export function guardarSessao(
  token: string,
  duracaoEmSegundos: number,
  conta: number | null = null,
): void {
  const expiraEm = Math.floor(Date.now() / 1000) + duracaoEmSegundos;

  localStorage.setItem(CHAVE, JSON.stringify({ token, expiraEm, conta }));
}

function eSessao(valor: unknown): valor is Sessao {
  const sessao = valor as Sessao | null;

  return (
    typeof sessao?.token === "string" &&
    sessao.token.length > 0 &&
    Number.isFinite(sessao.expiraEm) &&
    (sessao.conta === null || typeof sessao.conta === "number")
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

    return sessao;
  } catch {
    apagarSessao();
    return null;
  }
}

export function apagarSessao(): void {
  localStorage.removeItem(CHAVE);
  window.dispatchEvent(new Event(SESSAO_PERDIDA));
}
