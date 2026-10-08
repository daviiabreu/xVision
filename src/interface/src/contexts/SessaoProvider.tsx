import { useCallback, useEffect, useState } from "react";

import {
  apagarSessao,
  atualizarNomeDaSessao,
  CHAVE,
  lerSessao,
  SESSAO_PERDIDA,
} from "../services/sessao";
import { SessaoContext } from "./sessaoContext";

const MARGEM_DE_EXPIRACAO_MS = 50;
// setTimeout estoura acima de 2^31 - 1 ms e dispara na hora; a sessão dura horas, mas o teto evita o salto.
const ATRASO_MAXIMO_DO_TIMER_MS = 2 ** 31 - 1;

export function SessaoProvider({ children }: { children: React.ReactNode }) {
  const [sessao, setSessao] = useState(() => lerSessao());

  const entrou = useCallback(() => setSessao(lerSessao()), []);
  const sair = useCallback(() => apagarSessao(), []);
  const atualizarNome = useCallback((nome: string) => setSessao(atualizarNomeDaSessao(nome)), []);

  const expiraEm = sessao?.expiraEm;

  useEffect(() => {
    if (expiraEm === undefined) return;

    const restante = expiraEm * 1000 - Date.now() + MARGEM_DE_EXPIRACAO_MS;
    const timer = setTimeout(
      () => setSessao(lerSessao()),
      Math.min(Math.max(restante, 0), ATRASO_MAXIMO_DO_TIMER_MS),
    );

    return () => clearTimeout(timer);
  }, [expiraEm]);

  useEffect(() => {
    const perdeu = () => setSessao(null);
    const mudouEmOutraAba = (evento: StorageEvent) => {
      if (evento.key === null || evento.key === CHAVE) setSessao(lerSessao());
    };

    window.addEventListener(SESSAO_PERDIDA, perdeu);
    window.addEventListener("storage", mudouEmOutraAba);

    return () => {
      window.removeEventListener(SESSAO_PERDIDA, perdeu);
      window.removeEventListener("storage", mudouEmOutraAba);
    };
  }, []);

  return (
    <SessaoContext.Provider
      value={{
        autenticado: sessao !== null,
        nome: sessao?.nome ?? null,
        atualizarNome,
        entrou,
        sair,
      }}
    >
      {children}
    </SessaoContext.Provider>
  );
}
