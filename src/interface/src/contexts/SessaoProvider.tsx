import { useCallback, useEffect, useState } from "react";

import { apagarSessao, atualizarNomeDaSessao, lerSessao, SESSAO_PERDIDA } from "../services/sessao";
import { SessaoContext } from "./sessaoContext";

export function SessaoProvider({ children }: { children: React.ReactNode }) {
  const [sessao, setSessao] = useState(() => lerSessao());

  const entrou = useCallback(() => setSessao(lerSessao()), []);
  const sair = useCallback(() => apagarSessao(), []);
  const atualizarNome = useCallback((nome: string) => setSessao(atualizarNomeDaSessao(nome)), []);

  useEffect(() => {
    const perdeu = () => setSessao(null);

    window.addEventListener(SESSAO_PERDIDA, perdeu);

    return () => window.removeEventListener(SESSAO_PERDIDA, perdeu);
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
