import { useCallback, useEffect, useState } from "react";

import { apagarSessao, lerSessao, SESSAO_PERDIDA } from "../services/sessao";
import { SessaoContext } from "./sessaoContext";

export function SessaoProvider({ children }: { children: React.ReactNode }) {
  const [autenticado, setAutenticado] = useState(() => lerSessao() !== null);

  const entrou = useCallback(() => setAutenticado(true), []);
  const sair = useCallback(() => apagarSessao(), []);

  useEffect(() => {
    const perdeu = () => setAutenticado(false);

    window.addEventListener(SESSAO_PERDIDA, perdeu);

    return () => window.removeEventListener(SESSAO_PERDIDA, perdeu);
  }, []);

  return (
    <SessaoContext.Provider value={{ autenticado, entrou, sair }}>
      {children}
    </SessaoContext.Provider>
  );
}
