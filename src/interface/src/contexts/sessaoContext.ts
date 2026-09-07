import { createContext } from "react";

export interface SessaoContextData {
  autenticado: boolean;
  nome: string | null;
  atualizarNome: (nome: string) => void;
  entrou: () => void;
  sair: () => void;
}

export const SessaoContext = createContext<SessaoContextData | undefined>(undefined);
