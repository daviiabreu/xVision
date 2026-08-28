import { createContext } from "react";

export interface SessaoContextData {
  autenticado: boolean;
  entrou: () => void;
  sair: () => void;
}

export const SessaoContext = createContext<SessaoContextData | undefined>(undefined);
