import { useContext } from "react";

import { SessaoContext } from "./sessaoContext";

export function useSessao() {
  const context = useContext(SessaoContext);

  if (!context) {
    throw new Error("useSessao must be used inside a SessaoProvider");
  }

  return context;
}
