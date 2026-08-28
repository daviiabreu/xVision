import { useEffect } from "react";

const PRODUTO = "XVision";

export function useTituloDaPagina(titulo: string): void {
  useEffect(() => {
    document.title = `${titulo} | ${PRODUTO}`;
  }, [titulo]);
}
