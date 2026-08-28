import type { ReactNode } from "react";
import { Link } from "react-router-dom";

import logo from "../../assets/logo.svg";
import { useSessao } from "../../contexts/useSessao";
import Button from "../ui/Button";

interface LayoutProps {
  children: ReactNode;
}

function Layout({ children }: LayoutProps) {
  const { autenticado, sair } = useSessao();

  return (
    <div className="min-h-screen">
      <a
        href="#conteudo"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-10 focus:rounded-sm focus:bg-black focus:px-4 focus:py-2 focus:text-sm focus:text-white"
      >
        Pular para o conteúdo
      </a>

      <header className="flex h-16 items-center justify-between bg-[#f5f5f5] px-6">
        {autenticado ? (
          <Link to="/" className="h-full py-2">
            <img src={logo} alt="XVision, ir para o início" className="h-full" />
          </Link>
        ) : (
          <img src={logo} alt="XVision" className="h-full" />
        )}

        {autenticado && (
          <Button variante="secundario" tamanho="compacto" onClick={sair}>
            Sair
          </Button>
        )}
      </header>

      <main id="conteudo" className="flex min-h-[calc(100vh-4rem)] flex-col bg-[#fcfcf9]">
        {children}
      </main>
    </div>
  );
}

export default Layout;
