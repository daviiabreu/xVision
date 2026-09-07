import { Link } from "react-router-dom";

import { classesDeBotao } from "./classesDeBotao";

interface EmptyStateProps {
  titulo: string;
  descricao: string;
}

function EmptyState({ titulo, descricao }: EmptyStateProps) {
  return (
    <div className="flex min-h-[70vh] w-full items-center justify-center px-5 py-12">
      <div className="w-full max-w-xl rounded-3xl border border-stone-200 bg-white p-8 text-center shadow-[0_20px_60px_rgba(28,25,23,0.07)] sm:p-10">
        <h1 className="text-3xl font-bold tracking-tight text-stone-950">{titulo}</h1>

        <p className="mt-3 text-sm leading-6 text-stone-500">{descricao}</p>

        <Link to="/" className={classesDeBotao("primario", "normal", "mt-8")}>
          Voltar ao início
        </Link>
      </div>
    </div>
  );
}

export default EmptyState;
