import { Link } from "react-router-dom";

import { classesDeBotao } from "./classesDeBotao";

interface EmptyStateProps {
  titulo: string;
  descricao: string;
}

function EmptyState({ titulo, descricao }: EmptyStateProps) {
  return (
    <div className="flex w-full justify-center px-6 py-12">
      <div className="w-full max-w-3xl text-center">
        <h1 className="text-3xl font-semibold text-gray-900">{titulo}</h1>

        <p className="mt-3 text-sm text-gray-500">{descricao}</p>

        <Link to="/" className={classesDeBotao("primario", "normal", "mt-8 inline-block")}>
          Voltar ao início
        </Link>
      </div>
    </div>
  );
}

export default EmptyState;
