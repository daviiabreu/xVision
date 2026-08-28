interface LoadingProps {
  titulo: string;
  descricao: string;
}

function Loading({ titulo, descricao }: LoadingProps) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center px-6 text-center">
      <div
        role="status"
        aria-label={titulo}
        className="h-12 w-12 animate-spin rounded-full border-4 border-gray-200 border-t-amber-400"
      />

      <h1 className="mt-8 text-2xl font-semibold text-gray-900">{titulo}</h1>

      <p className="mt-3 max-w-md text-sm text-gray-500">{descricao}</p>
    </div>
  );
}

export default Loading;
