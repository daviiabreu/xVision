interface LoadingProps {
  titulo: string;
  descricao: string;
}

function Loading({ titulo, descricao }: LoadingProps) {
  return (
    <div className="flex min-h-[70vh] flex-col items-center justify-center px-5 text-center">
      <div
        role="status"
        aria-label={titulo}
        className="h-12 w-12 animate-spin rounded-full border-4 border-stone-200 border-t-amber-500"
      />

      <h1 className="mt-8 text-2xl font-bold text-stone-950">{titulo}</h1>

      <p className="mt-3 max-w-md text-sm leading-6 text-stone-500">{descricao}</p>
    </div>
  );
}

export default Loading;
