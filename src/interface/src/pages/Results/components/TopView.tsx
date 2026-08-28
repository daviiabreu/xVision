interface TopViewProps {
  src: string;
}

function TopView({ src }: TopViewProps) {
  return (
    <section className="w-full text-left">
      <h2 className="text-xl font-semibold text-gray-900">Visão superior</h2>

      <p className="mt-2 text-sm text-gray-500">
        À esquerda, cada jogador detectado e o rastro do movimento. À direita, a posição projetada
        no gramado a partir da geometria do campo.
      </p>

      <div className="mt-5 overflow-hidden rounded-sm border border-gray-200 bg-black">
        <video src={src} controls playsInline preload="metadata" className="block h-auto w-full" />
      </div>
    </section>
  );
}

export default TopView;
