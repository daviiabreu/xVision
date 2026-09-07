interface TopViewProps {
  src: string;
}

function TopView({ src }: TopViewProps) {
  return (
    <section className="overflow-hidden rounded-2xl border border-stone-200 bg-white shadow-[0_1px_2px_rgba(28,25,23,0.04)]">
      <div className="p-5 sm:p-6">
        <p className="text-xs font-bold tracking-wide text-amber-700 uppercase">
          Evidência consolidada
        </p>
        <h2 className="mt-1 text-xl font-bold text-stone-950">Visão superior</h2>
        <p className="mt-2 text-sm leading-6 text-stone-500">
          Compare as detecções no clipe com as posições projetadas no campo.
        </p>
      </div>
      <div className="bg-stone-950">
        <video
          src={src}
          controls
          playsInline
          preload="metadata"
          className="block aspect-video h-auto w-full object-contain"
        />
      </div>
    </section>
  );
}

export default TopView;
