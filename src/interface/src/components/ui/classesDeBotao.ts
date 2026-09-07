export type Variante = "primario" | "secundario";
export type Tamanho = "normal" | "compacto";

const BASE =
  "inline-flex cursor-pointer items-center justify-center gap-2 rounded-xl text-sm font-semibold transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#e7bf08] focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50";

const VARIANTES: Record<Variante, string> = {
  primario: "bg-stone-950 text-white shadow-sm hover:bg-stone-800",
  secundario: "border border-stone-300 bg-white text-stone-700 hover:bg-stone-50",
};

const TAMANHOS: Record<Tamanho, string> = {
  normal: "min-h-11 px-5 py-2.5",
  compacto: "min-h-9 px-3.5 py-1.5",
};

export function classesDeBotao(
  variante: Variante = "primario",
  tamanho: Tamanho = "normal",
  extra = "",
): string {
  return `${BASE} ${VARIANTES[variante]} ${TAMANHOS[tamanho]} ${extra}`.trimEnd();
}
