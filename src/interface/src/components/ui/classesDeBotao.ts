export type Variante = "primario" | "secundario";
export type Tamanho = "normal" | "compacto";

const BASE =
  "cursor-pointer rounded-sm text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-50";

const VARIANTES: Record<Variante, string> = {
  primario: "bg-black text-white hover:bg-gray-800",
  secundario: "border border-gray-300 text-gray-700 hover:bg-gray-100",
};

const TAMANHOS: Record<Tamanho, string> = {
  normal: "px-6 py-3",
  compacto: "px-4 py-2",
};

export function classesDeBotao(
  variante: Variante = "primario",
  tamanho: Tamanho = "normal",
  extra = "",
): string {
  return `${BASE} ${VARIANTES[variante]} ${TAMANHOS[tamanho]} ${extra}`.trimEnd();
}
