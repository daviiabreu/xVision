const INTERVALO_PADRAO_EM_SEGUNDOS = 3;
const INTERVALO_MINIMO_EM_SEGUNDOS = 1;

export function segundosAteReconsultar(informado: number | null): number {
  if (informado === null) {
    return INTERVALO_PADRAO_EM_SEGUNDOS;
  }

  return Number.isFinite(informado) && informado >= INTERVALO_MINIMO_EM_SEGUNDOS
    ? informado
    : INTERVALO_MINIMO_EM_SEGUNDOS;
}
