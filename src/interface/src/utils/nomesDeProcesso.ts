const NOMES: Record<string, string> = {
  shot: "Análise da finalização",
  ball: "Detecção da bola",
  players: "Detecção de jogadores",
  pose: "Pose do finalizador",
  tracking: "Rastreamento",
  field: "Geometria do campo",
  identity: "Identificação de jogador",
  assemble: "Montagem da percepção",
};

export function nomeDoProcesso(chave: string): string {
  return NOMES[chave] ?? chave;
}
