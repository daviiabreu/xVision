import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import Accordion from "@mui/material/Accordion";
import AccordionDetails from "@mui/material/AccordionDetails";
import AccordionSummary from "@mui/material/AccordionSummary";

import type { VideoResult } from "../../../services/videoService";
import { nomeDoProcesso } from "../../../utils/nomesDeProcesso";
import MetricGrid from "./MetricGrid";

interface ResultAccordionProps {
  result: VideoResult;
  principalMediu: boolean;
}

function fonteDoFrame(frame: string): string {
  return frame.startsWith("data:") ? frame : `data:image/png;base64,${frame}`;
}

function semMetrica(result: VideoResult, principalMediu: boolean): string {
  if (!result.ok) {
    return "Esta etapa não concluiu, então não há nada para mostrar.";
  }

  return principalMediu
    ? "Etapa concluída. As métricas aparecem em Análise da finalização."
    : "Etapa concluída. Esta análise não produziu métrica nenhuma.";
}

function ResultAccordion({ result, principalMediu }: ResultAccordionProps) {
  return (
    <Accordion
      disableGutters
      elevation={0}
      sx={{ borderBottom: "1px solid #e5e7eb", "&:before": { display: "none" } }}
    >
      <AccordionSummary
        expandIcon={<ExpandMoreIcon />}
        sx={{
          px: 2,
          py: 1,
          minHeight: "64px",
          "& .MuiAccordionSummary-content": { margin: 0 },
        }}
      >
        <div className="flex items-center gap-3">
          <span
            className={`h-2.5 w-2.5 rounded-full ${result.ok ? "bg-green-500" : "bg-red-500"}`}
          />

          <span className="text-base font-medium text-gray-900">{nomeDoProcesso(result.name)}</span>

          <span className="text-sm text-gray-500">{result.ok ? "Concluída" : "Não concluída"}</span>
        </div>
      </AccordionSummary>

      <AccordionDetails sx={{ px: 3, pb: 6 }}>
        {result.frame && (
          <div className="flex justify-center py-6">
            <img
              src={fonteDoFrame(result.frame)}
              alt={`Quadro de demonstração de ${nomeDoProcesso(result.name)}`}
              className="max-h-80 max-w-full rounded-lg object-contain"
            />
          </div>
        )}

        <MetricGrid result={result} vazio={semMetrica(result, principalMediu)} />
      </AccordionDetails>
    </Accordion>
  );
}

export default ResultAccordion;
