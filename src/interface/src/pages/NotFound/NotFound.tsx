import EmptyState from "../../components/ui/EmptyState";
import { useTituloDaPagina } from "../../hooks/useTituloDaPagina";

function NotFound() {
  useTituloDaPagina("Página não encontrada");

  return (
    <EmptyState titulo="Página não encontrada" descricao="O endereço que você abriu não existe." />
  );
}

export default NotFound;
