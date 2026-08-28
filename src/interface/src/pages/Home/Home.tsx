import { useTituloDaPagina } from "../../hooks/useTituloDaPagina";
import VideoProcessingList from "./components/VideoProcessingList";
import VideoUpload from "./components/VideoUpload";

function Home() {
  useTituloDaPagina("Enviar vídeo");

  return (
    <div className="flex flex-1 flex-col items-center px-6 py-12">
      <VideoUpload />

      <VideoProcessingList />
    </div>
  );
}

export default Home;
