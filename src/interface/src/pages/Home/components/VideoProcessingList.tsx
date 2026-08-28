import { useVideos } from "../../../contexts/useVideos";
import VideoProcessingCard from "./VideoProcessingCard";

function VideoProcessingList() {
  const { videos, recarregar, remover } = useVideos();

  if (videos.length === 0) {
    return null;
  }

  return (
    <section className="mt-12 w-full max-w-2xl">
      <h2 className="mb-4 text-lg font-semibold text-gray-900">Processamentos</h2>

      <div aria-live="polite" className="flex flex-col gap-3">
        {videos.map((video) => (
          <VideoProcessingCard
            key={video.id}
            id={video.id}
            name={video.name}
            status={video.status}
            statusError={video.statusError}
            onTentarDeNovo={() => void recarregar()}
            onRemover={() => remover(video.id)}
          />
        ))}
      </div>
    </section>
  );
}

export default VideoProcessingList;
