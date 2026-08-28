import { useRef } from "react";

import Button from "../../../components/ui/Button";
import { useVideos } from "../../../contexts/useVideos";
import { useVideoUpload } from "../../../hooks/useVideoUpload";
import { addCachedVideo } from "../../../services/videoCache";

function VideoUpload() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { recarregar } = useVideos();
  const { file, error, handleFileSelect, confirmVideo, isUploading } = useVideoUpload();

  const abrirSeletor = () => fileInputRef.current?.click();

  const handleConfirm = async () => {
    const enviado = file;
    const videoId = await confirmVideo();

    if (videoId === null || !enviado) {
      return;
    }

    addCachedVideo({ id: videoId, name: enviado.name });
    await recarregar();
  };

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selecionado = event.target.files?.[0];

    event.target.value = "";

    if (selecionado) {
      handleFileSelect(selecionado);
    }
  };

  return (
    <section className="flex w-full max-w-2xl flex-col items-center text-center">
      <h1 className="text-3xl font-semibold text-gray-900">
        Adicione o <span className="font-bold text-amber-700">vídeo da finalização</span>
      </h1>

      <p className="mt-3 text-sm text-gray-500">
        Selecione um arquivo de vídeo para iniciar a análise.
      </p>

      <input
        ref={fileInputRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={handleInputChange}
      />

      <div
        onClick={abrirSeletor}
        className="mt-8 flex w-full cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-gray-300 px-8 py-16 hover:bg-amber-50"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth={1.5}
          stroke="currentColor"
          aria-hidden="true"
          className="h-10 w-10 text-gray-400"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-9-12v12m0-12 3.75 3.75M12 4.5 8.25 8.25"
          />
        </svg>

        {file ? (
          <>
            <p className="mt-4 text-base font-medium text-gray-700">{file.name}</p>

            <p className="mt-2 text-sm text-gray-600">
              {(file.size / (1024 * 1024)).toFixed(2)} MB
            </p>
          </>
        ) : (
          <>
            <p className="mt-4 text-base font-medium text-gray-700">Selecione seu vídeo</p>

            <p className="mt-2 text-sm text-gray-600">
              Exemplos de formatos aceitos: MP4, MOV e AVI.
            </p>

            <p className="mt-1 text-sm text-gray-600">Tamanho máximo: 128 MB</p>
          </>
        )}

        {error && (
          <p role="alert" className="mt-4 text-sm font-medium text-red-600">
            {error}
          </p>
        )}

        {file ? (
          <div className="mt-6 flex gap-3">
            <Button
              variante="secundario"
              disabled={isUploading}
              onClick={(event) => {
                event.stopPropagation();
                abrirSeletor();
              }}
            >
              Alterar vídeo
            </Button>

            <Button
              disabled={isUploading}
              onClick={(event) => {
                event.stopPropagation();
                void handleConfirm();
              }}
            >
              {isUploading ? "Enviando vídeo..." : "Confirmar vídeo"}
            </Button>
          </div>
        ) : (
          <Button
            className="mt-6"
            onClick={(event) => {
              event.stopPropagation();
              abrirSeletor();
            }}
          >
            Selecionar vídeo
          </Button>
        )}
      </div>
    </section>
  );
}

export default VideoUpload;
