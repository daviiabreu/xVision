import { useRef, useState } from "react";

import Button from "../../../components/ui/Button";
import Icon from "../../../components/ui/Icon";
import ProgressBar from "../../../components/ui/ProgressBar";
import { useVideos } from "../../../contexts/useVideos";
import { useVideoUpload } from "../../../hooks/useVideoUpload";

function VideoUpload() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const state = useUploadPanel();
  return (
    <section className="rounded-2xl border border-stone-200 bg-white p-5 shadow-[0_1px_2px_rgba(28,25,23,0.04)] sm:p-6">
      <UploadHeader />
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*"
        className="sr-only"
        aria-label="Selecionar vídeo"
        onChange={state.onInputChange}
      />
      <UploadDropZone state={state} open={() => fileInputRef.current?.click()} />
      {state.error && <UploadError error={state.error} />}
      {state.isUploading && <UploadProgress progress={state.uploadProgress} />}
      <Button
        className="mt-5 w-full"
        disabled={!state.file || state.isUploading}
        onClick={() => void state.confirm()}
      >
        <Icon name="sparkles" className="h-4 w-4" />
        {state.isUploading ? "Enviando..." : "Iniciar análise"}
      </Button>
    </section>
  );
}

function useUploadPanel() {
  const [dragging, setDragging] = useState(false);
  const { recarregar } = useVideos();
  const { file, error, handleFileSelect, confirmVideo, isUploading, uploadProgress } =
    useVideoUpload();

  const confirm = async () => {
    const videoId = await confirmVideo();
    if (videoId !== null) {
      await recarregar();
    }
  };

  const selecionar = (selecionado: File | undefined) => {
    if (selecionado) {
      handleFileSelect(selecionado);
    }
  };

  const onInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    selecionar(event.target.files?.[0]);
    event.target.value = "";
  };

  return {
    dragging,
    setDragging,
    file,
    error,
    isUploading,
    uploadProgress,
    selecionar,
    confirm,
    onInputChange,
  };
}

function UploadHeader() {
  return (
    <div className="flex items-start gap-3">
      <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-[#fff8c5] text-amber-800">
        <Icon name="upload" />
      </span>
      <div>
        <h2 className="text-lg font-bold text-stone-950">Nova análise</h2>
        <p className="mt-1 text-sm leading-5 text-stone-500">
          Envie somente o clipe da finalização.
        </p>
      </div>
    </div>
  );
}

function UploadDropZone({
  state,
  open,
}: {
  state: ReturnType<typeof useUploadPanel>;
  open: () => void;
}) {
  return (
    <div
      className={`mt-6 rounded-2xl border-2 border-dashed px-5 py-8 text-center transition ${
        state.dragging ? "border-amber-500 bg-amber-50" : "border-stone-300 bg-stone-50"
      }`}
      onDragEnter={(event) => {
        event.preventDefault();
        state.setDragging(true);
      }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
          state.setDragging(false);
        }
      }}
      onDrop={(event) => {
        event.preventDefault();
        state.setDragging(false);
        if (!state.isUploading) {
          state.selecionar(event.dataTransfer.files[0]);
        }
      }}
    >
      <UploadSelection file={state.file} />
      <Button
        variante="secundario"
        tamanho="compacto"
        className="mt-5"
        disabled={state.isUploading}
        onClick={open}
      >
        {state.file ? "Trocar arquivo" : "Escolher arquivo"}
      </Button>
    </div>
  );
}

function UploadSelection({ file }: { file: File | null }) {
  return (
    <>
      <Icon name="file-video" className="mx-auto h-9 w-9 text-stone-400" />
      {file ? (
        <>
          <p className="mt-4 break-all text-sm font-semibold text-stone-900">{file.name}</p>
          <p className="mt-1 text-xs text-stone-500">{formatarTamanho(file.size)}</p>
        </>
      ) : (
        <>
          <p className="mt-4 text-sm font-semibold text-stone-900">Arraste um vídeo até aqui</p>
          <p className="mt-1 text-xs leading-5 text-stone-500">MP4, MOV ou AVI, até 128 MB</p>
        </>
      )}
    </>
  );
}

function UploadError({ error }: { error: string }) {
  return (
    <p
      role="alert"
      className="mt-4 flex gap-2 rounded-xl bg-red-50 px-3 py-2.5 text-sm text-red-700"
    >
      <Icon name="warning" className="mt-0.5 h-4 w-4 shrink-0" />
      {error}
    </p>
  );
}

function UploadProgress({ progress }: { progress: number }) {
  return (
    <div className="mt-5" aria-live="polite">
      <div className="mb-2 flex justify-between text-xs font-medium text-stone-600">
        <span>Enviando vídeo</span>
        <span>{progress}%</span>
      </div>
      <ProgressBar value={progress} label="Progresso do envio" />
    </div>
  );
}

function formatarTamanho(bytes: number): string {
  if (bytes < 1024 * 1024) {
    return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default VideoUpload;
