import { useContext } from "react";

import { VideoContext } from "./videoContext";

export function useVideos() {
  const context = useContext(VideoContext);

  if (!context) {
    throw new Error("useVideos must be used inside a VideoProvider");
  }

  return context;
}
