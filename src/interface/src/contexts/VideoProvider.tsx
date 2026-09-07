import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type Dispatch,
  type SetStateAction,
} from "react";

import {
  getVideos,
  terminou,
  type VideoStatistics,
  type VideoSummary,
} from "../services/videoService";
import { ERROR_MESSAGES } from "../utils/errorMessages";

import { VideoContext } from "./videoContext";

const EMPTY_STATISTICS = { pending: 0, processing: 0, completed: 0, failed: 0 };

interface FeedSetters {
  videos: Dispatch<SetStateAction<VideoSummary[]>>;
  total: Dispatch<SetStateAction<number>>;
  statistics: Dispatch<SetStateAction<VideoStatistics>>;
  loading: Dispatch<SetStateAction<boolean>>;
  error: Dispatch<SetStateAction<string | null>>;
}

function useReload(roundRef: React.RefObject<number>, setters: FeedSetters) {
  return useCallback((): Promise<void> => {
    const current = ++roundRef.current;
    let request: ReturnType<typeof getVideos>;
    try {
      request = getVideos();
    } catch (error) {
      if (current === roundRef.current) {
        setters.error(error instanceof Error ? error.message : ERROR_MESSAGES.SERVER_ERROR);
        setters.loading(false);
      }
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      request
        .then((response) => {
          if (current === roundRef.current) {
            setters.videos(response.videos);
            setters.total(response.total);
            setters.statistics(response.statistics ?? EMPTY_STATISTICS);
            setters.error(null);
            setters.loading(false);
          }
        })
        .catch((error: unknown) => {
          if (current === roundRef.current) {
            setters.error(error instanceof Error ? error.message : ERROR_MESSAGES.SERVER_ERROR);
            setters.loading(false);
          }
        })
        .finally(resolve);
    });
  }, [roundRef, setters]);
}

function useReloadSchedule(
  videos: VideoSummary[],
  reload: () => Promise<void>,
  roundRef: React.RefObject<number>,
) {
  useEffect(() => {
    const initial = setTimeout(() => void reload(), 0);
    return () => {
      clearTimeout(initial);
      roundRef.current += 1;
    };
  }, [reload, roundRef]);

  useEffect(() => {
    const waiting = videos.filter((video) => !terminou(video.videoStatus));
    if (waiting.length === 0) return;
    const seconds = Math.min(...waiting.map((video) => video.retryAfterSeconds ?? 3));
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const refresh = () => {
      void reload().finally(() => {
        if (active) timer = setTimeout(refresh, seconds * 1000);
      });
    };
    timer = setTimeout(refresh, seconds * 1000);
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [videos, reload]);
}

export function VideoProvider({ children }: { children: React.ReactNode }) {
  const [videos, setVideos] = useState<VideoSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [statistics, setStatistics] = useState<VideoStatistics>(EMPTY_STATISTICS);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const roundRef = useRef(0);
  const setters = useMemo(
    () => ({
      videos: setVideos,
      total: setTotal,
      statistics: setStatistics,
      loading: setLoading,
      error: setError,
    }),
    [],
  );
  const reload = useReload(roundRef, setters);
  useReloadSchedule(videos, reload, roundRef);
  return (
    <VideoContext.Provider
      value={{ videos, total, statistics, loading, error, recarregar: reload }}
    >
      {children}
    </VideoContext.Provider>
  );
}
