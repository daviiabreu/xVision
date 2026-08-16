'use client';

import Image from 'next/image';
import { useCallback, useRef, useState, useSyncExternalStore } from 'react';
import type { PointerEvent as ReactPointerEvent, ReactNode } from 'react';
import { useTheme } from 'next-themes';

const emptySubscribe = () => () => {};

const MIN_SCALE = 1;
const MAX_SCALE = 6;
const ZOOM_STEP = 1.4;

const INITIAL_VIEW = { x: 0, y: 0, scale: 1 };

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

export interface DiagramProps {
  /** Caminho do PNG exportado no tema claro, a partir de `public/`. */
  light: string;
  /** Caminho do PNG exportado no tema escuro, a partir de `public/`. */
  dark: string;
  /** Descrição do diagrama para leitores de tela. Obrigatória. */
  alt: string;
  width: number;
  height: number;
}

/**
 * Diagrama exportado do Eraser, com arrastar e zoom.
 *
 * Não renderiza no servidor porque depende do tema resolvido: renderizar antes
 * disso escolheria a imagem errada e causaria hydration mismatch.
 */
export function Diagram(props: DiagramProps) {
  const mounted = useSyncExternalStore(
    emptySubscribe,
    () => true,
    () => false,
  );

  if (!mounted) return null;
  return <DiagramContent {...props} />;
}

function DiagramContent({ light, dark, alt, width, height }: DiagramProps) {
  const { resolvedTheme } = useTheme();
  const src = resolvedTheme === 'dark' ? dark : light;

  const [view, setView] = useState(INITIAL_VIEW);
  const dragOrigin = useRef<{ x: number; y: number } | null>(null);

  const reset = useCallback(() => setView(INITIAL_VIEW), []);

  const zoomBy = useCallback((factor: number) => {
    setView((current) => {
      const scale = Math.min(
        MAX_SCALE,
        Math.max(MIN_SCALE, current.scale * factor),
      );

      if (scale === current.scale) return current;
      // Voltar ao zoom mínimo recentraliza: um pan preso fora da tela não teria volta.
      if (scale === MIN_SCALE) return INITIAL_VIEW;

      return { ...current, scale };
    });
  }, []);

  const onPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    // Sem zoom a imagem já cabe: arrastar só a tiraria do lugar. No celular isso
    // também devolve o gesto de rolagem para a página.
    if (view.scale === MIN_SCALE) return;

    dragOrigin.current = {
      x: event.clientX - view.x,
      y: event.clientY - view.y,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const onPointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const origin = dragOrigin.current;
    if (!origin) return;

    const box = event.currentTarget.getBoundingClientRect();
    // Limita o arrasto à área que o zoom revelou, senão a imagem some da tela.
    const limitX = (box.width * (view.scale - 1)) / 2;
    const limitY = (box.height * (view.scale - 1)) / 2;

    setView((current) => ({
      ...current,
      x: clamp(event.clientX - origin.x, -limitX, limitX),
      y: clamp(event.clientY - origin.y, -limitY, limitY),
    }));
  };

  const onPointerUp = (event: ReactPointerEvent<HTMLDivElement>) => {
    dragOrigin.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  return (
    <div className="group relative mx-auto mt-6 overflow-hidden">
      <div
        className={
          view.scale === MIN_SCALE
            ? undefined
            : 'cursor-grab touch-none active:cursor-grabbing'
        }
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onDoubleClick={reset}
      >
        <div
          style={{
            transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})`,
            transformOrigin: 'center center',
          }}
        >
          <Image
            src={src}
            alt={alt}
            width={width}
            height={height}
            draggable={false}
            className="pointer-events-none w-full select-none"
          />
        </div>
      </div>

      <div className="absolute right-2 top-2 flex gap-1 opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
        <ControlButton onClick={() => zoomBy(ZOOM_STEP)} label="Aproximar">
          +
        </ControlButton>
        <ControlButton onClick={() => zoomBy(1 / ZOOM_STEP)} label="Afastar">
          −
        </ControlButton>
        <ControlButton onClick={reset} label="Reposicionar">
          ⤢
        </ControlButton>
      </div>
    </div>
  );
}

function ControlButton({
  onClick,
  label,
  children,
}: {
  onClick: () => void;
  label: string;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
      className="size-7 rounded-md border border-fd-border bg-fd-card text-sm text-fd-muted-foreground hover:text-fd-foreground"
    >
      {children}
    </button>
  );
}
