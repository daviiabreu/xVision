interface ProgressBarProps {
  value: number;
  label: string;
}

function ProgressBar({ value, label }: ProgressBarProps) {
  const bounded = Math.max(0, Math.min(100, value));

  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(bounded)}
      className="h-2 overflow-hidden rounded-full bg-stone-200"
    >
      <div
        className="h-full rounded-full bg-[#e7bf08] transition-[width] duration-500"
        style={{ width: `${bounded}%` }}
      />
    </div>
  );
}

export default ProgressBar;
