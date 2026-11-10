export const Mark = ({ size = 28 }: { size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 40 40" fill="none" aria-hidden>
    <path d="M6 14V6h8M26 6h8v8M34 26v8h-8M14 34H6v-8" stroke="#22d3ee" strokeWidth="2.6" strokeLinecap="square" />
    <path d="M20 10l8.7 5v10L20 30l-8.7-5V15z" stroke="#e6edf3" strokeWidth="1.8" strokeLinejoin="round" />
    <circle cx="20" cy="20" r="3.2" fill="#22d3ee" />
  </svg>
);
export const Wordmark = ({ size = 22, tag = true }: { size?: number; tag?: boolean }) => (
  <div className="wordmark" style={{ fontSize: size }}>
    <Mark size={size * 1.35} /><span>SANKET<b>RA</b></span>{tag && <i>AI</i>}
  </div>
);
