/**
 * The STARK mark, inlined from `editors/vscode/icons/stark.svg`.
 *
 * Inlined rather than linked so the header renders in one paint with no asset request, and so the
 * gradient can pick up `currentColor`-independent brand stops that match the editor extension's
 * icon exactly. The two should always look like the same product.
 */
export function Mark({ size = 32 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 512 512"
      role="img"
      aria-label="STARK"
      className="mark"
    >
      <defs>
        <linearGradient id="mark-grad" x1="136" y1="128" x2="390" y2="388" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#62F6FF" />
          <stop offset="0.52" stopColor="#22C8F6" />
          <stop offset="1" stopColor="#6D7CFF" />
        </linearGradient>
      </defs>
      <rect x="32" y="32" width="448" height="448" rx="96" fill="#0B1220" />
      <path
        d="M340 176c-18-24-48-38-84-38-52 0-88 28-88 70 0 36 24 56 74 68l30 7c30 7 42 16 42 33 0 20-20 33-52 33-34 0-58-14-72-38l-40 30c20 34 60 53 112 53 58 0 96-30 96-76 0-38-24-59-76-71l-30-7c-28-7-40-15-40-31 0-18 18-30 46-30 28 0 48 11 60 30z"
        fill="url(#mark-grad)"
      />
    </svg>
  );
}
