// Line icons (2px strokes in currentColor); a few stat icons carry a soft fill in their accent color.

const line = (paths: string) =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${paths}</svg>`;

export const ICON = {
  /** ASL Dragon's mark: a start dot and a motion arrow, the way ASL dictionaries draw a sign's movement. */
  mark: `<svg viewBox="0 0 32 32"><circle cx="7" cy="23" r="2.8" fill="#18943F"/><path d="M7 23C9 13 17 7 26 9M20.6 4.4 26 9l-6.7 1.9" fill="none" stroke="#18943F" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
  flame: `<svg viewBox="0 0 24 24"><path d="M12 2.5c.8 3 4.5 5 4.5 10a4.5 4.5 0 0 1-9 0c0-2.2 1-3.7 2.2-4.8.2 1.5.9 2.4 1.9 2.8C11.2 8 11.4 5 12 2.5z" fill="#18943F" stroke="#13723A" stroke-width="1.5" stroke-linejoin="round"/></svg>`,
  flameOff: line(`<path d="M12 2.5c.8 3 4.5 5 4.5 10a4.5 4.5 0 0 1-9 0c0-2.2 1-3.7 2.2-4.8.2 1.5.9 2.4 1.9 2.8C11.2 8 11.4 5 12 2.5z"/>`),
  bolt: line(`<path d="M13 2.5 4.5 13.5H12l-1 8 8.5-11H12z"/>`),
  check: line(`<path d="m5 12.5 4.5 4.5L19 7.5"/>`),
  cross: line(`<path d="M6 6l12 12M18 6 6 18"/>`),
  lock: line(`<rect x="5" y="10.5" width="14" height="10" rx="2.5"/><path d="M8.5 10.5V7.5a3.5 3.5 0 0 1 7 0v3"/>`),
  trophy: line(`<path d="M8 3.5h8v5a4 4 0 0 1-8 0zM8 5.5H5v1.5a3 3 0 0 0 3 3M16 5.5h3v1.5a3 3 0 0 1-3 3M12 12.5v4M8.5 20.5h7M10 16.5h4"/>`),
  gift: line(`<rect x="3.5" y="8.5" width="17" height="4" rx="1"/><path d="M5 12.5v7.5h14v-7.5M12 8.5V20M12 8.5S10.5 4 8.5 4.5 7.5 8.5 12 8.5zM12 8.5s1.5-4.5 3.5-4 1 4-3.5 4z"/>`),
  giftOpen: line(`<rect x="3.5" y="6" width="17" height="4" rx="1" transform="rotate(-8 12 8)"/><path d="M5 12.5v7.5h14v-7.5M12 12.5V20"/>`),
  notes: line(`<path d="M6 3.5h9l3.5 3.5v13.5H6z"/><path d="M14.5 3.5V7.5h4M9 12h6M9 15.5h6"/>`),
  path: line(`<circle cx="6" cy="5" r="2"/><circle cx="18" cy="12" r="2"/><circle cx="6" cy="19" r="2"/><path d="M8 5.5c5 0 8 1.5 8 4.5M16 14c0 3-3 4.5-8 4.5"/>`),
  dumbbell: line(`<path d="M3 10v4M6.5 7.5v9M17.5 7.5v9M21 10v4M6.5 12h11"/>`),
  podium: line(`<path d="M9 21V9h6v12M3 21v-7h6M15 21v-5h6v5M2 21h20"/>`),
  target: line(`<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r="1" fill="currentColor"/>`),
  store: line(`<path d="M4 9.5 5.5 4h13L20 9.5M4 9.5h16M4 9.5a2.7 2.7 0 0 0 5.3 0 2.7 2.7 0 0 0 5.3 0 2.7 2.7 0 0 0 5.4 0M5.5 12v8h13v-8M10 20v-5h4v5"/>`),
  user: line(`<circle cx="12" cy="8.5" r="4"/><path d="M4.5 20.5a7.5 7.5 0 0 1 15 0"/>`),
  play: `<svg viewBox="0 0 24 24"><path d="M8 5.5v13a1 1 0 0 0 1.5.9l10-6.5a1 1 0 0 0 0-1.7l-10-6.5A1 1 0 0 0 8 5.5z" fill="currentColor"/></svg>`,
  slow: line(`<path d="M3.5 16.5h13.5a3 3 0 0 0 0-6h-.5a6 6 0 0 0-11.5 2.5M20 10.5l1.5-1.5M7 16.5v2M14 16.5v2"/>`),
  camera: line(`<rect x="2.5" y="6.5" width="13" height="11" rx="2.5"/><path d="m15.5 10.5 6-3v9l-6-3z"/>`),
  translate: line(`<path d="M4 6h9M8.5 4v2M6 6c.5 3 2.5 5.5 5 6.5M11 6c-.5 3-2.5 5.5-6 7M13 20l3.5-8 3.5 8M14.2 17.5h4.6"/>`),
  saver: line(`<path d="M12 3 5 6v5.5c0 4.3 3 7.8 7 9.5 4-1.7 7-5.2 7-9.5V6z"/><path d="M12 8.5c.4 1.5 2.3 2.4 2.3 4.7a2.3 2.3 0 0 1-4.6 0c0-1.1.5-1.8 1.1-2.4.1.7.5 1.1.9 1.3.1-1.3.1-2.5.3-3.6z" fill="currentColor"/>`),
  retry: line(`<path d="M4 12a8 8 0 1 0 2.3-5.6M4 4v4.5h4.5"/>`),
  arrow: line(`<path d="M5 12h14M13 6l6 6-6 6"/>`),
  clock: line(`<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>`),
  spark: line(`<path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M18 6l-2.5 2.5M8.5 15.5 6 18"/>`),
};
