export const CHANNEL_COLORS = {
  google_search: '#1e5c3a',
  google_shopping: '#c2650a',
  google_pmax: '#2563eb',
  google_youtube: '#b8860b',
  meta_feed: '#7c3aed',
  meta_instagram: '#9333ea',
  meta_stories: '#a855f7',
};

export const CHANNEL_DISPLAY_NAMES = {
  google_search: 'Google Search',
  google_shopping: 'Google Shopping',
  google_pmax: 'Google PMax',
  google_youtube: 'YouTube',
  meta_feed: 'Meta Feed',
  meta_instagram: 'Meta Instagram',
  meta_stories: 'Meta Stories',
};

export const CONTROL_COLORS = {
  sessions_organic: '#2d6a4f',
  sessions_direct: '#457b9d',
  sessions_email: '#e76f51',
  sessions_referral: '#8338ec',
};

export const CONTROL_DISPLAY_NAMES = {
  sessions_organic: 'Organic Sessions',
  sessions_direct: 'Direct Sessions',
  sessions_email: 'Email Sessions',
  sessions_referral: 'Referral Sessions',
};

export function getControlColor(control) {
  return CONTROL_COLORS[control] || '#7a6f62';
}

export function getControlName(control) {
  return CONTROL_DISPLAY_NAMES[control] || control.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

export const TRUST_TIERS = {
  reliable: {
    label: 'Model results are reliable',
    shortLabel: 'Reliable',
    color: '#1e5c3a',
    bg: '#e6f2eb',
  },
  caution: {
    label: 'Use with caution',
    shortLabel: 'Caution',
    color: '#7a4f00',
    bg: '#fef3c7',
  },
  insufficient: {
    label: 'Insufficient data',
    shortLabel: 'Insufficient',
    color: '#8b1e1e',
    bg: '#fde8e8',
  },
};

export function getTrustTier(tierString) {
  if (tierString === 'Model results are reliable') return TRUST_TIERS.reliable;
  if (tierString === 'Use with caution') return TRUST_TIERS.caution;
  return TRUST_TIERS.insufficient;
}

export function getChannelColor(channel) {
  return CHANNEL_COLORS[channel] || '#7a6f62';
}

export function getChannelName(channel) {
  return CHANNEL_DISPLAY_NAMES[channel] || channel.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

export const CAVEAT_TEXT = 'Observational causal inference. Not RCT-level certainty.';

export const PLOTLY_CONFIG = {
  displayModeBar: false,
  responsive: true,
};

export const PLOTLY_LAYOUT_DEFAULTS = {
  font: { family: 'DM Sans, sans-serif', color: '#18130e' },
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',
  margin: { l: 50, r: 20, t: 30, b: 40 },
  xaxis: { gridcolor: '#e0d9cc', zerolinecolor: '#e0d9cc' },
  yaxis: { gridcolor: '#e0d9cc', zerolinecolor: '#e0d9cc' },
};
