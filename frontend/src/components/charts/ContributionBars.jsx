import Plot from 'react-plotly.js';
import { getChannelColor, getChannelName, getControlColor, getControlName, PLOTLY_CONFIG, PLOTLY_LAYOUT_DEFAULTS } from '../../constants';

export default function ContributionBars({ channels, controls, showErrorBars = false }) {
  if ((!channels || channels.length === 0) && (!controls || controls.length === 0)) return null;

  const sorted = [...(channels || [])].sort((a, b) => b.contribution_pct - a.contribution_pct);
  const sortedControls = [...(controls || [])].sort((a, b) => b.contribution_pct - a.contribution_pct);

  const traces = [];

  // Channel trace
  if (sorted.length > 0) {
    const channelTrace = {
      type: 'bar',
      orientation: 'h',
      name: 'Paid Channels',
      y: sorted.map((c) => c.display_name || getChannelName(c.channel)),
      x: sorted.map((c) => c.contribution_pct),
      marker: {
        color: sorted.map((c) => getChannelColor(c.channel)),
      },
      text: sorted.map((c) => `${c.contribution_pct.toFixed(1)}%`),
      textposition: 'outside',
      hovertemplate: '%{y}: %{x:.1f}%<extra></extra>',
    };

    if (showErrorBars) {
      channelTrace.error_x = {
        type: 'data',
        symmetric: false,
        array: sorted.map((c) =>
          c.contribution_hdi_97 != null ? c.contribution_hdi_97 - c.contribution_pct : 0
        ),
        arrayminus: sorted.map((c) =>
          c.contribution_hdi_3 != null ? c.contribution_pct - c.contribution_hdi_3 : 0
        ),
        color: '#7a6f62',
        thickness: 1.5,
      };
    }

    traces.push(channelTrace);
  }

  // Control trace
  if (sortedControls.length > 0) {
    const controlTrace = {
      type: 'bar',
      orientation: 'h',
      name: 'Traffic Sources',
      y: sortedControls.map((c) => c.display_name || getControlName(c.control)),
      x: sortedControls.map((c) => c.contribution_pct),
      marker: {
        color: sortedControls.map((c) => getControlColor(c.control)),
        pattern: { shape: '/' },
      },
      text: sortedControls.map((c) => `${c.contribution_pct.toFixed(1)}%`),
      textposition: 'outside',
      hovertemplate: '%{y}: %{x:.1f}%<extra></extra>',
    };

    if (showErrorBars) {
      controlTrace.error_x = {
        type: 'data',
        symmetric: false,
        array: sortedControls.map((c) =>
          c.contribution_hdi_97 != null ? c.contribution_hdi_97 - c.contribution_pct : 0
        ),
        arrayminus: sortedControls.map((c) =>
          c.contribution_hdi_3 != null ? c.contribution_pct - c.contribution_hdi_3 : 0
        ),
        color: '#7a6f62',
        thickness: 1.5,
      };
    }

    traces.push(controlTrace);
  }

  const totalBars = sorted.length + sortedControls.length;

  return (
    <Plot
      data={traces}
      layout={{
        ...PLOTLY_LAYOUT_DEFAULTS,
        title: { text: 'Revenue Decomposition', font: { size: 14, family: 'Playfair Display' } },
        xaxis: { ...PLOTLY_LAYOUT_DEFAULTS.xaxis, title: '% of Revenue', ticksuffix: '%' },
        yaxis: { ...PLOTLY_LAYOUT_DEFAULTS.yaxis, automargin: true },
        margin: { l: 140, r: 60, t: 40, b: 40 },
        height: Math.max(240, totalBars * 50),
        barmode: 'group',
        showlegend: traces.length > 1,
        legend: { orientation: 'h', y: -0.15 },
      }}
      config={PLOTLY_CONFIG}
      style={{ width: '100%' }}
    />
  );
}
