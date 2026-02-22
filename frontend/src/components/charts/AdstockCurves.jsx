import Plot from 'react-plotly.js';
import { getChannelColor, getChannelName, PLOTLY_CONFIG, PLOTLY_LAYOUT_DEFAULTS } from '../../constants';

export default function AdstockCurves({ adstockCurves }) {
  if (!adstockCurves || adstockCurves.length === 0) return null;

  const traces = adstockCurves.map((curve) => ({
    type: 'scatter',
    mode: 'lines',
    name: curve.display_name || getChannelName(curve.channel),
    x: curve.decay_curve.map((_, i) => i),
    y: curve.decay_curve,
    line: { color: getChannelColor(curve.channel), width: 2 },
    hovertemplate: `Day %{x}: %{y:.3f}<extra>${curve.display_name || curve.channel}</extra>`,
  }));

  return (
    <Plot
      data={traces}
      layout={{
        ...PLOTLY_LAYOUT_DEFAULTS,
        title: { text: 'Adstock Decay Curves', font: { size: 14, family: 'Playfair Display' } },
        xaxis: { ...PLOTLY_LAYOUT_DEFAULTS.xaxis, title: 'Days after spend' },
        yaxis: { ...PLOTLY_LAYOUT_DEFAULTS.yaxis, title: 'Weight', range: [0, 1.05] },
        height: 300,
        showlegend: true,
        legend: { orientation: 'h', y: -0.2, font: { size: 11 } },
        margin: { l: 50, r: 20, t: 40, b: 60 },
      }}
      config={PLOTLY_CONFIG}
      style={{ width: '100%' }}
    />
  );
}
