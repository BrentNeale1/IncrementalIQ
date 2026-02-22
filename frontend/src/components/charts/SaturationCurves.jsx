import Plot from 'react-plotly.js';
import { getChannelColor, getChannelName, PLOTLY_CONFIG, PLOTLY_LAYOUT_DEFAULTS } from '../../constants';

export default function SaturationCurves({ saturationCurves }) {
  if (!saturationCurves || saturationCurves.length === 0) return null;

  const traces = saturationCurves.map((curve) => ({
    type: 'scatter',
    mode: 'lines',
    name: curve.display_name || getChannelName(curve.channel),
    x: curve.curve_x,
    y: curve.curve_y,
    line: { color: getChannelColor(curve.channel), width: 2 },
    hovertemplate: `Spend: %{x:,.0f}<br>Response: %{y:.3f}<extra>${curve.display_name || curve.channel}</extra>`,
  }));

  return (
    <Plot
      data={traces}
      layout={{
        ...PLOTLY_LAYOUT_DEFAULTS,
        title: { text: 'Saturation Response Curves', font: { size: 14, family: 'Playfair Display' } },
        xaxis: { ...PLOTLY_LAYOUT_DEFAULTS.xaxis, title: 'Spend' },
        yaxis: { ...PLOTLY_LAYOUT_DEFAULTS.yaxis, title: 'Saturated response' },
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
