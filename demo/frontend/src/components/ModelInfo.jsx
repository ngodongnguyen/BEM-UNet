export default function ModelInfo({ dataset, device }) {
  const rows = [
    ['Architecture', 'BEM-UNet'],
    ['Backbone',     'VMamba-S'],
    ['Input size',   dataset?.binary ? '256 × 256' : '224 × 224'],
    ['Classes',      dataset?.binary ? '2 (lesion / bg)' : '9 organs'],
    ['Device',       device ?? '–'],
  ]
  return (
    <div className="card">
      <div className="card-title">Model Info</div>
      {rows.map(([k, v]) => (
        <div key={k} className="stat-row">
          <span>{k}</span>
          <span className="val">{v}</span>
        </div>
      ))}
    </div>
  )
}
