export default function Legend({ palette, result }) {
  const pctMap = {}
  result?.classes?.forEach(c => { pctMap[c.id] = c.percent })

  return (
    <div className="card">
      <div className="card-title">Legend</div>
      {palette.length === 0
        ? <p style={{ color: 'var(--muted)', fontSize: 13 }}>Select a dataset</p>
        : (
          <ul className="legend-list">
            {palette.map(c => (
              <li key={c.id} className="legend-item">
                <span className="swatch" style={{ background: c.color }} />
                <span>{c.name}</span>
                {pctMap[c.id] != null && (
                  <span className="legend-pct">{pctMap[c.id]}%</span>
                )}
              </li>
            ))}
          </ul>
        )
      }
    </div>
  )
}
