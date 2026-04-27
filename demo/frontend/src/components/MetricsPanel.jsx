function diceClass(v) {
  if (v == null) return ''
  return v >= 0.8 ? 'good' : v >= 0.5 ? 'medium' : 'bad'
}

export default function MetricsPanel({ metrics }) {
  if (!metrics) return null
  const { per_class } = metrics

  return (
    <div className="metrics-panel">
      <div className="card-title">Metrics vs Ground Truth</div>

      <table className="dice-table">
        <thead>
          <tr>
            <th>Class</th>
            <th>Dice</th>
            <th style={{ width: 42, textAlign: 'right' }}>Score</th>
          </tr>
        </thead>
        <tbody>
          {per_class.map(c => {
            const pct = c.dice != null ? (c.dice * 100).toFixed(1) : null
            const dc  = diceClass(c.dice)
            return (
              <tr key={c.id}>
                <td>
                  <span style={{ display: 'inline-block', width: 10, height: 10, background: c.color, borderRadius: 2, marginRight: 7, verticalAlign: 'middle' }} />
                  {c.name}
                </td>
                <td>
                  {!c.has_gt
                    ? <span className="no-gt">not in GT</span>
                    : c.dice == null
                    ? <span className="no-gt">not predicted</span>
                    : (
                      <div className="bar-wrap">
                        <div className="bar-bg">
                          <div className="bar-fill" style={{ width: `${pct}%`, background: c.color, opacity: .85 }} />
                        </div>
                      </div>
                    )
                  }
                </td>
                <td>
                  {c.has_gt && c.dice != null &&
                    <span className={`dice-num ${dc}`}>{pct}%</span>
                  }
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
