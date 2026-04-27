import MetricsPanel from './MetricsPanel'

const b64 = src => `data:image/png;base64,${src}`

function ImgPanel({ title, src }) {
  return (
    <div className="img-panel">
      <div className="panel-title">{title}</div>
      <img src={src} alt={title} />
    </div>
  )
}

export default function ViewPanel({ result }) {
  if (!result) {
    return (
      <div className="results">
        <div className="placeholder">
          <svg width="60" height="60" fill="none" stroke="currentColor" strokeWidth="1.2" viewBox="0 0 24 24">
            <rect x="3" y="3" width="18" height="18" rx="2"/>
            <circle cx="8.5" cy="8.5" r="1.5"/>
            <path strokeLinecap="round" strokeLinejoin="round" d="M21 15l-5-5L5 21"/>
          </svg>
          Select a dataset, upload an image, then click Run
        </div>
      </div>
    )
  }

  const cols3 = result.gt_overlay ? 'cols3' : ''

  return (
    <div className="results">
      <div className={`img-grid ${cols3}`}>
        <ImgPanel title="Original"     src={b64(result.original)} />
        <ImgPanel title="Segmentation" src={b64(result.mask ?? result.overlay)} />
        {result.gt_overlay &&
          <ImgPanel title="Ground Truth" src={b64(result.gt_overlay)} />
        }
      </div>

      {result.metrics && <MetricsPanel metrics={result.metrics} />}
    </div>
  )
}
