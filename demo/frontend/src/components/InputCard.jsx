import { useRef, useState } from 'react'

export default function InputCard({
  dataset, inputMode, onModeChange,
  selectedFile, onFileChange,
  selectedPreset, onPresetSelect,
  presets, sliceIdx, onSliceChange,
  onRun, loading, status,
}) {
  const fileRef   = useRef()
  const [drag, setDrag] = useState(false)
  const isSynapse = dataset?.key === 'synapse'
  const isNpz     = selectedFile?.name?.toLowerCase().endsWith('.npz')
  const canRun    = !loading && (
    inputMode === 'preset' ? selectedPreset != null : selectedFile != null
  )

  const hint = dataset?.binary
    ? 'PNG · JPG (skin lesion photo)'
    : 'PNG · JPG · NPZ (CT slice / volume)'

  function onDrop(e) {
    e.preventDefault(); setDrag(false)
    if (e.dataTransfer.files[0]) onFileChange(e.dataTransfer.files[0])
  }

  return (
    <div className="card">
      <div className="card-title">Input Image</div>

      {/* Mode toggle — Synapse only */}
      {isSynapse && (
        <div className="mode-toggle">
          <button className={`mode-btn ${inputMode === 'upload' ? 'active' : ''}`}
                  onClick={() => onModeChange('upload')}>Upload</button>
          <button className={`mode-btn ${inputMode === 'preset' ? 'active' : ''}`}
                  onClick={() => onModeChange('preset')}>Preset</button>
        </div>
      )}

      {/* Upload area */}
      {inputMode === 'upload' && (
        <>
          <div
            className={`drop-zone ${drag ? 'drag-over' : ''}`}
            onClick={() => fileRef.current.click()}
            onDragOver={e => { e.preventDefault(); setDrag(true) }}
            onDragLeave={() => setDrag(false)}
            onDrop={onDrop}
          >
            <svg width="34" height="34" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"/>
            </svg>
            <p>Drop or <strong>click</strong> to upload</p>
            <p className="hint">{hint}</p>
          </div>
          <input ref={fileRef} type="file" style={{ display: 'none' }}
                 accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff,.npz"
                 onChange={e => e.target.files[0] && onFileChange(e.target.files[0])} />

          {selectedFile && (
            <div className="file-info">File: <span>{selectedFile.name}</span></div>
          )}

          {isNpz && (
            <div className="slice-row" style={{ marginTop: 11 }}>
              <label>Slice: <strong>{sliceIdx}</strong></label>
              <input type="range" min="0" max="200" value={sliceIdx}
                     onChange={e => onSliceChange(+e.target.value)} />
            </div>
          )}
        </>
      )}

      {/* Preset area */}
      {inputMode === 'preset' && (
        <div className="preset-grid">
          {presets.length === 0 && (
            <div style={{ color: 'var(--muted)', fontSize: 13 }}>Loading…</div>
          )}
          {presets.map(p => (
            <div
              key={p.slice}
              className={`preset-card ${selectedPreset === p.slice ? 'active' : ''}`}
              onClick={() => onPresetSelect(p.slice)}
            >
              <img src={`data:image/png;base64,${p.thumb}`} alt={p.label} />
              <div className="info">
                <div className="name">Slice {p.slice}</div>
              </div>
              <div className="radio" />
            </div>
          ))}
        </div>
      )}

      <button className="btn-run" disabled={!canRun} onClick={onRun}>
        {loading ? <><span className="spinner" />Running…</> : 'Run Segmentation'}
      </button>

      {status && (
        <div className={`status ${status.type}`}>{status.msg}</div>
      )}
    </div>
  )
}
