import { useEffect, useState } from 'react'
import { fetchDatasets, fetchPalette, fetchPresets, predict } from './api'

import Header      from './components/Header'
import DatasetBar  from './components/DatasetBar'
import InputCard   from './components/InputCard'
import Legend      from './components/Legend'
import ModelInfo   from './components/ModelInfo'
import ViewPanel   from './components/ViewPanel'

export default function App() {
  const [datasets,       setDatasets]       = useState([])
  const [selectedDs,     setSelectedDs]     = useState(null)
  const [palette,        setPalette]        = useState([])
  const [presets,        setPresets]        = useState([])

  const [inputMode,      setInputMode]      = useState('upload')  // 'upload' | 'preset'
  const [selectedFile,   setSelectedFile]   = useState(null)
  const [selectedPreset, setSelectedPreset] = useState(null)
  const [sliceIdx,       setSliceIdx]       = useState(100)

  const [result,         setResult]         = useState(null)
  const [loading,        setLoading]        = useState(false)
  const [status,         setStatus]         = useState(null)

  // ── Load datasets on mount ───────────────────────────────
  useEffect(() => {
    fetchDatasets().then(data => {
      setDatasets(data)
      if (data.length) selectDataset(data[0], data)
    })
  }, [])

  // ── Load presets once ────────────────────────────────────
  useEffect(() => {
    fetchPresets().then(setPresets)
  }, [])

  // ── Select dataset ───────────────────────────────────────
  function selectDataset(ds) {
    setSelectedDs(ds)
    setResult(null)
    setStatus(null)
    setInputMode('upload')
    setSelectedFile(null)
    setSelectedPreset(null)
    fetchPalette(ds.key).then(setPalette)
  }

  // ── Change input mode ────────────────────────────────────
  function handleModeChange(mode) {
    setInputMode(mode)
    setSelectedFile(null)
    setSelectedPreset(null)
    setStatus(null)
  }

  // ── Run inference ────────────────────────────────────────
  async function handleRun() {
    if (!selectedDs) return
    setLoading(true)
    setStatus({ type: 'loading', msg: 'Running inference…' })
    const t0 = performance.now()
    try {
      const data = await predict({
        dataset:      selectedDs.key,
        file:         inputMode === 'upload' ? selectedFile : null,
        presetSlice:  inputMode === 'preset' ? selectedPreset : null,
        slice:        inputMode === 'upload' && selectedFile?.name?.endsWith('.npz')
                        ? sliceIdx : null,
      })
      const secs = ((performance.now() - t0) / 1000).toFixed(2)
      if (data.error) {
        setStatus({ type: 'error', msg: `⚠ ${data.error}` })
      } else {
        setResult(data)
        setStatus({
          type: 'success',
          msg: `✓ Done · ${secs}s`,
        })
      }
    } catch (e) {
      setStatus({ type: 'error', msg: `⚠ ${e.message}` })
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <Header dataset={selectedDs} />
      <DatasetBar datasets={datasets} selected={selectedDs} onSelect={selectDataset} />

      <div className="layout">
        <aside className="sidebar">
          <InputCard
            dataset={selectedDs}
            inputMode={inputMode}        onModeChange={handleModeChange}
            selectedFile={selectedFile}  onFileChange={f => { setSelectedFile(f); setStatus(null) }}
            selectedPreset={selectedPreset} onPresetSelect={s => { setSelectedPreset(s); setStatus(null) }}
            presets={presets}
            sliceIdx={sliceIdx}          onSliceChange={setSliceIdx}
            onRun={handleRun}
            loading={loading}
            status={status}
          />
          <Legend palette={palette} result={result} />
          <ModelInfo dataset={selectedDs} device={result?.device} />
        </aside>

        <section>
          <ViewPanel result={result} />
        </section>
      </div>
    </>
  )
}
