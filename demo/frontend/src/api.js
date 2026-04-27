const BASE = '/api'

export const fetchDatasets = () =>
  fetch(`${BASE}/datasets`).then(r => r.json())

export const fetchPalette = (dataset) =>
  fetch(`${BASE}/palette?dataset=${dataset}`).then(r => r.json())

export const fetchPresets = () =>
  fetch(`${BASE}/presets`).then(r => r.json())

export const predict = async ({ dataset, file, presetSlice, slice }) => {
  const fd = new FormData()
  fd.append('dataset', dataset)
  if (presetSlice != null) {
    fd.append('preset_slice', presetSlice)
  } else {
    fd.append('file', file)
    if (slice != null) fd.append('slice', slice)
  }
  const res = await fetch(`${BASE}/predict`, { method: 'POST', body: fd })
  return res.json()
}
