export default function DatasetBar({ datasets, selected, onSelect }) {
  return (
    <div className="ds-bar">
      {datasets.map(ds => (
        <button
          key={ds.key}
          className={`ds-btn ${selected?.key === ds.key ? 'active' : ''}`}
          onClick={() => onSelect(ds)}
        >
          <span>{ds.label}</span>
        </button>
      ))}
    </div>
  )
}
