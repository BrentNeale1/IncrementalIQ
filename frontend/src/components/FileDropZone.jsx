import { useState, useRef } from 'react';

export default function FileDropZone({ onFile, accept = '.csv', label }) {
  const [active, setActive] = useState(false);
  const [fileName, setFileName] = useState(null);
  const inputRef = useRef();

  function handleDrop(e) {
    e.preventDefault();
    setActive(false);
    const file = e.dataTransfer.files[0];
    if (file) {
      setFileName(file.name);
      onFile(file);
    }
  }

  function handleChange(e) {
    const file = e.target.files[0];
    if (file) {
      setFileName(file.name);
      onFile(file);
    }
  }

  return (
    <div
      className={`dropzone${active ? ' active' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setActive(true); }}
      onDragLeave={() => setActive(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        onChange={handleChange}
        style={{ display: 'none' }}
      />
      {fileName ? (
        <div className="dropzone-label">{fileName}</div>
      ) : (
        <>
          <div className="dropzone-label">
            {label || 'Drop a CSV file here, or click to browse'}
          </div>
          <div className="dropzone-hint">
            Requires: date, channel, campaign, spend, impressions, clicks,
            in_platform_conversions, revenue, orders, sessions_*
          </div>
        </>
      )}
    </div>
  );
}
