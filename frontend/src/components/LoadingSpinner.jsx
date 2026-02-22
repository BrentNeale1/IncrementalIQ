export default function LoadingSpinner({ text = 'Loading...' }) {
  return (
    <div className="spinner-container">
      <div className="spinner" />
      <div className="spinner-text">{text}</div>
    </div>
  );
}
