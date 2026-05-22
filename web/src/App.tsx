import { useMemo, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

type ScoreMap = Record<string, number | boolean>;

type FileResult = {
  filename: string;
  winning_model: string;
  winning_metric: string;
  summary_sentences: string[];
  metadata: {
    description: string;
    functions: Array<Record<string, unknown>>;
    classes: Array<Record<string, unknown>>;
    imports: string[];
  };
  rouge_scores: Record<string, ScoreMap>;
  bertscore_scores: Record<string, ScoreMap> | null;
  coverage_scores: Record<string, ScoreMap>;
  readme: string;
};

type ApiResponse = {
  uploaded_count: number;
  metric: string;
  top_n: number;
  files: FileResult[];
  warnings: string[];
  max_files: number;
  error?: string;
};

const MAX_FILES = 10;
const METRICS = [
  { value: 'rougeL', label: 'ROUGE-L' },
  { value: 'rouge1', label: 'ROUGE-1' },
  { value: 'rouge2', label: 'ROUGE-2' },
  { value: 'bertscore_f1', label: 'BERTScore F1' },
  { value: 'coverage', label: 'Coverage' },
  { value: 'diversity', label: 'Diversity' },
  { value: 'combined', label: 'Combined' },
];

function formatScore(value: number | boolean | undefined): string {
  if (typeof value !== 'number') {
    return '—';
  }
  return value.toFixed(4);
}

function scoreLabel(value: number | boolean | undefined): string {
  if (typeof value !== 'number') {
    return 'N/A';
  }
  return value.toFixed(3);
}

function scoreFor(entry: Record<string, ScoreMap> | null | undefined, key: string): number | boolean | undefined {
  const model = entry ? entry[key] : undefined;
  const score = model && typeof model === 'object' ? model : undefined;
  return score ? (score[key] as number | boolean | undefined) : undefined;
}

export default function App() {
  const [files, setFiles] = useState<File[]>([]);
  const [referenceFiles, setReferenceFiles] = useState<File[]>([]);
  const [metric, setMetric] = useState('rougeL');
  const [topN, setTopN] = useState(2);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [expandedFiles, setExpandedFiles] = useState<string[]>([]);

  const fileSummary = useMemo(() => {
    if (files.length === 0) {
      return 'No files selected yet.';
    }
    if (files.length === 1) {
      return `${files[0].name} is ready to analyze.`;
    }
    return `${files.length} Python files queued for a batch run.`;
  }, [files]);

  const referenceSummary = useMemo(() => {
    if (referenceFiles.length === 0) {
      return 'Optional: upload hand-written reference summaries as .txt files.';
    }
    if (referenceFiles.length === 1) {
      return `${referenceFiles[0].name} will be matched by exact filename or stem.`;
    }
    return `${referenceFiles.length} reference files ready for filename matching.`;
  }, [referenceFiles]);

  function mergeFiles(nextFiles: File[]) {
    const combined = [...files];

    for (const file of nextFiles) {
      if (!file.name.endsWith('.py')) {
        setError('Only Python (.py) files are supported.');
        return;
      }

      const duplicate = combined.some(
        (existing) =>
          existing.name === file.name &&
          existing.size === file.size &&
          existing.lastModified === file.lastModified,
      );

      if (!duplicate) {
        combined.push(file);
      }
    }

    if (combined.length > MAX_FILES) {
      setError(`Select up to ${MAX_FILES} files.`);
      return;
    }

    setError('');
    setFiles(combined);
  }

  function handleFileInput(event: React.ChangeEvent<HTMLInputElement>) {
    const selected = Array.from(event.target.files ?? []);
    if (selected.length > 0) {
      mergeFiles(selected);
    }
    event.target.value = '';
  }

  function handleDrop(event: React.DragEvent<HTMLDivElement>) {
    event.preventDefault();
    mergeFiles(Array.from(event.dataTransfer.files));
  }

  function removeFile(name: string) {
    setFiles((current) => current.filter((file) => file.name !== name));
  }

  function mergeReferenceFiles(nextFiles: File[]) {
    const combined = [...referenceFiles];

    for (const file of nextFiles) {
      if (!file.name.endsWith('.txt') && !file.name.endsWith('.md')) {
        setError('Reference files must be .txt or .md documents.');
        return;
      }

      const duplicate = combined.some(
        (existing) =>
          existing.name === file.name &&
          existing.size === file.size &&
          existing.lastModified === file.lastModified,
      );

      if (!duplicate) {
        combined.push(file);
      }
    }

    setError('');
    setReferenceFiles(combined);
  }

  function handleReferenceInput(event: React.ChangeEvent<HTMLInputElement>) {
    const selected = Array.from(event.target.files ?? []);
    if (selected.length > 0) {
      mergeReferenceFiles(selected);
    }
    event.target.value = '';
  }

  function removeReference(name: string) {
    setReferenceFiles((current) => current.filter((file) => file.name !== name));
  }

  async function handleAnalyze() {
    if (files.length === 0) {
      setError('Add at least one Python file.');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const formData = new FormData();
      files.forEach((file) => formData.append('files', file));
      referenceFiles.forEach((file) => formData.append('references', file));
      formData.append('metric', metric);
      formData.append('top_n', String(topN));

      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });

      const rawBody = await response.text();
      let data: ApiResponse | null = null;

      if (rawBody) {
        try {
          data = JSON.parse(rawBody) as ApiResponse;
        } catch {
          data = null;
        }
      }

      if (!response.ok) {
        const detail = data?.error ?? rawBody.trim();
        if (!detail && response.status === 502) {
          throw new Error('The API server is not running. Start it with ./dev.sh or source ../.venv/bin/activate && python server.py.');
        }
        throw new Error(detail || 'The analysis request failed.');
      }

      if (!data) {
        throw new Error('The API returned an empty response.');
      }

      setResult(data);
    } catch (analysisError) {
      if (analysisError instanceof TypeError && analysisError.message.includes('fetch')) {
        setError('Cannot reach the API server. Start it with ./dev.sh or source ../.venv/bin/activate && python server.py.');
      } else {
        setError(analysisError instanceof Error ? analysisError.message : 'Something went wrong.');
      }
    } finally {
      setLoading(false);
    }
  }

  function downloadMarkdown(fileResult: FileResult) {
    const blob = new Blob([fileResult.readme], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = fileResult.filename.replace('.py', '-README.md');
    anchor.click();
    URL.revokeObjectURL(url);
  }

  function copyMarkdown(fileResult: FileResult) {
    void navigator.clipboard.writeText(fileResult.readme);
  }

  function toggleFile(filename: string) {
    setExpandedFiles((current) =>
      current.includes(filename)
        ? current.filter((entry) => entry !== filename)
        : [...current, filename],
    );
  }

  return (
    <div className="shell">
      <div className="ambient ambient-a" />
      <div className="ambient ambient-b" />

      <main className="layout">
        <section className="hero card">
          <div className="eyebrow">Auto README Studio</div>
          <h1>Turn Python examples into README files in one batch.</h1>
          <p className="lede">
            Upload up to 10 Python files, compare the local TF-IDF, BM25, and LSA summarizers,
            and preview a generated README for each file.
          </p>

          <div className="hero-grid">
            <div className="hero-stat">
              <span>Files queued</span>
              <strong>{files.length}</strong>
            </div>
            <div className="hero-stat">
              <span>Selected metric</span>
              <strong>{METRICS.find((entry) => entry.value === metric)?.label ?? metric}</strong>
            </div>
            <div className="hero-stat">
              <span>Summary length</span>
              <strong>{topN} sentences</strong>
            </div>
          </div>
        </section>

        <section className="workspace">
          <div className="card uploader" onDragOver={(event) => event.preventDefault()} onDrop={handleDrop}>
            <div>
              <h2>Upload example Python files</h2>
              <p>{fileSummary}</p>
            </div>

            <label className="dropzone">
              <input type="file" multiple accept=".py,text/x-python" onChange={handleFileInput} />
              <span>Choose files or drag and drop them here</span>
              <small>Up to {MAX_FILES} files, analyzed together as one demo batch.</small>
            </label>

            <label className="dropzone reference-zone">
              <input type="file" multiple accept=".txt,.md,text/plain" onChange={handleReferenceInput} />
              <span>Optional reference summaries</span>
              <small>Upload .txt or .md files that match each Python file by exact name or stem, like scaler.txt.</small>
            </label>

            {referenceFiles.length > 0 ? (
              <div className="file-list reference-list">
                {referenceFiles.map((file) => (
                  <div key={`${file.name}-${file.size}-${file.lastModified}`} className="file-pill reference-pill">
                    <span>{file.name}</span>
                    <button type="button" onClick={() => removeReference(file.name)}>
                      Remove
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <p className="reference-hint">{referenceSummary}</p>
            )}

            <div className="controls">
              <label>
                <span>Winning metric</span>
                <select value={metric} onChange={(event) => setMetric(event.target.value)}>
                  {METRICS.map((entry) => (
                    <option key={entry.value} value={entry.value}>
                      {entry.label}
                    </option>
                  ))}
                </select>
              </label>

              <label>
                <span>Summary sentences</span>
                <input
                  type="range"
                  min="1"
                  max="5"
                  value={topN}
                  onChange={(event) => setTopN(Number(event.target.value))}
                />
                <strong>{topN}</strong>
              </label>

              <button className="primary" type="button" onClick={handleAnalyze} disabled={loading}>
                {loading ? 'Analyzing files…' : 'Generate READMEs'}
              </button>
            </div>

            {error ? <div className="notice error">{error}</div> : null}

            {files.length > 0 ? (
              <div className="file-list">
                {files.map((file) => (
                  <div key={`${file.name}-${file.size}-${file.lastModified}`} className="file-pill">
                    <span>{file.name}</span>
                    <button type="button" onClick={() => removeFile(file.name)}>
                      Remove
                    </button>
                  </div>
                ))}
              </div>
            ) : null}
          </div>

          {result ? (
            <section className="results">
              <div className="card result-summary">
                <div>
                  <h2>Batch completed</h2>
                  <p>
                    Generated {result.uploaded_count} README file{result.uploaded_count === 1 ? '' : 's'}
                    with {METRICS.find((entry) => entry.value === result.metric)?.label ?? result.metric} as the selection metric.
                  </p>
                </div>

                <div className="summary-grid">
                  <div>
                    <span>Uploaded</span>
                    <strong>{result.uploaded_count}</strong>
                  </div>
                  <div>
                    <span>Top N</span>
                    <strong>{result.top_n}</strong>
                  </div>
                  <div>
                    <span>Warnings</span>
                    <strong>{result.warnings.length}</strong>
                  </div>
                </div>
              </div>

              {result.files.map((fileResult) => (
                <article key={fileResult.filename} className={`card file-result${expandedFiles.includes(fileResult.filename) ? ' is-open' : ''}`}>
                  <button
                    type="button"
                    className="file-toggle"
                    onClick={() => toggleFile(fileResult.filename)}
                    aria-expanded={expandedFiles.includes(fileResult.filename)}
                  >
                    <div className="file-toggle-text">
                      <h3>{fileResult.filename}</h3>
                      <p>
                        Winner: <strong>{fileResult.winning_model.toUpperCase()}</strong>
                        <span> · </span>
                        {fileResult.summary_sentences.length} selected sentence{fileResult.summary_sentences.length === 1 ? '' : 's'}
                      </p>
                    </div>

                    <div className="file-toggle-meta">
                      <span>{expandedFiles.includes(fileResult.filename) ? 'Hide details' : 'Show details'}</span>
                      <strong>{formatScore(fileResult.rouge_scores[fileResult.winning_model]?.rougeL)}</strong>
                    </div>
                  </button>

                  {expandedFiles.includes(fileResult.filename) ? (
                    <div className="file-body">
                      <div className="metric-row">
                        <div>
                          <span>ROUGE-L</span>
                          <strong>{formatScore(fileResult.rouge_scores[fileResult.winning_model]?.rougeL)}</strong>
                        </div>
                        <div>
                          <span>Coverage</span>
                          <strong>{formatScore(fileResult.coverage_scores[fileResult.winning_model]?.combined)}</strong>
                        </div>
                        <div>
                          <span>Sentence count</span>
                          <strong>{fileResult.summary_sentences.length}</strong>
                        </div>
                        <div>
                          <span>Classes</span>
                          <strong>{fileResult.metadata.classes.length}</strong>
                        </div>
                      </div>

                      <div className="actions file-actions">
                        <button type="button" onClick={() => copyMarkdown(fileResult)}>
                          Copy README
                        </button>
                        <button type="button" onClick={() => downloadMarkdown(fileResult)}>
                          Download README
                        </button>
                      </div>

                      <div className="summary-block">
                        <h4>Chosen summary sentences</h4>
                        <ul>
                          {fileResult.summary_sentences.map((sentence) => (
                            <li key={sentence}>{sentence}</li>
                          ))}
                        </ul>
                      </div>

                      <div className="markdown-panel">
                        <h4>README preview</h4>
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{fileResult.readme}</ReactMarkdown>
                      </div>
                    </div>
                  ) : null}
                </article>
              ))}
            </section>
          ) : (
            <div className="card empty-state">
              <h2>What happens after upload</h2>
              <p>
                The backend trains the local ranking models on the uploaded corpus, summarizes each
                file, and returns a complete README with model comparisons and usage guidance.
              </p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}