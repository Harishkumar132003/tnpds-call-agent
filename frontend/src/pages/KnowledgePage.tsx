import { useEffect, useRef, useState, type FormEvent } from 'react';
import { Empty } from '../components/Empty';
import { Icon } from '../components/Icon';
import { Topbar } from '../components/Topbar';
import { useToast } from '../components/Toast';
import { api } from '../api';
import { when } from '../format';
import type { KnowledgeDocument, KnowledgeSearch, KnowledgeStatus } from '../types';

const kb = (n: number | null) => (n === null ? '—' : n < 1024 ? `${n} B` : n < 1024 * 1024 ? `${(n / 1024).toFixed(0)} KB` : `${(n / 1024 / 1024).toFixed(1)} MB`);

export function KnowledgePage() {
  const toast = useToast();
  const [status, setStatus] = useState<KnowledgeStatus | null>(null);
  const [docs, setDocs] = useState<KnowledgeDocument[]>([]);
  const [uploading, setUploading] = useState(false);
  const [question, setQuestion] = useState('');
  const [scope, setScope] = useState('');
  const [searching, setSearching] = useState(false);
  const [result, setResult] = useState<KnowledgeSearch | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const load = async () => {
    const [s, d] = await Promise.all([api.knowledgeStatus(), api.documents()]);
    setStatus(s);
    setDocs(d);
  };
  useEffect(() => { load().catch((e) => toast.show({ tone: 'error', title: 'Could not load knowledge base', body: String(e) })); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const upload = async (file: File) => {
    setUploading(true);
    try {
      const doc = await api.uploadDocument(file);
      toast.show({ tone: 'success', title: `Indexed ${doc.name}`, body: `${doc.chunk_count} chunks from ${doc.chars?.toLocaleString()} characters.` });
      await load();
    } catch (e) {
      toast.show({ tone: 'error', title: 'Upload failed', body: (e as Error).message });
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = '';
    }
  };

  const remove = async (doc: KnowledgeDocument) => {
    if (!window.confirm(`Delete "${doc.name}" and its ${doc.chunk_count ?? 0} chunks?`)) return;
    try {
      await api.deleteDocument(doc.id);
      await load();
    } catch (e) {
      toast.show({ tone: 'error', title: 'Delete failed', body: (e as Error).message });
    }
  };

  const search = async (e: FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;
    setSearching(true);
    try {
      setResult(await api.knowledgeSearch(question.trim(), 5, scope || undefined));
    } catch (err) {
      toast.show({ tone: 'error', title: 'Search failed', body: (err as Error).message });
    } finally {
      setSearching(false);
    }
  };

  const ready = status?.qdrant_ok && status?.embedding_key_set;
  const accept = status?.supported_extensions.join(',') ?? '.pdf,.docx,.txt,.md,.csv';

  return (
    <>
      <Topbar
        title="Knowledge"
        actions={status && (
          <>
            <span className={`pill ${status.qdrant_ok ? 'pill--up' : 'pill--down'}`}>Qdrant {status.qdrant_ok ? 'connected' : 'unreachable'}</span>
            <span className={`pill ${status.embedding_key_set ? 'pill--up' : 'pill--warn'}`}>{status.embedding_key_set ? status.embedding_model : 'OPENAI_API_KEY missing'}</span>
          </>
        )}
      />
      <div className="page stack">
        {status && !status.embedding_key_set && (
          <div className="banner banner--warn">Set OPENAI_API_KEY in backend/.env and restart the portal to enable uploads and retrieval tests. Embeddings use OpenAI's multilingual {status.embedding_model}.</div>
        )}
        {status && !status.qdrant_ok && (
          <div className="banner banner--error">Qdrant is not reachable at {status.qdrant_url}.</div>
        )}

        <div className="grid grid--main">
          <div className="stack">
            <section className="card">
              <div className="card__head">
                <span className="card__label">Documents</span>
                <span className="topbar__spacer" />
                <span className="card__note">collection <span className="mono">{status?.collection ?? '…'}</span>{status?.points !== null && status?.points !== undefined ? ` · ${status.points.toLocaleString()} chunks` : ''}</span>
              </div>
              <div className="card__body">
                <label className={`dropzone${uploading ? ' is-busy' : ''}`}>
                  <input ref={fileInput} type="file" accept={accept} disabled={!ready || uploading} style={{ display: 'none' }} onChange={(e) => { const f = e.target.files?.[0]; if (f) void upload(f); }} />
                  <span className="dropzone__glyph"><Icon name="upload" size={18} /></span>
                  <span className="dropzone__title">{uploading ? 'Extracting, chunking and embedding…' : 'Upload a document'}</span>
                  <span className="dropzone__hint">{accept.replaceAll(',', ' · ')} · text is split into ~1,200-character chunks with 200 overlap</span>
                </label>
              </div>
              {docs.length === 0 ? (
                <Empty icon="file" title="No documents yet" body="Upload a PDF, Word, text, Markdown or CSV file to index it." />
              ) : (
                <div className="table-wrap">
                  <table className="table">
                    <thead><tr><th>Name</th><th>Uploaded</th><th>Size</th><th>Chars</th><th>Chunks</th><th>Status</th><th></th></tr></thead>
                    <tbody>
                      {docs.map((d) => (
                        <tr key={d.id}>
                          <td><span className="cell-link">{d.name}</span><span className="cell-sub mono">{d.embedding_model}</span></td>
                          <td>{when(d.created_at)}</td>
                          <td>{kb(d.size_bytes)}</td>
                          <td>{d.chars?.toLocaleString() ?? '—'}</td>
                          <td>{d.chunk_count ?? '—'}</td>
                          <td>{d.status === 'ready' ? <span className="pill pill--up">ready</span> : d.status === 'failed' ? <span className="pill pill--down" title={d.error ?? ''}>failed</span> : <span className="pill pill--info">{d.status}</span>}</td>
                          <td><button type="button" className="btn btn--ghost btn--sm" onClick={() => remove(d)}>Delete</button></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
          </div>

          <section className="card">
            <div className="card__head"><span className="card__label">Test retrieval</span></div>
            <div className="card__body">
              <form onSubmit={search} className="stack">
                <div className="field">
                  <label className="field__label" htmlFor="q">Sample question <span className="muted">(any language)</span></label>
                  <textarea id="q" className="field__textarea lang-select" rows={3} value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="e.g. ரேஷன் கார்டு புதுப்பிக்க என்ன ஆவணங்கள் தேவை?" style={{ minHeight: 0, fontFamily: 'var(--sans)' }} />
                </div>
                <div className="field">
                  <label className="field__label" htmlFor="scope">Search in</label>
                  <select id="scope" className="field__select" value={scope} onChange={(e) => setScope(e.target.value)}>
                    <option value="">All documents</option>
                    {docs.filter((d) => d.status === 'ready').map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
                  </select>
                </div>
                <div className="form-actions">
                  <button type="submit" className="btn btn--primary" disabled={!ready || searching || !question.trim()}>{searching ? 'Searching…' : 'Find relevant chunks'}</button>
                  {result && <span className="card__note">{result.translate_ms > 0 ? `translate ${result.translate_ms} ms · ` : ''}embed {result.embed_ms} ms · search {result.search_ms} ms</span>}
                </div>
              </form>
            </div>
            {result?.translated_query && (
              <div className="card__body" style={{ paddingTop: 0 }}>
                <span className="card__note">Searched as: <strong>{result.translated_query}</strong> <span className="muted">(translated with Sarvam, the same way the agent does on calls)</span></span>
              </div>
            )}
            {result && (
              <div className="list">
                {result.hits.length === 0 && <div className="card__body"><p className="card__note">No chunks found.</p></div>}
                {result.hits.map((h, i) => (
                  <div className="hit" key={`${h.document_id}-${h.chunk_index}`}>
                    <div className="hit__head">
                      <span className="hit__rank">#{i + 1}</span>
                      <span className="hit__source">{h.document_name} <span className="muted">· chunk {h.chunk_index}</span></span>
                      <span className="topbar__spacer" />
                      <span className={`pill ${h.score >= 0.5 ? 'pill--up' : h.score >= 0.35 ? 'pill--warn' : 'pill--idle'}`}>{(h.score * 100).toFixed(0)}% match</span>
                    </div>
                    <div className="bar__track"><div className="bar__fill" style={{ width: `${Math.max(2, Math.min(100, h.score * 100))}%` }} /></div>
                    <p className="hit__text">{h.text}</p>
                  </div>
                ))}
              </div>
            )}
          </section>
        </div>
      </div>
    </>
  );
}
