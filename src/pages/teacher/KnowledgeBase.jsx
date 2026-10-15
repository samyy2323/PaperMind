import { useState } from 'react';
import { deleteKnowledgeDoc, errorMessage, getKnowledgeDocs, uploadKnowledgeDoc } from '../../api';
import useAsync from '../../utils/useAsync';
import { SUB_DOMAINS, formatDate } from '../../utils/grading';
import FileUpload from '../../components/FileUpload';
import Icon from '../../components/Icon';
import { Alert, Button, Card, EmptyState, Loading, PageHeader, ProgressBar } from '../../components/ui';

export default function KnowledgeBase() {
  const [filter, setFilter] = useState('');
  const [uploadDomain, setUploadDomain] = useState(SUB_DOMAINS[0]);
  const [uploads, setUploads] = useState([]); // in-flight: { key, name, progress }
  const [error, setError] = useState('');
  const [confirming, setConfirming] = useState(null);
  const { data: docs, loading, error: loadError, setData } = useAsync(() => getKnowledgeDocs(filter || undefined), [filter]);

  const handleFiles = (files) => {
    setError('');
    files.forEach(async (file) => {
      const key = `${file.name}-${Date.now()}-${Math.random()}`;
      setUploads((u) => [...u, { key, name: file.name, progress: 0 }]);
      try {
        const doc = await uploadKnowledgeDoc(file, uploadDomain, (p) =>
          setUploads((u) => u.map((x) => (x.key === key ? { ...x, progress: p } : x))),
        );
        if (!filter || filter === doc.subDomain) setData((d) => [doc, ...(d || [])]);
      } catch (err) {
        setError(`${file.name}: ${errorMessage(err)}`);
      } finally {
        setUploads((u) => u.filter((x) => x.key !== key));
      }
    });
  };

  const remove = async (id) => {
    try {
      await deleteKnowledgeDoc(id);
      setData((d) => d.filter((x) => x.id !== id));
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setConfirming(null);
    }
  };

  const totalChunks = docs?.reduce((t, d) => t + d.chunksIndexed, 0) ?? 0;

  return (
    <>
      <PageHeader
        title="Knowledge base"
        subtitle="Syllabus and textbook material used to ground subjective grading (RAG). Answers are compared against these sources."
      />

      <Card className="mb-6 p-5 sm:p-6">
        <div className="grid gap-5 lg:grid-cols-[16rem_1fr]">
          <div>
            <h2 className="text-lg font-semibold">Add material</h2>
            <p className="mt-1 text-sm text-ink-soft">PDF or plain text. Documents are split into chunks and embedded for retrieval.</p>
            <label htmlFor="kb-domain" className="field-label mt-4">Sub-domain</label>
            <select id="kb-domain" className="field" value={uploadDomain} onChange={(e) => setUploadDomain(e.target.value)}>
              {SUB_DOMAINS.map((d) => (
                <option key={d}>{d}</option>
              ))}
            </select>
          </div>
          <div>
            <FileUpload
              multiple
              accept={{ 'application/pdf': ['.pdf'], 'text/plain': ['.txt', '.md'] }}
              onFiles={handleFiles}
              title="Drag & drop syllabus or textbook files"
              hint="PDF, TXT"
              icon="book"
            />
            {uploads.length > 0 && (
              <ul className="mt-4 space-y-3">
                {uploads.map((u) => (
                  <li key={u.key}>
                    <ProgressBar value={u.progress} label={`${u.progress < 60 ? 'Uploading' : 'Chunking & embedding'} · ${u.name}`} />
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </Card>

      {(error || loadError) && <Alert className="mb-4">{error || loadError}</Alert>}

      <div className="mb-3 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 className="text-lg font-semibold">Indexed documents</h2>
          {docs && (
            <p className="text-sm text-ink-soft">
              {docs.length} {docs.length === 1 ? 'document' : 'documents'} · <span className="tabular-nums">{totalChunks.toLocaleString()}</span> chunks
            </p>
          )}
        </div>
        <div className="w-full sm:w-64">
          <label htmlFor="kb-filter" className="sr-only">Filter by sub-domain</label>
          <select id="kb-filter" className="field" value={filter} onChange={(e) => setFilter(e.target.value)}>
            <option value="">All sub-domains</option>
            {SUB_DOMAINS.map((d) => (
              <option key={d}>{d}</option>
            ))}
          </select>
        </div>
      </div>

      {loading && !docs && <Loading />}
      {docs && docs.length === 0 && (
        <EmptyState icon="book" title="No documents in the knowledge base">
          {filter ? `Nothing uploaded for ${filter} yet. ` : ''}
          Upload the syllabus and key textbook chapters above so subjective answers can be checked against course material.
        </EmptyState>
      )}
      {docs && docs.length > 0 && (
        <Card className="overflow-x-auto">
          <table className="w-full min-w-[40rem] text-sm">
            <thead>
              <tr className="border-b border-line bg-paper-2/60 text-left text-xs font-semibold uppercase tracking-wide text-muted">
                <th className="px-4 py-2.5">Document</th>
                <th className="px-4 py-2.5">Sub-domain</th>
                <th className="px-4 py-2.5">Uploaded</th>
                <th className="px-4 py-2.5 text-right">Chunks indexed</th>
                <th className="w-32 px-4 py-2.5" />
              </tr>
            </thead>
            <tbody className="divide-y divide-line/70">
              {docs.map((d) => (
                <tr key={d.id} className="hover:bg-paper/60">
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2.5">
                      <Icon name="file" size={18} className="text-muted" />
                      <div className="min-w-0">
                        <div className="truncate font-medium text-ink">{d.fileName}</div>
                        <div className="text-xs text-muted">{d.sizeKb >= 1024 ? `${(d.sizeKb / 1024).toFixed(1)} MB` : `${d.sizeKb} KB`}</div>
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-ink-soft">{d.subDomain}</td>
                  <td className="px-4 py-3 text-ink-soft">{formatDate(d.uploadedAt)}</td>
                  <td className="px-4 py-3 text-right font-medium tabular-nums">{d.chunksIndexed.toLocaleString()}</td>
                  <td className="px-4 py-3 text-right">
                    {confirming === d.id ? (
                      <span className="inline-flex gap-1">
                        <Button size="sm" variant="danger" onClick={() => remove(d.id)}>Delete</Button>
                        <Button size="sm" variant="ghost" onClick={() => setConfirming(null)}>Cancel</Button>
                      </span>
                    ) : (
                      <button
                        type="button"
                        onClick={() => setConfirming(d.id)}
                        className="rounded p-1.5 text-muted hover:bg-bad-soft hover:text-bad"
                        aria-label={`Delete ${d.fileName}`}
                      >
                        <Icon name="trash" size={16} />
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </>
  );
}
