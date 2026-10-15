// PaperMind API service layer.
// Each function picks the mock implementation or the real HTTP call — swapping one endpoint
// to the real backend is a one-line change, or set VITE_USE_MOCK=false to switch them all.
import { USE_MOCK } from './config';
import { http, data } from './client';
import { mock } from './mockApi';

export { PIPELINE_STAGES } from './mockApi';

const upload = (url, form, onProgress) =>
  data(
    http.post(url, form, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e) => e.total && onProgress?.(Math.round((e.loaded / e.total) * 100)),
    }),
  );

const toForm = (fields) => {
  const f = new FormData();
  Object.entries(fields).forEach(([k, v]) => {
    if (Array.isArray(v)) v.forEach((item) => f.append(k, item));
    else if (v !== undefined && v !== null) f.append(k, v);
  });
  return f;
};

// Real-backend demo accounts (created by `python backend/scripts/seed_demo.py`).
const DEMO_CREDENTIALS = {
  teacher: { email: 'kulkarni@college.edu', password: 'demo1234' },
  student: { email: 'rohan@students.papermind.local', password: 'demo1234' },
};

// Knowledge-base documents are stored per class; the UI works in sub-domains, so resolve the class first.
const ensureClass = (subDomain, subject = subDomain) =>
  data(http.post('/classes/ensure', { subject, subDomain }));

// ---- Auth -----------------------------------------------------------------
export const login = (role) =>
  USE_MOCK ? mock.login(role) : data(http.post('/auth/login', DEMO_CREDENTIALS[role]));

// ---- Exams ----------------------------------------------------------------
export const getExams = () =>
  USE_MOCK ? mock.getExams() : data(http.get('/exams'));

export const getExam = (examId) =>
  USE_MOCK ? mock.getExam(examId) : data(http.get(`/exams/${examId}`));

export const createExam = (payload) =>
  USE_MOCK
    ? mock.createExam(payload)
    : data(http.post('/exams', { title: payload.title, totalMarks: payload.totalMarks, subject: payload.subject, subDomain: payload.subDomain }));

// Saves the exam fields and the whole (draft) question list.
export const updateExam = async (examId, patch) => {
  if (USE_MOCK) return mock.updateExam(examId, patch);
  await http.patch(`/exams/${examId}`, { title: patch.title, totalMarks: patch.totalMarks });
  if (patch.questions) await http.put(`/exams/${examId}/questions`, patch.questions);
  return data(http.get(`/exams/${examId}`));
};

export const publishExam = (examId) =>
  USE_MOCK ? mock.publishExam(examId) : data(http.post(`/exams/${examId}/publish`));

// Uploads the answer-key PDF; the backend extracts and stores the questions and returns them.
// meta = { subDomain, numQuestions, totalMarks } — sent as form fields for forward-compatibility.
export const extractAnswerKey = (examId, file, meta, onProgress) =>
  USE_MOCK
    ? mock.extractAnswerKey(examId, file, meta, onProgress)
    : upload(
        `/exams/${examId}/answer-key`,
        toForm({ file, subDomain: meta?.subDomain, numQuestions: meta?.numQuestions, totalMarks: meta?.totalMarks }),
        onProgress,
      );

// ---- Roster ---------------------------------------------------------------
export const getRoster = (classId) =>
  USE_MOCK ? mock.getRoster(classId) : data(http.get(`/classes/${classId}/roster`));

export const addStudent = (classId, student) =>
  USE_MOCK ? mock.addStudent(classId, student) : data(http.post(`/classes/${classId}/roster`, student));

export const updateStudent = (classId, studentId, patch) =>
  USE_MOCK ? mock.updateStudent(classId, studentId, patch) : data(http.patch(`/classes/${classId}/roster/${studentId}`, patch));

export const removeStudent = (classId, studentId) =>
  USE_MOCK ? mock.removeStudent(classId, studentId) : data(http.delete(`/classes/${classId}/roster/${studentId}`));

// ---- Knowledge base (RAG source material) ---------------------------------
export const getKnowledgeDocs = (subDomain) =>
  USE_MOCK ? mock.getKnowledgeDocs(subDomain) : data(http.get('/knowledge-base', { params: { subDomain } }));

export const uploadKnowledgeDoc = async (file, subDomain, onProgress) => {
  if (USE_MOCK) return mock.uploadKnowledgeDoc(file, subDomain, onProgress);
  const klass = await ensureClass(subDomain);
  return upload(`/classes/${klass.id}/knowledge-base`, toForm({ file }), onProgress);
};

export const deleteKnowledgeDoc = (docId) =>
  USE_MOCK ? mock.deleteKnowledgeDoc(docId) : data(http.delete(`/knowledge-base/${docId}`));

// ---- Answer sheets (teacher uploads photos for a student) ------------------
export const uploadAnswerSheet = ({ studentId, examId, files }, onProgress) =>
  USE_MOCK
    ? mock.uploadAnswerSheet({ studentId, examId, files }, onProgress)
    : upload(`/students/${studentId}/answer-sheet`, toForm({ examId, files }), onProgress);

export const getSubmissionStatus = (submissionId) =>
  USE_MOCK ? mock.getSubmissionStatus(submissionId) : data(http.get(`/answer-sheets/${submissionId}/status`));

export const getStudentSubmissions = (studentId) =>
  USE_MOCK ? mock.getStudentSubmissions(studentId) : data(http.get(`/students/${studentId}/answer-sheets`));

export const getSubmission = (submissionId) =>
  USE_MOCK ? mock.getSubmission(submissionId) : data(http.get(`/answer-sheets/${submissionId}/results`));

export const getSubmissions = (examId) =>
  USE_MOCK ? mock.getSubmissions(examId) : data(http.get(`/exams/${examId}/submissions`));

export const getReviewQueue = (examId) =>
  USE_MOCK ? mock.getReviewQueue(examId) : data(http.get(`/exams/${examId}/review-queue`));

export const reprocessSubmission = (submissionId) =>
  USE_MOCK ? Promise.reject(new Error('Not available in demo mode.')) : data(http.post(`/answer-sheets/${submissionId}/reprocess`));

// Opens the reference PDF (the uploaded photos combined into one file) in a new tab.
// Fetched with the auth header, so it can't be a plain <a href>.
export const openReferencePdf = async (submissionId) => {
  if (USE_MOCK) return mock.openReferencePdf(submissionId);
  const res = await http.get(`/answer-sheets/${submissionId}/reference-pdf`, { responseType: 'blob' });
  const url = URL.createObjectURL(res.data);
  window.open(url, '_blank', 'noopener');
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
};

// ---- Teacher review ---------------------------------------------------------
// Answers are addressed by their own id (`answer.id`).
export const overrideScore = (answerId, score, reason, teacher) =>
  USE_MOCK ? mock.overrideScore(answerId, score, reason, teacher) : data(http.patch(`/student-answers/${answerId}/override`, { score, reason }));

export const markReviewed = (answerId) =>
  USE_MOCK ? mock.markReviewed(answerId) : data(http.post(`/student-answers/${answerId}/review`));

// ---- Analytics ------------------------------------------------------------
export const getAnalytics = (examId) =>
  USE_MOCK ? mock.getAnalytics(examId) : data(http.get(`/exams/${examId}/analytics`));

// ---- Demo only ------------------------------------------------------------
export const resetDemoData = () => (USE_MOCK ? mock.resetDemoData() : Promise.resolve());

export function errorMessage(err) {
  const detail = err?.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map((d) => d.msg).join('; ');
  return err?.response?.data?.message || err?.message || 'Something went wrong.';
}
