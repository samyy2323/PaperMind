// In-memory (localStorage-backed) implementation of the PaperMind API.
// Every function mirrors an endpoint in ./index.js and returns the same JSON shape.
import { MOCK_LATENCY } from './config';
import { buildSeed, QUESTION_BANK, DEMO_USERS, synthesizeAnswers, mulberry32 } from './mockData';
import { summarizeSubmission, needsReview, answerFlags, effectiveScore } from '../utils/grading';

const STORE_KEY = 'papermind.mockdb.v2';
const STAGE_MS = 1500;
export const PIPELINE_STAGES = ['ocr', 'correction', 'grading', 'done'];

let db = load();

function load() {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    if (raw) return JSON.parse(raw);
  } catch {
    /* fall through to seed */
  }
  return buildSeed();
}

function save() {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(db));
  } catch {
    /* storage unavailable */
  }
}

function classIdFor(subDomain) {
  return `class-${({ 'Object-Oriented Programming': 'oop', 'Software Engineering': 'se', 'AI & ML': 'ml' })[subDomain] || String(subDomain || 'x').toLowerCase().replace(/\W+/g, '-')}`;
}

const clone = (v) => JSON.parse(JSON.stringify(v));
const delay = (ms = MOCK_LATENCY) => new Promise((r) => setTimeout(r, ms));
const uid = (p) => `${p}-${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;

function fail(message, status = 400) {
  const err = new Error(message);
  err.status = status;
  throw err;
}

function findExam(id) {
  return db.exams.find((e) => e.id === id) || fail('Exam not found', 404);
}
function findSubmission(id) {
  const sub = db.submissions.find((s) => String(s.id) === String(id)) || fail('Submission not found', 404);
  sub.answers.forEach((a) => {
    if (!a.id) a.id = `${sub.id}:${a.questionId}`;
  });
  return sub;
}
function findAnswer(answerId) {
  for (const sub of db.submissions) {
    const ans = findSubmission(sub.id).answers.find((a) => a.id === answerId);
    if (ans) return { sub, ans };
  }
  return fail('Answer not found', 404);
}

// Simulated backend pipeline: a submission advances one stage every STAGE_MS.
function advance(sub) {
  if (sub.status !== 'processing' || sub.stalled || !sub.pipelineStartedAt) return sub;
  const elapsed = Date.now() - sub.pipelineStartedAt;
  const stageIndex = Math.min(PIPELINE_STAGES.length - 1, Math.floor(elapsed / STAGE_MS));
  if (PIPELINE_STAGES[stageIndex] === 'done') {
    const exam = findExam(sub.examId);
    sub.answers = synthesizeAnswers(exam, mulberry32(sub.pipelineStartedAt % 100000), 0.72);
    sub.status = 'graded';
    save();
  }
  return sub;
}
const advanceAll = () => db.submissions.forEach(advance);

function submissionRow(sub, { forStudent = false } = {}) {
  const exam = findExam(sub.examId);
  const s = summarizeSubmission(sub, exam);
  // Students only see marks for answers that are final (not awaiting teacher review).
  if (forStudent && s.total != null) {
    s.total -= sub.answers.reduce((t, a) => {
      const q = exam.questions.find((qq) => qq.id === a.questionId);
      return q && needsReview(a, q) ? t + effectiveScore(a) : t;
    }, 0);
  }
  return {
    id: sub.id,
    examId: sub.examId,
    examTitle: exam.title,
    examSubject: exam.subject,
    studentId: sub.studentId,
    studentName: sub.studentName,
    usn: sub.usn,
    submittedAt: sub.submittedAt,
    pages: sub.pages,
    status: s.status,
    total: s.total,
    max: s.max,
    pending: s.pending,
    mismatch: s.mismatch,
    lowConfidence: s.lowConf,
    hasReferencePdf: true,
    error: null,
  };
}

// ---------------------------------------------------------------------------
export const mock = {
  async login(role) {
    await delay();
    return clone(DEMO_USERS[role]);
  },

  // ---- Exams --------------------------------------------------------------
  async getExams() {
    await delay();
    advanceAll();
    return db.exams.map((e) => ({
      ...clone(e),
      submissionCount: db.submissions.filter((s) => s.examId === e.id).length,
    }));
  },

  async getExam(examId) {
    await delay();
    return clone(findExam(examId));
  },

  async createExam(payload) {
    await delay();
    const exam = {
      id: uid('exam'),
      status: 'draft',
      classId: classIdFor(payload.subDomain),
      publishedAt: null,
      createdAt: new Date().toISOString(),
      questions: [],
      answerKeyFile: null,
      ...payload,
    };
    db.exams.unshift(exam);
    save();
    return clone(exam);
  },

  async updateExam(examId, patch) {
    await delay();
    const exam = findExam(examId);
    if (exam.status === 'published') fail('Published exams are locked.', 409);
    Object.assign(exam, patch);
    save();
    return clone(exam);
  },

  async publishExam(examId) {
    await delay(700);
    const exam = findExam(examId);
    if (!exam.questions.length) fail('Add at least one question before publishing.');
    Object.assign(exam, { status: 'published', publishedAt: new Date().toISOString() });
    save();
    return clone(exam);
  },

  // Simulates PDF text extraction of the answer key (~2.5s).
  async extractAnswerKey(examId, file, { subDomain, numQuestions, totalMarks }, onProgress) {
    const steps = 25;
    for (let i = 1; i <= steps; i += 1) {
      await delay(100);
      onProgress?.(Math.round((i / steps) * 100));
    }
    const bank = QUESTION_BANK[subDomain] || [];
    const n = Math.max(1, Number(numQuestions) || bank.length);
    const base = Math.floor(Number(totalMarks) / n) || 1;
    let remainder = Number(totalMarks) - base * n;
    return Array.from({ length: n }, (_, i) => {
      const src = bank[i];
      const marks = base + (remainder-- > 0 ? 1 : 0);
      return src
        ? { id: uid('q'), number: i + 1, ...src, marks, extractionConfidence: 0.9 + Math.random() * 0.09 }
        : {
            id: uid('q'),
            number: i + 1,
            type: 'subjective',
            marks,
            extractionConfidence: 0.41,
            text: '',
            referenceAnswer: '',
          };
    });
  },

  // ---- Knowledge base -----------------------------------------------------
  async getKnowledgeDocs(subDomain) {
    await delay();
    return clone(db.knowledgeDocs.filter((d) => !subDomain || d.subDomain === subDomain));
  },

  async uploadKnowledgeDoc(file, subDomain, onProgress) {
    for (let i = 1; i <= 10; i += 1) {
      await delay(120);
      onProgress?.(i * 10);
    }
    const sizeKb = Math.max(1, Math.round(file.size / 1024));
    const doc = {
      id: uid('kb'),
      fileName: file.name,
      subDomain,
      sizeKb,
      chunksIndexed: Math.max(3, Math.round(sizeKb / 9)),
      uploadedAt: new Date().toISOString(),
      status: 'indexed',
    };
    db.knowledgeDocs.unshift(doc);
    save();
    return clone(doc);
  },

  async deleteKnowledgeDoc(docId) {
    await delay();
    db.knowledgeDocs = db.knowledgeDocs.filter((d) => d.id !== docId);
    save();
    return { ok: true };
  },

  // ---- Roster ---------------------------------------------------------------
  async getRoster(classId) {
    await delay();
    db.roster ||= {};
    return clone(db.roster[classId] || []);
  },

  async addStudent(classId, { name, rollNumber }) {
    await delay();
    db.roster ||= {};
    const list = (db.roster[classId] ||= []);
    if (!name?.trim() || !rollNumber?.trim()) fail('Name and roll number are required.');
    if (list.some((s) => s.rollNumber.toLowerCase() === rollNumber.trim().toLowerCase())) fail('That roll number is already on the roster.', 409);
    const student = { id: uid('stu'), name: name.trim(), rollNumber: rollNumber.trim() };
    list.push(student);
    list.sort((a, b) => a.rollNumber.localeCompare(b.rollNumber));
    save();
    return clone(student);
  },

  async updateStudent(classId, studentId, patch) {
    await delay();
    const s = (db.roster[classId] || []).find((x) => x.id === studentId) || fail('Student not found', 404);
    Object.assign(s, { name: patch.name?.trim() || s.name, rollNumber: patch.rollNumber?.trim() || s.rollNumber });
    save();
    return clone(s);
  },

  async removeStudent(classId, studentId) {
    await delay();
    db.roster[classId] = (db.roster[classId] || []).filter((x) => x.id !== studentId);
    save();
    return { ok: true };
  },

  // ---- Submissions (the teacher uploads photos on a student's behalf) -----------
  async uploadAnswerSheet({ studentId, examId, files }, onProgress) {
    for (let i = 1; i <= 5; i += 1) {
      await delay(120);
      onProgress?.(i * 20);
    }
    const exam = findExam(examId);
    if (exam.status !== 'published') fail('Publish the exam before uploading answer sheets.', 409);
    const student = (db.roster?.[exam.classId] || []).find((s) => s.id === studentId) || fail('Student is not on this exam’s roster.', 404);
    const sub = {
      id: uid('sub'),
      examId: exam.id,
      studentId: student.id,
      studentName: student.name,
      usn: student.rollNumber,
      submittedAt: new Date().toISOString(),
      pages: files.length,
      status: 'processing',
      pipelineStartedAt: Date.now(),
      answers: [],
    };
    db.submissions.push(sub);
    save();
    return { submissionId: sub.id, examId: exam.id, status: 'processing', stage: PIPELINE_STAGES[0] };
  },

  async getSubmissionStatus(submissionId) {
    await delay(150);
    const sub = advance(findSubmission(submissionId));
    if (sub.status !== 'processing') return { status: sub.status, stage: 'done', error: null };
    if (sub.stalled) return { status: 'processing', stage: 'grading', error: null };
    const idx = Math.floor((Date.now() - sub.pipelineStartedAt) / STAGE_MS);
    return { status: 'processing', stage: PIPELINE_STAGES[Math.min(idx, PIPELINE_STAGES.length - 2)], error: null };
  },

  async openReferencePdf() {
    await delay(150);
    throw new Error('Demo mode: reference PDFs are assembled by the real backend from the uploaded photos.');
  },

  async getStudentSubmissions(studentId) {
    await delay();
    advanceAll();
    return db.submissions
      .filter((s) => s.studentId === studentId)
      .sort((x, y) => y.submittedAt.localeCompare(x.submittedAt))
      .map((s) => submissionRow(s, { forStudent: true }));
  },

  async getSubmission(submissionId) {
    await delay();
    const sub = advance(findSubmission(submissionId));
    return { exam: clone(findExam(sub.examId)), submission: clone(sub) };
  },

  async getSubmissions(examId) {
    await delay();
    advanceAll();
    return db.submissions.filter((s) => s.examId === examId).map(submissionRow);
  },

  async getReviewQueue(examId) {
    const rows = await mock.getSubmissions(examId);
    return rows.filter((r) => r.status === 'needs_review');
  },

  async overrideScore(answerId, score, reason, teacher) {
    await delay();
    const { sub, ans } = findAnswer(answerId);
    const q = findExam(sub.examId).questions.find((x) => x.id === ans.questionId);
    const n = Number(score);
    if (score === '' || Number.isNaN(n) || n < 0 || n > q.marks) fail(`Score must be between 0 and ${q.marks}.`);
    if (!reason || !reason.trim()) fail('A reason is required when overriding a score.');
    ans.override = { score: n, reason: reason.trim(), by: teacher?.name || 'Teacher', at: new Date().toISOString() };
    save();
    return clone(ans);
  },

  async markReviewed(answerId) {
    await delay();
    const { ans } = findAnswer(answerId);
    ans.reviewed = true;
    save();
    return clone(ans);
  },

  // ---- Analytics ----------------------------------------------------------
  async getAnalytics(examId) {
    await delay();
    advanceAll();
    const exam = findExam(examId);
    const subs = db.submissions.filter((s) => s.examId === examId);
    const graded = subs.filter((s) => s.status !== 'processing');
    const rows = graded.map((s) => ({ sub: s, sum: summarizeSubmission(s, exam) }));
    const pcts = rows.map((r) => (r.sum.total / exam.totalMarks) * 100);

    const bins = Array.from({ length: 10 }, (_, i) => ({
      range: `${i * 10}â€“${i * 10 + 10}%`,
      lo: i * 10,
      count: 0,
    }));
    pcts.forEach((p) => {
      bins[Math.min(9, Math.floor(p / 10))].count += 1;
    });

    const perQuestion = exam.questions.map((q) => {
      const scores = graded.map((s) => s.answers.find((x) => x.questionId === q.id)).filter(Boolean);
      const avg = scores.reduce((t, x) => t + effectiveScore(x), 0) / (scores.length || 1);
      const flagged = scores.filter((x) => needsReview(x, q)).length;
      return {
        question: `Q${q.number}`,
        type: q.type,
        avgPct: Math.round((avg / q.marks) * 100),
        avg: +avg.toFixed(1),
        marks: q.marks,
        flagged,
      };
    });

    let lowConfidence = 0;
    let mismatch = 0;
    graded.forEach((s) =>
      s.answers.forEach((x) => {
        const q = exam.questions.find((qq) => qq.id === x.questionId);
        if (!q || !needsReview(x, q)) return;
        const f = answerFlags(x, q);
        if (f.relevanceMismatch) mismatch += 1;
        else if (f.lowConfidence) lowConfidence += 1;
      }),
    );

    const sorted = [...pcts].sort((x, y) => x - y);
    const median = sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0;
    return {
      exam: { id: exam.id, title: exam.title, subject: exam.subject, totalMarks: exam.totalMarks },
      totalSubmissions: subs.length,
      processing: subs.length - graded.length,
      averageScore: rows.length ? rows.reduce((t, r) => t + r.sum.total, 0) / rows.length : 0,
      averagePct: pcts.length ? pcts.reduce((t, p) => t + p, 0) / pcts.length : 0,
      medianPct: median,
      highestPct: sorted.length ? sorted[sorted.length - 1] : 0,
      flaggedSubmissions: rows.filter((r) => r.sum.status === 'needs_review').length,
      cleanSubmissions: rows.filter((r) => r.sum.status === 'graded').length,
      flaggedAnswers: { lowConfidence, mismatch },
      distribution: bins,
      perQuestion,
    };
  },

  async resetDemoData() {
    db = buildSeed();
    save();
    return { ok: true };
  },
};
