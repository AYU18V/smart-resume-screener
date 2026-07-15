import { useMemo, useState } from "react"
import {
  BarChart3,
  CheckCircle2,
  ClipboardList,
  Download,
  Eye,
  FileSearch,
  FileText,
  FileUp,
  Search,
  Trash2,
  Users
} from "lucide-react"
import Sidebar from "../components/Sidebar"
import Navbar from "../components/Navbar"
import StatCard from "../components/StatCard"
import ResumeAnalyzer from "../components/ResumeAnalyzer"

// Removed: Unrelated analytics dashboard
// import SkillChart from "../components/SkillChart"
// Removed: Forecasting module
// import SkillTrendChart from "../components/SkillTrendChart"
// Removed: Forecasting module
// import ForecastPanel from "../components/ForecastPanel"
// Removed: Geographic workforce analytics
// import GeographicWorkforcePanel from "../components/GeographicWorkforcePanel"
// Removed: Unrelated job-market feed
// import JobTable from "../components/JobTable"

const average = (items) => {
  if (!items.length) return 0
  return Math.round(items.reduce((total, item) => total + item.matchScore, 0) / items.length)
}

function Dashboard() {
  const [analyses, setAnalyses] = useState([])
  const [search, setSearch] = useState("")
  const [statusFilter, setStatusFilter] = useState("all")
  const [activeReportId, setActiveReportId] = useState(null)

  const activeReport = analyses.find((item) => item.id === activeReportId) || analyses[0]
  const shortlistedCount = analyses.filter((item) => item.status === "Shortlisted").length
  const averageScore = average(analyses)

  const filteredAnalyses = useMemo(() => {
    return analyses.filter((item) => {
      const haystack = `${item.candidateName} ${item.fileName} ${item.topSkills.join(" ")}`.toLowerCase()
      const matchesSearch = haystack.includes(search.toLowerCase())
      const matchesStatus = statusFilter === "all" || item.status === statusFilter
      return matchesSearch && matchesStatus
    })
  }, [analyses, search, statusFilter])

  const handleAnalysisComplete = (analysis) => {
    setAnalyses((current) => [analysis, ...current.filter((item) => item.id !== analysis.id)])
    setActiveReportId(analysis.id)
  }

  const deleteAnalysis = (id) => {
    setAnalyses((current) => current.filter((item) => item.id !== id))
    if (activeReportId === id) {
      setActiveReportId(null)
    }
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-950">
      <Sidebar />

      <div className="lg:pl-72">
        <Navbar />

        <main className="mx-auto flex w-full max-w-7xl flex-col gap-8 px-4 py-6 sm:px-6 lg:px-8">
          <section className="panel-grid overflow-hidden rounded-lg border border-slate-200 bg-white">
            <div className="grid gap-8 p-6 lg:grid-cols-[1.1fr_0.9fr] lg:p-8">
              <div className="flex flex-col justify-center">
                <p className="section-kicker">Smart Resume Screener</p>
                <h1 className="mt-3 max-w-3xl text-4xl font-semibold tracking-normal text-slate-950 sm:text-5xl">
                  Screen resumes against job descriptions with focused LLM matching.
                </h1>
                <p className="mt-4 max-w-2xl text-base leading-7 text-slate-600">
                  Upload a resume, add a job description, extract candidate skills, experience, and education, then review the match score, reasoning, and shortlist status in one recruitment dashboard.
                </p>
                <div className="mt-6 flex flex-col gap-3 sm:flex-row">
                  <a href="#upload-resume" className="btn-primary">
                    <FileUp size={18} />
                    Upload Resume
                  </a>
                  <a href="#job-description" className="btn-secondary">
                    <ClipboardList size={18} />
                    Upload Job Description
                  </a>
                </div>
              </div>

              <div className="rounded-lg border border-slate-200 bg-slate-950 p-5 text-white shadow-sm">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium text-slate-300">Current Candidate</p>
                  <span className="rounded-full bg-emerald-400/15 px-3 py-1 text-xs font-semibold text-emerald-200">
                    {activeReport?.status || "Ready"}
                  </span>
                </div>
                <h2 className="mt-8 text-3xl font-semibold">
                  {activeReport?.candidateName || "No resume analyzed yet"}
                </h2>
                <div className="mt-5 grid grid-cols-2 gap-3">
                  <div className="rounded-lg bg-white/10 p-4">
                    <p className="text-xs text-slate-300">Match Score</p>
                    <p className="mt-2 text-3xl font-semibold">{activeReport ? `${activeReport.matchScore}%` : "0%"}</p>
                  </div>
                  <div className="rounded-lg bg-white/10 p-4">
                    <p className="text-xs text-slate-300">Top Skills</p>
                    <p className="mt-2 text-sm font-medium">
                      {activeReport?.topSkills.slice(0, 3).join(", ") || "Awaiting resume"}
                    </p>
                  </div>
                </div>
                <p className="mt-5 text-sm leading-6 text-slate-300">
                  {activeReport?.justification || "Upload a resume and job description to generate the first candidate report."}
                </p>
              </div>
            </div>
          </section>

          <section id="dashboard" className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <StatCard
              title="Total Uploaded Resumes"
              value={analyses.length}
              caption="Files reviewed in this session"
              icon={FileUp}
              tone="blue"
            />
            <StatCard
              title="Processed Resumes"
              value={analyses.length}
              caption="Completed resume analyses"
              icon={CheckCircle2}
              tone="emerald"
            />
            <StatCard
              title="Shortlisted Candidates"
              value={shortlistedCount}
              caption="Candidates above shortlist threshold"
              icon={Users}
              tone="amber"
            />
            <StatCard
              title="Average Match Score"
              value={`${averageScore}%`}
              caption="Mean resume-to-JD score"
              icon={BarChart3}
              tone="slate"
            />
          </section>

          <section className="grid gap-4 lg:grid-cols-2">
            <div className="surface-card p-5">
              <div className="mb-4 flex items-center justify-between gap-3">
                <div>
                  <p className="section-kicker">Recent Uploads</p>
                  <h2 className="section-title">Uploaded Resumes</h2>
                </div>
                <FileText className="text-slate-400" size={20} />
              </div>
              <div className="space-y-3">
                {analyses.slice(0, 4).map((item) => (
                  <div key={item.id} className="flex items-center justify-between gap-4 rounded-lg border border-slate-200 bg-white p-3">
                    <div>
                      <p className="font-medium text-slate-900">{item.fileName}</p>
                      <p className="text-sm text-slate-500">{item.createdAt}</p>
                    </div>
                    <span className="rounded-full bg-slate-100 px-3 py-1 text-sm font-semibold text-slate-700">
                      {item.matchScore}%
                    </span>
                  </div>
                ))}
                {!analyses.length && <p className="empty-copy">Resume uploads will appear here after analysis.</p>}
              </div>
            </div>

            <div className="surface-card p-5">
              <div className="mb-4 flex items-center justify-between gap-3">
                <div>
                  <p className="section-kicker">Recent Analysis</p>
                  <h2 className="section-title">Candidate Decisions</h2>
                </div>
                <FileSearch className="text-slate-400" size={20} />
              </div>
              <div className="space-y-3">
                {analyses.slice(0, 4).map((item) => (
                  <div key={item.id} className="rounded-lg border border-slate-200 bg-white p-3">
                    <div className="flex items-center justify-between gap-3">
                      <p className="font-medium text-slate-900">{item.candidateName}</p>
                      <span className={item.status === "Shortlisted" ? "badge-success" : "badge-muted"}>
                        {item.status}
                      </span>
                    </div>
                    <p className="mt-2 text-sm text-slate-500">{item.recommendation}</p>
                  </div>
                ))}
                {!analyses.length && <p className="empty-copy">LLM matching results will appear here after screening.</p>}
              </div>
            </div>
          </section>

          <ResumeAnalyzer onAnalysisComplete={handleAnalysisComplete} />

          <section id="history" className="surface-card p-5">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
              <div>
                <p className="section-kicker">History</p>
                <h2 className="section-title">Resume History</h2>
                <p className="mt-2 text-sm text-slate-600">
                  Search uploaded resumes, filter by shortlist status, delete session entries, or view a candidate report.
                </p>
              </div>
              <div className="grid gap-3 sm:grid-cols-[1fr_auto]">
                <label className="relative">
                  <Search className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={16} />
                  <input
                    className="input-field pl-9"
                    value={search}
                    onChange={(event) => setSearch(event.target.value)}
                    placeholder="Search resume, candidate, or skill"
                  />
                </label>
                <select className="input-field" value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)}>
                  <option value="all">All Status</option>
                  <option value="Shortlisted">Shortlisted</option>
                  <option value="Review">Review</option>
                </select>
              </div>
            </div>

            <div className="mt-5 overflow-hidden rounded-lg border border-slate-200">
              <div className="hidden grid-cols-[1.1fr_0.7fr_0.8fr_0.6fr_0.8fr] gap-4 bg-slate-100 px-4 py-3 text-xs font-semibold uppercase text-slate-500 md:grid">
                <span>Candidate</span>
                <span>Score</span>
                <span>Education</span>
                <span>Status</span>
                <span>Actions</span>
              </div>
              {filteredAnalyses.map((item) => (
                <div key={item.id} className="grid gap-3 border-t border-slate-200 bg-white px-4 py-4 md:grid-cols-[1.1fr_0.7fr_0.8fr_0.6fr_0.8fr] md:items-center">
                  <div>
                    <p className="font-medium text-slate-900">{item.candidateName}</p>
                    <p className="text-sm text-slate-500">{item.fileName}</p>
                  </div>
                  <p className="font-semibold text-slate-900">{item.matchScore}%</p>
                  <p className="text-sm text-slate-600">{item.education}</p>
                  <span className={item.status === "Shortlisted" ? "badge-success w-fit" : "badge-muted w-fit"}>
                    {item.status}
                  </span>
                  <div className="flex gap-2">
                    <button className="icon-button" onClick={() => setActiveReportId(item.id)} aria-label="View report">
                      <Eye size={16} />
                    </button>
                    <button className="icon-button danger" onClick={() => deleteAnalysis(item.id)} aria-label="Delete resume">
                      <Trash2 size={16} />
                    </button>
                  </div>
                </div>
              ))}
              {!filteredAnalyses.length && (
                <div className="border-t border-slate-200 bg-white p-6 text-center text-sm text-slate-500">
                  No resume history matches the current filters.
                </div>
              )}
            </div>
          </section>

          <section id="reports" className="surface-card p-5">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
              <div>
                <p className="section-kicker">Reports</p>
                <h2 className="section-title">Candidate Match Report</h2>
                <p className="mt-2 text-sm text-slate-600">
                  Review resume summary, JD summary, comparison score, reasoning, and recommendation.
                </p>
              </div>
              <button className="btn-secondary" onClick={() => window.print()} disabled={!activeReport}>
                <Download size={18} />
                Download PDF
              </button>
            </div>

            {activeReport ? (
              <div className="mt-5 grid gap-4 lg:grid-cols-2">
                <div className="rounded-lg border border-slate-200 bg-white p-4">
                  <h3 className="font-semibold text-slate-900">Resume Summary</h3>
                  <p className="mt-2 text-sm leading-6 text-slate-600">{activeReport.resumeSummary}</p>
                </div>
                <div className="rounded-lg border border-slate-200 bg-white p-4">
                  <h3 className="font-semibold text-slate-900">JD Summary</h3>
                  <p className="mt-2 text-sm leading-6 text-slate-600">{activeReport.jdSummary}</p>
                </div>
                <div className="rounded-lg border border-slate-200 bg-white p-4">
                  <h3 className="font-semibold text-slate-900">Comparison</h3>
                  <p className="mt-2 text-sm leading-6 text-slate-600">{activeReport.comparison}</p>
                </div>
                <div className="rounded-lg border border-slate-200 bg-white p-4">
                  <h3 className="font-semibold text-slate-900">Reasoning</h3>
                  <p className="mt-2 text-sm leading-6 text-slate-600">{activeReport.justification}</p>
                </div>
                <div className="rounded-lg border border-slate-200 bg-white p-4 lg:col-span-2">
                  <h3 className="font-semibold text-slate-900">Recommendation</h3>
                  <p className="mt-2 text-sm leading-6 text-slate-600">{activeReport.recommendation}</p>
                </div>
              </div>
            ) : (
              <p className="mt-5 rounded-lg border border-slate-200 bg-white p-6 text-center text-sm text-slate-500">
                Analyze a resume to generate a report.
              </p>
            )}
          </section>
        </main>
      </div>
    </div>
  )
}

export default Dashboard
