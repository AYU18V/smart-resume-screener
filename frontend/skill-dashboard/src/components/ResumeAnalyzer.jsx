import { createElement, useMemo, useRef, useState } from "react"
import {
  BadgeCheck,
  BookOpen,
  BriefcaseBusiness,
  CheckCircle2,
  ClipboardList,
  FileText,
  GraduationCap,
  Languages,
  LoaderCircle,
  RotateCcw,
  Sparkles,
  Target,
  UploadCloud,
  XCircle
} from "lucide-react"
import { analyzeResume, extractSkills, matchResume, matchSkills } from "../services/apiService"

const asArray = (value) => {
  if (Array.isArray(value)) return value.filter(Boolean)
  if (typeof value === "string" && value.trim()) return [value.trim()]
  return []
}

const skillName = (item) => (typeof item === "string" ? item : item?.skill || item?.name || "")
const percent = (value) => Math.max(0, Math.min(100, Math.round(Number(value || 0))))

const inferRating = (score) => {
  if (score >= 85) return "Excellent Match"
  if (score >= 75) return "Strong Match"
  if (score >= 60) return "Good Match"
  return "Needs Review"
}

const readTextFile = (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result || ""))
    reader.onerror = () => reject(new Error("Unable to read job description file"))
    reader.readAsText(file)
  })
}

function ResumeAnalyzer({ onAnalysisComplete }) {
  const [resumeFile, setResumeFile] = useState(null)
  const [jobDescription, setJobDescription] = useState("")
  const [jobFileName, setJobFileName] = useState("")
  const [result, setResult] = useState(null)
  const [matchResult, setMatchResult] = useState(null)
  const [jobSkills, setJobSkills] = useState([])
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState("")
  const [dragActive, setDragActive] = useState(false)
  const resumeInputRef = useRef(null)

  const detectedSkills = useMemo(() => asArray(result?.detected_skills).map(skillName).filter(Boolean), [result])
  const matchedSkills = useMemo(() => {
    const explicit = asArray(result?.matching_skills).map(skillName).filter(Boolean)
    if (explicit.length) return explicit
    return asArray(matchResult?.matches)
      .filter((item) => item?.recommended !== true)
      .map(skillName)
      .filter(Boolean)
  }, [matchResult, result])

  const missingSkills = useMemo(() => {
    const explicit = asArray(result?.missing_skills).map(skillName).filter(Boolean)
    if (explicit.length) return explicit
    return jobSkills.filter((skill) => !detectedSkills.some((candidateSkill) => candidateSkill.toLowerCase() === skill.toLowerCase()))
  }, [detectedSkills, jobSkills, result])

  const matchScore = useMemo(() => {
    if (!result) return 0
    if (result.match_score !== undefined) return percent(result.match_score)
    if (result.semantic_match_score !== undefined) return percent(result.semantic_match_score)
    if (result.ats_score !== undefined) return percent(result.ats_score)
    if (matchResult?.matches?.length) {
      const averageScore = matchResult.matches.reduce((total, item) => total + Number(item.score || 0), 0) / matchResult.matches.length
      return percent(averageScore * 100)
    }
    return 0
  }, [matchResult, result])

  const handleResumeFile = (file) => {
    if (!file) return
    const validTypes = ["application/pdf", "text/plain"]
    const validExtension = /\.(pdf|txt)$/i.test(file.name)
    if (!validTypes.includes(file.type) && !validExtension) {
      setError("Upload a PDF or TXT resume.")
      return
    }
    setError("")
    setResumeFile(file)
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setDragActive(false)
    handleResumeFile(event.dataTransfer.files?.[0])
  }

  const handleJobFile = async (file) => {
    if (!file) return
    if (!/\.txt$/i.test(file.name)) {
      setError("Upload a TXT job description or paste the job description text.")
      return
    }
    try {
      const text = await readTextFile(file)
      setJobDescription(text)
      setJobFileName(file.name)
      setError("")
    } catch (err) {
      setError(err.message)
    }
  }

  const clearJobDescription = () => {
    setJobDescription("")
    setJobFileName("")
    setJobSkills([])
    setMatchResult(null)
  }

  const analyzeCandidate = async () => {
    if (!resumeFile) {
      setError("Upload a resume before starting analysis.")
      return
    }
    if (!jobDescription.trim()) {
      setError("Paste or upload a job description before matching.")
      return
    }

    setLoading(true)
    setProgress(18)
    setError("")

    try {
      const resumeData = await analyzeResume(resumeFile)
      setProgress(55)

      const jdData = await extractSkills(jobDescription)
      const jdSkills = asArray(jdData.skills).map(skillName).filter(Boolean)
      setJobSkills(jdSkills)
      setProgress(75)

      const matches = await matchSkills(asArray(resumeData.detected_skills), jdSkills, 12)
      setMatchResult(matches)

      let llmResult = null
      let llmError = ""
      try {
        setProgress(88)
        llmResult = await matchResume(resumeData.resume_text || "", jobDescription)
      } catch (llmErr) {
        llmError = llmErr.message
      }

      const combinedResult = {
        ...resumeData,
        ...(llmResult || {}),
        llm_match: llmResult,
        llm_error: llmError
      }
      setResult(combinedResult)
      setProgress(100)

      const score = combinedResult.match_score !== undefined
        ? percent(combinedResult.match_score)
        : combinedResult.semantic_match_score !== undefined
          ? percent(combinedResult.semantic_match_score)
          : combinedResult.ats_score !== undefined
            ? percent(combinedResult.ats_score)
            : percent((matches.matches || []).reduce((total, item) => total + Number(item.score || 0), 0) / Math.max((matches.matches || []).length, 1) * 100)

      const topSkills = asArray(combinedResult.matching_skills?.length ? combinedResult.matching_skills : combinedResult.detected_skills).map(skillName).filter(Boolean).slice(0, 6)
      const status = score >= 75 ? "Shortlisted" : "Review"
      const candidateName = combinedResult.candidate_name || combinedResult.name || resumeFile.name.replace(/\.[^.]+$/, "")

      onAnalysisComplete?.({
        id: `${resumeFile.name}-${Date.now()}`,
        candidateName,
        fileName: resumeFile.name,
        matchScore: score,
        status,
        education: combinedResult.education_analysis || combinedResult.education || combinedResult.highest_education || "Not detected",
        experience: combinedResult.experience_analysis || combinedResult.experience || combinedResult.total_experience || "Not detected",
        topSkills,
        createdAt: new Date().toLocaleString(),
        resumeSummary: combinedResult.summary || `Parsed ${resumeFile.name} for skills, experience, education, projects, certifications, and languages.`,
        jdSummary: jobDescription.trim().slice(0, 220) || "No job description summary available.",
        comparison: `${topSkills.length} resume skills were compared against ${jdSkills.length} job description skills.`,
        justification: combinedResult.justification || combinedResult.ai_suggestion || `The candidate received a ${score}% match based on detected resume skills and job description requirements.`,
        recommendation: combinedResult.recommendation || (status === "Shortlisted"
          ? "Shortlist this candidate for recruiter review."
          : "Review missing skills before shortlisting this candidate.")
      })

      if (llmError) {
        setError(`Gemini matching unavailable: ${llmError}. Showing parser and semantic matching results.`)
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const strengths = asArray(result?.strengths || result?.resume_strengths)
  const weaknesses = asArray(result?.weaknesses || result?.resume_weaknesses)
  const recommendations = asArray(result?.career_recommendations || result?.recommendations)
  const candidateName = result?.candidate_name || result?.name || resumeFile?.name?.replace(/\.[^.]+$/, "") || "Candidate"
  const status = matchScore >= 75 ? "Shortlisted" : "Review"

  return (
    <section id="upload-resume" className="surface-card p-5">
      <div className="mb-5 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <p className="section-kicker">Upload Resume</p>
          <h2 className="section-title">Resume and Job Description Matching</h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
            Upload a PDF or TXT resume, paste or upload a job description, then run LLM matching for candidate score, justification, missing skills, and shortlist status.
          </p>
        </div>
        <button className="btn-primary" onClick={analyzeCandidate} disabled={loading}>
          {loading ? <LoaderCircle className="animate-spin" size={18} /> : <Sparkles size={18} />}
          {loading ? "Analyzing Candidate" : "Extract and Match"}
        </button>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <div
          className={dragActive ? "upload-zone active" : "upload-zone"}
          onDragOver={(event) => {
            event.preventDefault()
            setDragActive(true)
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleDrop}
          onClick={() => resumeInputRef.current?.click()}
        >
          <input
            ref={resumeInputRef}
            className="hidden"
            type="file"
            accept=".pdf,.txt,application/pdf,text/plain"
            onChange={(event) => handleResumeFile(event.target.files?.[0])}
          />
          <UploadCloud className="text-blue-600" size={28} />
          <div>
            <p className="font-semibold text-slate-900">{resumeFile?.name || "Drop resume here or choose a file"}</p>
            <p className="mt-1 text-sm text-slate-500">PDF and TXT resumes are accepted. File type is validated before analysis.</p>
          </div>
        </div>

        <div id="job-description" className="rounded-lg border border-slate-200 bg-white p-4">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <p className="font-semibold text-slate-900">Job Description</p>
              <p className="text-sm text-slate-500">{jobFileName || "Paste text or upload a TXT job description."}</p>
            </div>
            <label className="btn-secondary cursor-pointer">
              <ClipboardList size={18} />
              Upload JD
              <input className="hidden" type="file" accept=".txt,text/plain" onChange={(event) => handleJobFile(event.target.files?.[0])} />
            </label>
          </div>
          <textarea
            className="input-field min-h-40 resize-y"
            value={jobDescription}
            onChange={(event) => setJobDescription(event.target.value)}
            placeholder="Paste the role requirements, skills, experience, education, and hiring criteria."
          />
          <div className="mt-3 flex flex-wrap gap-2">
            <button className="btn-secondary" onClick={clearJobDescription} type="button">
              <RotateCcw size={18} />
              Clear
            </button>
            <button className="btn-secondary" onClick={() => document.getElementById("job-description")?.querySelector("textarea")?.focus()} type="button">
              <FileText size={18} />
              Edit
            </button>
          </div>
        </div>
      </div>

      {loading && (
        <div className="mt-5 rounded-lg border border-blue-100 bg-blue-50 p-4">
          <div className="flex items-center justify-between text-sm font-medium text-blue-800">
            <span className="inline-flex items-center gap-2">
              <LoaderCircle className="animate-spin" size={16} />
              Processing resume and job description
            </span>
            <span>{progress}%</span>
          </div>
          <div className="mt-3 h-2 overflow-hidden rounded-full bg-white">
            <div className="h-full rounded-full bg-blue-600 transition-all" style={{ width: `${progress}%` }} />
          </div>
        </div>
      )}

      {error && (
        <div className="mt-5 flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          <XCircle size={18} />
          {error}
        </div>
      )}

      {result && (
        <div className="mt-6 grid gap-4">
          <div className="grid gap-4 xl:grid-cols-[0.8fr_1.2fr]">
            <article className="rounded-lg border border-slate-200 bg-white p-5">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="section-kicker">Candidate Card</p>
                  <h3 className="mt-2 text-2xl font-semibold text-slate-950">{candidateName}</h3>
                </div>
                <span className={status === "Shortlisted" ? "badge-success" : "badge-muted"}>{status}</span>
              </div>
              <div className="mt-5 grid grid-cols-2 gap-3">
                <Metric label="Match Score" value={`${matchScore}%`} icon={Target} />
                <Metric label="Overall Rating" value={inferRating(matchScore)} icon={BadgeCheck} />
                <Metric label="Experience" value={result.experience_analysis || result.experience || result.total_experience || "Not detected"} icon={BriefcaseBusiness} />
                <Metric label="Education" value={result.education_analysis || result.education || result.highest_education || "Not detected"} icon={GraduationCap} />
              </div>
              <div className="mt-5">
                <p className="text-sm font-semibold text-slate-900">Top Skills</p>
                <TagList items={detectedSkills.slice(0, 8)} empty="No skills detected" tone="blue" />
              </div>
            </article>

            <article className="rounded-lg border border-slate-200 bg-white p-5">
              <p className="section-kicker">LLM Matching</p>
              <h3 className="mt-2 text-xl font-semibold text-slate-950">Resume vs Job Description</h3>
              <div className="mt-4 h-3 overflow-hidden rounded-full bg-slate-100">
                <div className="h-full rounded-full bg-blue-600" style={{ width: `${matchScore}%` }} />
              </div>
              <div className="mt-5 grid gap-4 md:grid-cols-2">
                <ResultBlock title="Strengths" items={strengths} fallback="Strengths were not returned by the parser." tone="emerald" />
                <ResultBlock title="Weaknesses" items={weaknesses} fallback="Weaknesses were not returned by the parser." tone="amber" />
                <ResultBlock title="Matched Skills" items={matchedSkills.length ? matchedSkills : detectedSkills.slice(0, 8)} fallback="No matched skills found." tone="blue" />
                <ResultBlock title="Missing Skills" items={missingSkills.slice(0, 10)} fallback="No missing skills found." tone="red" />
              </div>
              <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-4">
                <p className="font-semibold text-slate-900">AI Justification</p>
                <p className="mt-2 text-sm leading-6 text-slate-600">
                  {result.justification || result.ai_suggestion || `The candidate scored ${matchScore}% based on resume skills compared with job description requirements.`}
                </p>
              </div>
              <div className="mt-4 rounded-lg border border-slate-200 bg-white p-4">
                <p className="font-semibold text-slate-900">Hiring Recommendation</p>
                <p className="mt-2 text-sm leading-6 text-slate-600">
                  {result.recommendation || (status === "Shortlisted" ? "Shortlist this candidate for recruiter review." : "Review missing skills before shortlisting this candidate.")}
                </p>
              </div>
            </article>
          </div>

          <div className="grid gap-4 lg:grid-cols-3">
            <ParsingCard title="Skills" icon={CheckCircle2} items={detectedSkills} />
            <ParsingCard title="Experience" icon={BriefcaseBusiness} items={asArray(result.experience || result.total_experience)} />
            <ParsingCard title="Education" icon={GraduationCap} items={asArray(result.education || result.highest_education)} />
            <ParsingCard title="Projects" icon={BookOpen} items={asArray(result.projects)} />
            <ParsingCard title="Certifications" icon={BadgeCheck} items={asArray(result.certifications)} />
            <ParsingCard title="Languages" icon={Languages} items={asArray(result.languages)} />
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-5">
            <p className="section-kicker">Recommendations</p>
            <ResultBlock
              title="Recruiter Notes"
              items={recommendations.map((item) => (typeof item === "string" ? item : `${item.skill}: ${item.reason}`))}
              fallback={status === "Shortlisted" ? "Shortlist this candidate for recruiter review." : "Review the missing skills before shortlisting this candidate."}
              tone="blue"
            />
          </div>
        </div>
      )}
    </section>
  )
}

function Metric({ label, value, icon: Icon }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
      {createElement(Icon, { className: "text-slate-500", size: 18 })}
      <p className="mt-3 text-xs font-medium uppercase text-slate-500">{label}</p>
      <p className="mt-1 text-sm font-semibold text-slate-900">{value}</p>
    </div>
  )
}

function TagList({ items, empty, tone }) {
  const toneClass = {
    blue: "bg-blue-50 text-blue-700 border-blue-100",
    emerald: "bg-emerald-50 text-emerald-700 border-emerald-100",
    amber: "bg-amber-50 text-amber-700 border-amber-100",
    red: "bg-red-50 text-red-700 border-red-100"
  }[tone]

  return (
    <div className="mt-3 flex flex-wrap gap-2">
      {items.length ? items.map((item) => (
        <span key={item} className={`rounded-full border px-3 py-1 text-xs font-medium ${toneClass}`}>
          {item}
        </span>
      )) : <span className="text-sm text-slate-500">{empty}</span>}
    </div>
  )
}

function ResultBlock({ title, items, fallback, tone }) {
  return (
    <div>
      <p className="font-semibold text-slate-900">{title}</p>
      <TagList items={items} empty={fallback} tone={tone} />
    </div>
  )
}

function ParsingCard({ title, icon: Icon, items }) {
  return (
    <article className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="flex items-center gap-2">
        {createElement(Icon, { className: "text-blue-600", size: 18 })}
        <h3 className="font-semibold text-slate-900">{title}</h3>
      </div>
      <div className="mt-3 space-y-2">
        {items.length ? items.slice(0, 8).map((item) => (
          <p key={item} className="rounded-lg bg-slate-50 px-3 py-2 text-sm text-slate-600">
            {item}
          </p>
        )) : <p className="text-sm text-slate-500">Not detected</p>}
      </div>
    </article>
  )
}

export default ResumeAnalyzer
