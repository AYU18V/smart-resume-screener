const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"


const request = async (path, options = {}) => {
  const response = await fetch(`${API_BASE}${path}`, options)

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(error.detail || "API request failed")
  }

  return response.json()
}


export const fetchSkills = () => request("/skills")

export const fetchJobs = () => request("/top-jobs")

export const fetchTrends = () => request("/skill-trends")

export const fetchLiveJobs = () => request("/live-jobs")

export const analyzeResume = (file) => {
  const formData = new FormData()
  formData.append("file", file)

  return request("/analyze-resume", {
    method: "POST",
    body: formData
  })
}

export const extractSkills = (text) => request("/extract-skills", {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify({ text })
})

export const matchSkills = (resumeSkills, marketSkills, topK = 10) => request("/match-skills", {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    resume_skills: resumeSkills,
    market_skills: marketSkills,
    top_k: topK
  })
})

export const matchResume = (resumeText, jobDescription) => request("/match-resume", {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    resume_text: resumeText,
    job_description: jobDescription
  })
})

export const fetchForecast = (months = 6) => request(`/forecast-skills?months=${months}`)

export const fetchDatasetInsights = () => request("/dataset-insights")

export const fetchModelExplainability = () => request("/model-explainability")

export const predictDemand = (skill, months = 6) => request("/predict-demand", {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify({ skill, months })
})

export const fetchWorkforceAnalytics = () => request("/workforce-analytics")

export const fetchCareerIntelligence = (skills = [], text = "") => request("/career-intelligence", {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify({ skills, text })
})
