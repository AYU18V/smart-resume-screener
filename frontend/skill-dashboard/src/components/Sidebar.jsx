import { createElement } from "react"
import { BarChart3, ClipboardList, FileSearch, FileText, FileUp } from "lucide-react"

const items = [
  { label: "Dashboard", href: "#dashboard", icon: BarChart3, active: true },
  { label: "Upload Resume", href: "#upload-resume", icon: FileUp },
  { label: "Resume History", href: "#history", icon: ClipboardList },
  { label: "Analysis", href: "#upload-resume", icon: FileSearch },
  { label: "Reports", href: "#reports", icon: FileText }
]

function Sidebar() {
  return (
    <aside className="fixed inset-y-0 left-0 z-30 hidden w-72 border-r border-slate-200 bg-white px-5 py-6 lg:block">
      <div className="mb-8">
        <div className="grid h-10 w-10 place-items-center rounded-lg bg-slate-950 text-white">
          <FileSearch size={20} />
        </div>
        <p className="mt-4 text-xs font-semibold uppercase text-slate-500">Recruitment AI</p>
        <h2 className="mt-1 text-xl font-semibold text-slate-950">Smart Resume Screener</h2>
      </div>

      <nav className="space-y-2" aria-label="Primary navigation">
        {items.map(({ label, href, icon: Icon, active }) => (
          <a
            key={label}
            href={href}
            className={active ? "nav-item active" : "nav-item"}
          >
            {createElement(Icon, { size: 18 })}
            {label}
          </a>
        ))}
      </nav>

      <div className="mt-8 rounded-lg border border-slate-200 bg-slate-50 p-4">
        <p className="text-sm font-semibold text-slate-900">Screening Scope</p>
        <p className="mt-2 text-sm leading-6 text-slate-600">
          Resume parsing, job description matching, candidate scoring, shortlist reports.
        </p>
      </div>
    </aside>
  )
}

export default Sidebar
