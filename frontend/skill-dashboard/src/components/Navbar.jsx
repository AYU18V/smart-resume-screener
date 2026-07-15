import { Moon, UserCircle } from "lucide-react"

function Navbar() {
  return (
    <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/90 px-4 py-4 backdrop-blur sm:px-6 lg:px-8">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase text-slate-500">Smart Resume Screener</p>
          <h1 className="text-lg font-semibold text-slate-950 sm:text-xl">
            Resume Screening Dashboard
          </h1>
        </div>

        <div className="flex items-center gap-2">
          <button className="icon-button" aria-label="Theme toggle">
            <Moon size={17} />
          </button>
          <button className="icon-button" aria-label="User profile">
            <UserCircle size={18} />
          </button>
        </div>
      </div>
    </header>
  )
}

export default Navbar
