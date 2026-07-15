function StatCard({ title, value, caption, icon: Icon, tone = "blue" }) {
  const tones = {
    blue: "bg-blue-50 text-blue-700 border-blue-100",
    emerald: "bg-emerald-50 text-emerald-700 border-emerald-100",
    amber: "bg-amber-50 text-amber-700 border-amber-100",
    slate: "bg-slate-100 text-slate-700 border-slate-200"
  }

  return (
    <article className="surface-card p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase text-slate-500">{title}</p>
          <h2 className="mt-3 text-3xl font-semibold text-slate-950">{value}</h2>
          {caption && <p className="mt-2 text-sm text-slate-500">{caption}</p>}
        </div>
        {Icon && (
          <div className={`grid h-11 w-11 place-items-center rounded-lg border ${tones[tone]}`}>
            <Icon size={20} />
          </div>
        )}
      </div>
    </article>
  )
}

export default StatCard
