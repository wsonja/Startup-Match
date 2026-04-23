import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import { Startup } from './types'
import logo from './assets/logo-with-slogan.png'
import logoMark from './assets/logo-mark.png'

function App(): JSX.Element {
  const [skills, setSkills] = useState<string>('')
  const [experience, setExperience] = useState<string>('')
  const [interests, setInterests] = useState<string>('')
  const [locationFilter, setLocationFilter] = useState<string>('')
  const [availableLocations, setAvailableLocations] = useState<string[]>([])
  const [availableRegions, setAvailableRegions] = useState<{ name: string; count: number }[]>([])
  const [stageFilter, setStageFilter] = useState<string>('')
  const [availableStages, setAvailableStages] = useState<string[]>([])
  const [roleFilter, setRoleFilter] = useState<string>('')
  const [availableRoles, setAvailableRoles] = useState<string[]>([])
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false)
  const [startups, setStartups] = useState<Startup[]>([])
  const [navOpacity, setNavOpacity] = useState<number>(0)
  const [mouse, setMouse] = useState({ x: 50, y: 35 })
  const [uploading, setUploading] = useState<boolean>(false)
  const [uploadedFileName, setUploadedFileName] = useState<string>('')
  const [hasSearched, setHasSearched] = useState<boolean>(false)
  const [ragExplanations, setRagExplanations] = useState<Record<string, string>>({})
  const [ragLoading, setRagLoading] = useState<Record<string, boolean>>({})
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    fetch('/api/locations')
      .then((r) => r.json())
      .then((locs: string[]) => setAvailableLocations(locs))
      .catch(() => {})
    fetch('/api/funding-stages')
      .then((r) => r.json())
      .then((stages: string[]) => setAvailableStages(stages))
      .catch(() => {})
    fetch('/api/roles')
      .then((r) => r.json())
      .then((roles: string[]) => setAvailableRoles(roles))
      .catch(() => {})
    fetch('/api/regions')
      .then((r) => r.json())
      .then((regions: { name: string; count: number }[]) => setAvailableRegions(regions))
      .catch(() => {})
  }, [])

  useEffect(() => {
    handleSearch()
  }, [locationFilter, stageFilter, roleFilter])

  useEffect(() => {
    const handleScroll = () => {
      const fadeStart = 120
      const fadeEnd = 360
      const scroll = window.scrollY

      let opacity = 0
      if (scroll > fadeStart) {
        opacity = Math.min((scroll - fadeStart) / (fadeEnd - fadeStart), 1)
      }
      setNavOpacity(opacity)
    }

    const handleMouseMove = (e: MouseEvent) => {
      const x = (e.clientX / window.innerWidth) * 100
      const y = (e.clientY / window.innerHeight) * 100
      setMouse({ x, y })
    }

    window.addEventListener('scroll', handleScroll)
    window.addEventListener('mousemove', handleMouseMove)

    return () => {
      window.removeEventListener('scroll', handleScroll)
      window.removeEventListener('mousemove', handleMouseMove)
    }
  }, [])

  useEffect(() => {
    if (!startups.length) return

    const query = [skills, experience, interests].filter(Boolean).join(' ').trim()

    startups.forEach((startup) => {
      if (ragExplanations[startup.name] || ragLoading[startup.name]) return

      setRagLoading((prev) => ({
        ...prev,
        [startup.name]: true,
      }))

      fetch('/api/rag-explanation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          startup,
          query,
        }),
      })
        .then((res) => res.json())
        .then((data) => {
          setRagExplanations((prev) => ({
            ...prev,
            [startup.name]: data.explanation || '',
          }))
        })
        .catch(() => {
          setRagExplanations((prev) => ({
            ...prev,
            [startup.name]: '',
          }))
        })
        .finally(() => {
          setRagLoading((prev) => ({
            ...prev,
            [startup.name]: false,
          }))
        })
    })
  }, [startups, skills, experience, interests, ragExplanations, ragLoading])

  const handleSearch = async (skillsOverride?: string): Promise<void> => {
    const s = skillsOverride !== undefined ? skillsOverride : skills

    const hasQuery = s.trim() !== '' || experience.trim() !== '' || interests.trim() !== ''
    const hasFilter = locationFilter.trim() !== '' || stageFilter.trim() !== '' || roleFilter.trim() !== ''

    if (!hasQuery && !hasFilter) {
      setStartups([])
      setHasSearched(false)
      setRagExplanations({})
      setRagLoading({})
      return
    }

    const params = new URLSearchParams()
    if (s.trim()) params.append('skills', s)
    if (experience.trim()) params.append('experience', experience)
    if (interests.trim()) params.append('interests', interests)
    if (locationFilter.trim()) params.append('location', locationFilter)
    if (stageFilter.trim()) params.append('stage', stageFilter)
    if (roleFilter.trim()) params.append('role', roleFilter)

    const response = await fetch(`/api/startups?${params.toString()}`)
    const data: Startup[] = await response.json()
    setStartups(data)
    setHasSearched(true)
    setRagExplanations({})
    setRagLoading({})
  }

  const handleUploadClick = () => {
    fileInputRef.current?.click()
  }

  const handleImageUpload = async (event: React.ChangeEvent<HTMLInputElement>): Promise<void> => {
    const file = event.target.files?.[0]
    if (!file) return

    setUploading(true)
    setUploadedFileName(file.name)

    try {
      const formData = new FormData()
      formData.append('image', file)

      const response = await fetch('/api/parse-skills-image', {
        method: 'POST',
        body: formData
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Failed to parse image')
      }

      const extractedSkills = data.skills?.join(', ') || ''
      setSkills(extractedSkills)

      if (extractedSkills.trim() !== '') {
        await handleSearch(extractedSkills)
      }
    } catch (error) {
      console.error('Upload error:', error)
      alert('Could not parse that image.')
    } finally {
      setUploading(false)
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  const heroStyle = useMemo(
    () => ({
      background: `
        radial-gradient(
          circle at ${mouse.x}% ${mouse.y}%,
          rgba(254, 91, 79, 0.28) 0%,
          rgba(254, 91, 79, 0.14) 18%,
          rgba(254, 91, 79, 0.05) 34%,
          rgba(243, 243, 235, 0.96) 62%
        ),
        linear-gradient(180deg, #f7f7f1 0%, #f3f3eb 55%, #efefe6 100%)
      `
    }),
    [mouse]
  )

  const scrollToSearch = () => {
    const el = document.getElementById('search-section')
    el?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <div className="app-shell">
      <header className="floating-nav" style={{ opacity: navOpacity }}>
        <div className="floating-nav-inner">
          <div className="nav-brand">
            <div className="nav-brand-mark"><img
              src={logoMark}
              alt="StartupMatch logo mark"
              className="nav-brand-mark"
            /></div>
            <span>StartupMatch</span>
          </div>
          <button className="nav-button" onClick={scrollToSearch}>
            Explore
          </button>
        </div>
      </header>

      <section className="hero" style={heroStyle}>
        <div className="hero-inner">
          <div className="hero-logo-wrap">
            <img
              src={logo}
              alt="StartupMatch logo"
              className="hero-logo"
            />
          </div>
        </div>

        <button className="scroll-indicator" onClick={scrollToSearch} aria-label="Scroll down">
          <span className="scroll-indicator-text">Scroll</span>
          <span className="scroll-indicator-arrow">↓</span>
        </button>
      </section>

      <main id="search-section" className="content-section">
        <section className="intro-card glass-card">
          <div className="intro-copy">
            <p className="eyebrow">Student to startup matching</p>
            <h1>Find early-stage startups that actually fit your skills</h1>
            <p className="intro-text">
              Enter skills directly, and optionally add experience and interests for more targeted startup matches.
            </p>
          </div>

          <div className="search-box">
            <span className="search-icon">⌕</span>

            <input
              id="search-input"
              placeholder="Skills: Python, React, NLP, SQL..."
              value={skills}
              onChange={(e) => setSkills(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') handleSearch() }}
            />

            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg,image/jpg,image/webp"
              className="hidden-file-input"
              onChange={handleImageUpload}
            />

            <button
              type="button"
              className="upload-button"
              onClick={handleUploadClick}
              disabled={uploading}
            >
              {uploading ? 'Parsing...' : 'Upload'}
            </button>

            <button
              type="button"
              className="search-button"
              onClick={() => handleSearch()}
            >
              Search →
            </button>
          </div>

          <div className="filter-row">
            <div className="location-filter-row">
              <span className="location-filter-icon">📍</span>
              <select
                className="location-select"
                value={locationFilter}
                onChange={(e) => setLocationFilter(e.target.value)}
              >
                <option value="">All locations</option>
                {availableRegions.length > 0 && (
                  <optgroup label="── Regions ──">
                    {availableRegions.map((r) => (
                      <option key={r.name} value={r.name}>{r.name}</option>
                    ))}
                  </optgroup>
                )}
                <optgroup label="── Cities ──">
                  {availableLocations.map((loc) => (
                    <option key={loc} value={loc}>{loc}</option>
                  ))}
                </optgroup>
              </select>
            </div>

            <div className="location-filter-row">
              <span className="location-filter-icon">💰</span>
              <select
                className="location-select"
                value={stageFilter}
                onChange={(e) => setStageFilter(e.target.value)}
              >
                <option value="">All stages</option>
                {availableStages.map((stage) => (
                  <option key={stage} value={stage}>{stage}</option>
                ))}
              </select>
            </div>

            <div className="location-filter-row">
              <span className="location-filter-icon">💼</span>
              <select
                className="location-select"
                value={roleFilter}
                onChange={(e) => setRoleFilter(e.target.value)}
              >
                <option value="">All roles</option>
                {availableRoles.map((role) => (
                  <option key={role} value={role}>{role}</option>
                ))}
              </select>
            </div>
          </div>

          <button
            type="button"
            className="advanced-toggle"
            onClick={() => setShowAdvanced((prev) => !prev)}
          >
            {showAdvanced ? 'Hide optional fields' : 'Add experience and interests'}
          </button>

          {showAdvanced && (
            <div className="advanced-fields">
              <textarea
                className="advanced-input"
                placeholder="Experience (optional): built a React app, worked on backend APIs, ML project..."
                value={experience}
                onChange={(e) => setExperience(e.target.value)}
              />
              <textarea
                className="advanced-input"
                placeholder="Interests (optional): fintech, healthtech, developer tools, AI infrastructure..."
                value={interests}
                onChange={(e) => setInterests(e.target.value)}
              />
            </div>
          )}
          {uploadedFileName && (
            <p className="upload-status">
              {uploading ? `Parsing ${uploadedFileName}...` : `Uploaded: ${uploadedFileName}`}
            </p>
          )}
        </section>

        <section className="results-grid">
          {!hasSearched && (
            <div className="glass-card empty-state">
              <h2>Start with your profile</h2>
              <p>
                Try skills like Python, machine learning, frontend development,
                backend, NLP, React, data analysis, or robotics.
              </p>
            </div>
          )}

          {hasSearched && startups.length === 0 && (
            <div className="glass-card empty-state">
              <h2>No matches found</h2>
              <p>Try different skills, broaden your location filter, or remove optional fields.</p>
            </div>
          )}

          {startups.map((startup) => (
            <article key={startup.id ?? startup.name} className="glass-card startup-card">
              <div className="card-top">
                <h3>{startup.name}</h3>
                <div className="score-pill">Match {startup.match_score}%</div>
              </div>

              <div className="meta-row">
                <span className="meta-pill">{startup.stage}</span>
                {startup.yc_batch && <span className="meta-pill">YC {startup.yc_batch}</span>}
                <span className="meta-pill">{startup.industry}</span>
                {startup.location && <span className="meta-pill">{startup.location}</span>}
              </div>

              <p className="startup-description">{startup.description}</p>

              {startup.tech_stack && startup.tech_stack.length > 0 && (
                <div className="info-block">
                  <p><strong>Tech Stack</strong></p>
                  <div className="tag-row">
                    {startup.tech_stack.map((item) => (
                      <span key={item} className="soft-tag">{item}</span>
                    ))}
                  </div>
                </div>
              )}

              {startup.roles && startup.roles.length > 0 && (
                <div className="info-block">
                  <p><strong>Roles</strong></p>
                  <div className="tag-row">
                    {startup.roles.map((item) => (
                      <span key={item} className="soft-tag">{item}</span>
                    ))}
                  </div>
                </div>
              )}

              <div className="info-block">
                <p><strong>Matched Terms</strong></p>
                <div className="tag-row">
                  {startup.matched_terms.map((item) => (
                    <span key={item} className="soft-tag highlight-tag">{item}</span>
                  ))}
                </div>
              </div>

              <div className="info-block">
                <p><strong>Why this matches</strong></p>
                {ragLoading[startup.name] ? (
                  <p className="startup-description">Generating AI explanation...</p>
                ) : ragExplanations[startup.name] ? (
                  <p className="startup-description">{ragExplanations[startup.name]}</p>
                ) : null}
              </div>

              {startup.svd_dimensions && startup.svd_dimensions.length > 0 && (
                <div style={{ marginTop: "12px" }}>
                  <div style={{ fontWeight: 600, marginBottom: "8px" }}>Dimensions</div>

                  {startup.svd_dimensions.slice(0, 3).map((dim) => (
                    <div
                      key={dim.dimension}
                      style={{
                        border: "1px solid rgba(0,0,0,0.08)",
                        borderRadius: "12px",
                        padding: "10px",
                        marginBottom: "8px",
                        background: "rgba(255,255,255,0.7)"
                      }}
                    >
                      <div style={{ fontWeight: 500, marginBottom: "6px" }}>
                        {dim.label} ({dim.score})
                      </div>

                      <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                        {dim.top_terms.map((item) => (
                          <span
                            key={`${dim.dimension}-${item.term}`}
                            style={{
                              fontSize: "12px",
                              padding: "4px 8px",
                              borderRadius: "999px",
                              border: "1px solid rgba(0,0,0,0.08)",
                              background: "white"
                            }}
                          >
                            {item.term} ({item.weight})
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {startup.url && (
                <a href={startup.url} target="_blank" rel="noreferrer" className="site-link">
                  Visit company →
                </a>
              )}
            </article>
          ))}
        </section>
      </main>
    </div>
  )
}

export default App