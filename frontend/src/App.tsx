import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import { Startup } from './types'

function App(): JSX.Element {
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [startups, setStartups] = useState<Startup[]>([])
  const [navOpacity, setNavOpacity] = useState<number>(0)
  const [mouse, setMouse] = useState({ x: 50, y: 35 })
  const [uploading, setUploading] = useState<boolean>(false)
  const [uploadedFileName, setUploadedFileName] = useState<string>('')
  const fileInputRef = useRef<HTMLInputElement>(null)


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

  const handleSearch = async (value: string): Promise<void> => {
    setSearchTerm(value)

    if (value.trim() === '') {
      setStartups([])
      return
    }

    const response = await fetch(`/api/startups?query=${encodeURIComponent(value)}`)
    const data: Startup[] = await response.json()
    setStartups(data)
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
      setSearchTerm(extractedSkills)

      if (extractedSkills.trim() !== '') {
        const searchResponse = await fetch(
          `/api/startups?query=${encodeURIComponent(extractedSkills)}`
        )
        const searchData: Startup[] = await searchResponse.json()
        setStartups(searchData)
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
              src="src/assets/logo-mark.png"
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
              src="src/assets/logo.png"
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
              Match students to YC-backed, Series A, and Series B startups based on skills,
              interests, and experience.
            </p>
          </div>

          <div className="search-box">
            <span className="search-icon">⌕</span>

            <input
              id="search-input"
              placeholder="Python, React, NLP, data analysis, worked on LLM projects..."
              value={searchTerm}
              onChange={(e) => handleSearch(e.target.value)}
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
          </div>

          {uploadedFileName && (
            <p className="upload-status">
              {uploading ? `Parsing ${uploadedFileName}...` : `Uploaded: ${uploadedFileName}`}
            </p>
          )}
        </section>

        <section className="results-grid">
          {startups.length === 0 && searchTerm.trim() === '' && (
            <div className="glass-card empty-state">
              <h2>Start with your profile</h2>
              <p>
                Try skills like Python, machine learning, frontend development,
                backend, NLP, React, data analysis, or robotics.
              </p>
            </div>
          )}

          {startups.map((startup) => (
            <article key={startup.id} className="glass-card startup-card">
              <div className="card-top">
                <h3>{startup.name}</h3>
                <div className="score-pill">Match {startup.match_score}</div>
              </div>

              <div className="meta-row">
                <span className="meta-pill">{startup.stage}</span>
                {startup.yc_batch && <span className="meta-pill">YC {startup.yc_batch}</span>}
                <span className="meta-pill">{startup.industry}</span>
                {startup.location && <span className="meta-pill">{startup.location}</span>}
              </div>

              <p className="startup-description">{startup.description}</p>

              <div className="info-block">
                <p><strong>Tech Stack</strong></p>
                <div className="tag-row">
                  {startup.tech_stack.map((item) => (
                    <span key={item} className="soft-tag">{item}</span>
                  ))}
                </div>
              </div>

              <div className="info-block">
                <p><strong>Roles</strong></p>
                <div className="tag-row">
                  {startup.roles.map((item) => (
                    <span key={item} className="soft-tag">{item}</span>
                  ))}
                </div>
              </div>

              <div className="info-block">
                <p><strong>Matched Terms</strong></p>
                <div className="tag-row">
                  {startup.matched_terms.map((item) => (
                    <span key={item} className="soft-tag highlight-tag">{item}</span>
                  ))}
                </div>
              </div>

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