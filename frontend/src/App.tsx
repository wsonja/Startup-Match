import { useState } from 'react'
import './App.css'
import SearchIcon from './assets/mag.png'
import { Startup } from './types'

function App(): JSX.Element {
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [startups, setStartups] = useState<Startup[]>([])

  const handleSearch = async (value: string): Promise<void> => {
    setSearchTerm(value)

    if (value.trim() === '') {
      setStartups([])
      return
    }

    try {
      const response = await fetch(`/api/startups/search?q=${encodeURIComponent(value)}`)
      const data: Startup[] = await response.json()
      setStartups(data)
    } catch (error) {
      console.error('Error fetching startups:', error)
    }
  }

  return (
    <div className="full-body-container">
      <div className="top-text">

        <h1>StartupMatch</h1>

        <div className="input-box" onClick={() => document.getElementById('search-input')?.focus()}>
          <img src={SearchIcon} alt="search" />

          <input
            id="search-input"
            placeholder="Search for startups by skills (Python, ML, React...)"
            value={searchTerm}
            onChange={(e) => handleSearch(e.target.value)}
          />

        </div>
      </div>

      <div id="answer-box">

        {startups.map((startup) => (
          <div key={startup.id} className="episode-item">

            <h3 className="episode-title">
              {startup.name} ({startup.stage})
            </h3>

            <p className="episode-desc">
              {startup.description}
            </p>

            {startup.tags && (
              <p><strong>Tech:</strong> {startup.tags}</p>
            )}

            {startup.url && (
              <a href={startup.url} target="_blank">
                Visit Company
              </a>
            )}

          </div>
        ))}

      </div>
    </div>
  )
}

export default App