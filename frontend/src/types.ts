export interface Startup {
  id: number
  name: string
  stage: string
  yc_batch?: string
  industry: string
  location?: string
  description: string
  tech_stack: string[]
  roles: string[]
  keywords: string[]
  url?: string
  match_score: number
  matched_terms: string[]
}