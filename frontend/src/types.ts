export interface CompanyEvidence {
  source: string
  label: string
  text: string
  score: number
}

export interface Startup {
  id?: number
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
  svd_expansion_terms?: string[]
  evidence: CompanyEvidence[]
  rag_explanation: string
}