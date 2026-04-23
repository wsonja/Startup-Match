export interface SvdDimensionTerm {
  term: string
  weight: number
}

export interface SvdDimension {
  dimension: number
  label: string
  score: number
  top_terms: SvdDimensionTerm[]
}

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
  svd_expansion_terms: string[]
  svd_dimensions: SvdDimension[]
}