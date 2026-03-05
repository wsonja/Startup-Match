export interface Startup {
  id: number;
  name: string;
  stage: string;
  description: string;
  tags?: string;
  url?: string;
}