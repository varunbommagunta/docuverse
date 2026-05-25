export interface Document {
  document_id: string;
  filename: string;
  chunk_count: number;
  document_type: string;
  classification_confidence: number;
  classification_method: string;
  chunker_used: string;
}

export interface ArticleFilterDebug {
  matched: boolean;
  article_id?: string | null;
  pinned_count: number;
}

export interface RerankerDebug {
  candidates_in: number;
  results_out: number;
}

export interface ChunkDebug {
  id: string;
  score: number;
  pinned: boolean;
  source: string;
  article_id?: string | null;
  section_title?: string | null;
  preview: string;
}

export interface QueryDebug {
  original_query: string;
  rewritten_query: string;
  article_filter: ArticleFilterDebug;
  retrieval_strategy: string;
  chunks: ChunkDebug[];
  reranker?: RerankerDebug | null;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: number[];
  chunks?: Array<{
    chunk_index: number;
    chunk_id: string;
    text: string;
    score: number;
    metadata: Record<string, unknown>;
  }>;
  debug?: QueryDebug;
}
