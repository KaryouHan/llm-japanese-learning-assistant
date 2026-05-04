<script setup lang="ts">
import { computed, onMounted, ref } from "vue";

type GrammarPoint = {
  pattern: string;
  explanation: string;
  example: string;
};

type VocabularyItem = {
  word: string;
  reading: string;
  meaning: string;
  note: string;
};

type PracticeQuestion = {
  question: string;
  answer: string;
  explanation: string;
};

type AnalyzeResponse = {
  summary: string;
  natural_translation: string;
  grammar_points: GrammarPoint[];
  vocabulary: VocabularyItem[];
  nuance: string;
  examples: string[];
  practice_questions: PracticeQuestion[];
  model_used: string;
  source: "mock" | "llm";
};

type KnowledgeStatus = {
  raw_pdf_count: number;
  uploaded_pdf_count: number;
  indexed_document_count: number;
  indexed_chunk_count: number;
  indexed_sentence_count: number;
  index_exists: boolean;
};

type RelatedExample = {
  source: string;
  year: string | null;
  month: string | null;
  section: string;
  question_id: string | null;
  page: number;
  related_pattern: string | null;
  excerpt: string;
  why_related: string;
  score: number;
};

type RelatedExamplesResponse = {
  detected_patterns: string[];
  related_examples: RelatedExample[];
  study_note: string;
};

const API_BASE = "http://localhost:8000";

const activeTab = ref<"analysis" | "related">("analysis");

const sentence = ref("昨日、友達に日本語を教えてもらいました。");
const jlptLevel = ref("N4");
const focus = ref("grammar");
const loading = ref(false);
const errorMessage = ref("");
const result = ref<AnalyzeResponse | null>(null);

const relatedSentence = ref("雨が降らないとも限らない。");
const relatedTopK = ref(5);
const relatedLoading = ref(false);
const relatedError = ref("");
const relatedResult = ref<RelatedExamplesResponse | null>(null);
const knowledgeStatus = ref<KnowledgeStatus | null>(null);
const selectedPdf = ref<File | null>(null);
const uploadMessage = ref("");
const ingestLoading = ref(false);
const uploadLoading = ref(false);

const canSubmit = computed(() => sentence.value.trim().length > 0 && !loading.value);
const canFindRelated = computed(
  () => relatedSentence.value.trim().length > 0 && !relatedLoading.value,
);

onMounted(() => {
  refreshKnowledgeStatus();
});

async function analyzeSentence() {
  if (!canSubmit.value) return;

  loading.value = true;
  errorMessage.value = "";

  try {
    const response = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sentence: sentence.value,
        jlpt_level: jlptLevel.value,
        focus: focus.value,
      }),
    });

    if (!response.ok) {
      const errorBody = await response.json().catch(() => null);
      throw new Error(
        errorBody?.detail ?? `Request failed with status ${response.status}`,
      );
    }

    result.value = await response.json();
  } catch (error) {
    errorMessage.value =
      error instanceof Error ? error.message : "Something went wrong.";
  } finally {
    loading.value = false;
  }
}

async function refreshKnowledgeStatus() {
  try {
    const response = await fetch(`${API_BASE}/api/knowledge/status`);
    if (!response.ok) return;
    knowledgeStatus.value = await response.json();
  } catch {
    knowledgeStatus.value = null;
  }
}

function onPdfSelected(event: Event) {
  const input = event.target as HTMLInputElement;
  selectedPdf.value = input.files?.[0] ?? null;
  uploadMessage.value = "";
}

async function uploadPdf() {
  if (!selectedPdf.value) return;

  uploadLoading.value = true;
  uploadMessage.value = "";
  relatedError.value = "";

  try {
    const formData = new FormData();
    formData.append("file", selectedPdf.value);

    const response = await fetch(`${API_BASE}/api/knowledge/upload`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorBody = await response.json().catch(() => null);
      throw new Error(errorBody?.detail ?? "Upload failed.");
    }

    const payload = await response.json();
    uploadMessage.value = `Uploaded ${payload.filename}`;
    selectedPdf.value = null;
    await refreshKnowledgeStatus();
  } catch (error) {
    relatedError.value =
      error instanceof Error ? error.message : "Something went wrong.";
  } finally {
    uploadLoading.value = false;
  }
}

async function ingestKnowledgeBase() {
  ingestLoading.value = true;
  uploadMessage.value = "";
  relatedError.value = "";

  try {
    const response = await fetch(`${API_BASE}/api/knowledge/ingest`, {
      method: "POST",
    });

    if (!response.ok) {
      const errorBody = await response.json().catch(() => null);
      throw new Error(errorBody?.detail ?? "Ingestion failed.");
    }

    const payload = await response.json();
    uploadMessage.value = `Indexed ${payload.document_count} documents and ${payload.chunk_count} chunks.`;
    await refreshKnowledgeStatus();
  } catch (error) {
    relatedError.value =
      error instanceof Error ? error.message : "Something went wrong.";
  } finally {
    ingestLoading.value = false;
  }
}

async function findRelatedExamples() {
  if (!canFindRelated.value) return;

  relatedLoading.value = true;
  relatedError.value = "";

  try {
    const response = await fetch(`${API_BASE}/api/knowledge/related`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sentence: relatedSentence.value,
        jlpt_level: "N1",
        top_k: relatedTopK.value,
      }),
    });

    if (!response.ok) {
      const errorBody = await response.json().catch(() => null);
      throw new Error(errorBody?.detail ?? "Related search failed.");
    }

    relatedResult.value = await response.json();
  } catch (error) {
    relatedError.value =
      error instanceof Error ? error.message : "Something went wrong.";
  } finally {
    relatedLoading.value = false;
  }
}
</script>

<template>
  <main class="app-shell">
    <section class="workspace">
      <div class="editor-pane">
        <p class="eyebrow">LLM Japanese Learning Assistant</p>
        <h1>Practice Japanese with structured LLM feedback</h1>

        <div class="tabs">
          <button
            class="tab-button"
            :class="{ active: activeTab === 'analysis' }"
            @click="activeTab = 'analysis'"
          >
            Sentence Analysis
          </button>
          <button
            class="tab-button"
            :class="{ active: activeTab === 'related' }"
            @click="activeTab = 'related'"
          >
            Related N1 Examples
          </button>
        </div>

        <template v-if="activeTab === 'analysis'">
          <label class="field">
            <span>Japanese sentence</span>
            <textarea v-model="sentence" rows="6" />
          </label>

          <div class="controls">
            <label class="field">
              <span>JLPT level</span>
              <select v-model="jlptLevel">
                <option>N5</option>
                <option>N4</option>
                <option>N3</option>
                <option>N2</option>
                <option>N1</option>
              </select>
            </label>

            <label class="field">
              <span>Focus</span>
              <select v-model="focus">
                <option value="general">General</option>
                <option value="grammar">Grammar</option>
                <option value="vocabulary">Vocabulary</option>
                <option value="nuance">Nuance</option>
                <option value="exam">Exam</option>
              </select>
            </label>
          </div>

          <button :disabled="!canSubmit" @click="analyzeSentence">
            {{ loading ? "Analyzing..." : "Analyze sentence" }}
          </button>

          <p v-if="errorMessage" class="error">{{ errorMessage }}</p>
        </template>

        <template v-else>
          <div class="status-panel">
            <strong>Local knowledge base</strong>
            <p v-if="knowledgeStatus">
              {{ knowledgeStatus.indexed_document_count }} documents /
              {{ knowledgeStatus.indexed_sentence_count }} sentence records indexed
            </p>
            <p v-else>Backend status unavailable.</p>
          </div>

          <label class="field">
            <span>Upload local PDF</span>
            <input type="file" accept="application/pdf" @change="onPdfSelected" />
          </label>

          <div class="button-row">
            <button
              class="secondary-button"
              :disabled="!selectedPdf || uploadLoading"
              @click="uploadPdf"
            >
              {{ uploadLoading ? "Uploading..." : "Upload PDF" }}
            </button>
            <button
              class="secondary-button"
              :disabled="ingestLoading"
              @click="ingestKnowledgeBase"
            >
              {{ ingestLoading ? "Indexing..." : "Build index" }}
            </button>
          </div>

          <p v-if="uploadMessage" class="success">{{ uploadMessage }}</p>

          <label class="field">
            <span>Japanese sentence</span>
            <textarea v-model="relatedSentence" rows="5" />
          </label>

          <label class="field">
            <span>Related examples</span>
            <select v-model="relatedTopK">
              <option :value="3">Top 3</option>
              <option :value="5">Top 5</option>
              <option :value="8">Top 8</option>
            </select>
          </label>

          <button :disabled="!canFindRelated" @click="findRelatedExamples">
            {{ relatedLoading ? "Searching..." : "Find related N1 examples" }}
          </button>

          <p v-if="relatedError" class="error">{{ relatedError }}</p>
        </template>
      </div>

      <div class="result-pane">
        <template v-if="activeTab === 'analysis'">
          <template v-if="result">
            <div class="result-header">
              <div>
                <p class="eyebrow">Result</p>
                <h2>{{ result.natural_translation }}</h2>
              </div>
              <span class="pill">{{ result.source }} / {{ result.model_used }}</span>
            </div>

            <p v-if="result.source === 'mock'" class="mock-warning">
              Demo mode is active. This response is a placeholder because no model
              API key is configured yet.
            </p>

            <section>
              <h3>Summary</h3>
              <p>{{ result.summary }}</p>
            </section>

            <section>
              <h3>Grammar</h3>
              <article v-for="item in result.grammar_points" :key="item.pattern">
                <strong>{{ item.pattern }}</strong>
                <p>{{ item.explanation }}</p>
                <small>{{ item.example }}</small>
              </article>
            </section>

            <section>
              <h3>Vocabulary</h3>
              <div class="vocab-grid">
                <article v-for="item in result.vocabulary" :key="item.word">
                  <strong>{{ item.word }}</strong>
                  <span>{{ item.reading }}</span>
                  <p>{{ item.meaning }}</p>
                  <small>{{ item.note }}</small>
                </article>
              </div>
            </section>

            <section>
              <h3>Nuance</h3>
              <p>{{ result.nuance }}</p>
            </section>

            <section>
              <h3>Examples</h3>
              <ul>
                <li v-for="example in result.examples" :key="example">{{ example }}</li>
              </ul>
            </section>

            <section>
              <h3>Practice</h3>
              <article
                v-for="question in result.practice_questions"
                :key="question.question"
              >
                <strong>{{ question.question }}</strong>
                <p>Answer: {{ question.answer }}</p>
                <small>{{ question.explanation }}</small>
              </article>
            </section>
          </template>

          <div v-else class="empty-state">
            <h2>Enter a sentence to begin</h2>
            <p>
              The first version uses a mock response. Add an API key later to turn
              this into a real LLM-powered tutor.
            </p>
          </div>
        </template>

        <template v-else>
          <template v-if="relatedResult">
            <div class="result-header">
              <div>
                <p class="eyebrow">Related N1 Examples</p>
                <h2>Similar grammar and exam contexts</h2>
              </div>
              <span class="pill">{{ relatedResult.related_examples.length }} results</span>
            </div>

            <section>
              <h3>Detected pattern</h3>
              <div v-if="relatedResult.detected_patterns.length" class="tag-row">
                <span
                  v-for="pattern in relatedResult.detected_patterns"
                  :key="pattern"
                  class="tag"
                >
                  {{ pattern }}
                </span>
              </div>
              <p v-else>No explicit grammar pattern detected.</p>
            </section>

            <section>
              <h3>Related N1 examples</h3>
              <article
                v-for="(example, index) in relatedResult.related_examples"
                :key="`${example.source}-${example.page}-${index}`"
              >
                <div class="example-meta">
                  <strong>
                    {{ index + 1 }}. {{ example.year ?? "Unknown" }}-{{
                      example.month ?? "--"
                    }}
                    {{ example.section }}
                    {{ example.question_id ?? "" }}
                  </strong>
                  <span>page {{ example.page }}</span>
                </div>
                <p v-if="example.related_pattern">
                  Related pattern: {{ example.related_pattern }}
                </p>
                <p>{{ example.why_related }}</p>
                <blockquote>{{ example.excerpt }}</blockquote>
                <small>{{ example.source }} / score {{ example.score }}</small>
              </article>
            </section>

            <section>
              <h3>Study note</h3>
              <p>{{ relatedResult.study_note }}</p>
            </section>
          </template>

          <div v-else class="empty-state">
            <h2>Build a local N1 knowledge base</h2>
            <p>
              Upload text-based JLPT PDFs, build the local index, then search for
              related N1 examples by sentence.
            </p>
          </div>
        </template>
      </div>
    </section>
  </main>
</template>
