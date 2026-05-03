<script setup lang="ts">
import { computed, ref } from "vue";

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

const sentence = ref("昨日、友達に日本語を教えてもらいました。");
const jlptLevel = ref("N4");
const focus = ref("grammar");
const loading = ref(false);
const errorMessage = ref("");
const result = ref<AnalyzeResponse | null>(null);

const canSubmit = computed(() => sentence.value.trim().length > 0 && !loading.value);

async function analyzeSentence() {
  if (!canSubmit.value) return;

  loading.value = true;
  errorMessage.value = "";

  try {
    const response = await fetch("http://localhost:8000/api/analyze", {
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
</script>

<template>
  <main class="app-shell">
    <section class="workspace">
      <div class="editor-pane">
        <p class="eyebrow">LLM Japanese Learning Assistant</p>
        <h1>Practice Japanese with structured LLM feedback</h1>

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
      </div>

      <div class="result-pane">
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
            <article v-for="question in result.practice_questions" :key="question.question">
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
      </div>
    </section>
  </main>
</template>
