package nlp

import (
	"context"
	"log/slog"
	"math"
	"regexp"
	"strings"
	"unicode"

	"github.com/DennisMRitchie/go-llm-agent-framework/internal/config"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/llmclient"
)

type Token struct {
	Text    string
	Lemma   string
	IsStop  bool
	IsAlpha bool
	Index   int
}

type ProcessedText struct {
	Original   string
	Normalized string
	Tokens     []Token
	Sentences  []string
	WordCount  int
	CharCount  int
	Language   string
}

var stopwords = map[string]bool{
	"a": true, "an": true, "the": true, "is": true, "are": true, "was": true,
	"were": true, "be": true, "been": true, "being": true, "have": true,
	"has": true, "had": true, "do": true, "does": true, "did": true, "will": true,
	"would": true, "shall": true, "should": true, "may": true, "might": true,
	"must": true, "can": true, "could": true, "to": true, "of": true, "in": true,
	"for": true, "on": true, "with": true, "at": true, "by": true, "from": true,
	"that": true, "this": true, "it": true, "its": true, "and": true, "or": true,
	"but": true, "not": true, "no": true, "so": true, "as": true,
}

var (
	rePunct   = regexp.MustCompile(`[^\w\s]`)
	reMultiSp = regexp.MustCompile(`\s+`)
	reSentEnd = regexp.MustCompile(`[.!?]+\s+`)
)

type Preprocessor struct {
	cfg    config.NLPConfig
	client llmclient.Client
	logger *slog.Logger
}

func NewPreprocessor(cfg config.NLPConfig, client llmclient.Client, logger *slog.Logger) *Preprocessor {
	return &Preprocessor{cfg: cfg, client: client, logger: logger}
}

func (p *Preprocessor) Process(text string) *ProcessedText {
	normalized := text
	if p.cfg.EnableNormalize {
		normalized = normalize(text)
	}
	tokens := tokenize(normalized, p.cfg.EnableStopwords)
	if p.cfg.MaxTokens > 0 && len(tokens) > p.cfg.MaxTokens {
		tokens = tokens[:p.cfg.MaxTokens]
	}
	return &ProcessedText{
		Original:   text,
		Normalized: normalized,
		Tokens:     tokens,
		Sentences:  splitSentences(text),
		WordCount:  countWords(text),
		CharCount:  len(text),
		Language:   detectLanguage(text),
	}
}

func (p *Preprocessor) Keywords(text string, topN int) []string {
	pt := p.Process(text)
	freq := make(map[string]int)
	for _, tok := range pt.Tokens {
		if tok.IsAlpha && !tok.IsStop {
			freq[strings.ToLower(tok.Text)]++
		}
	}
	return topByFreq(freq, topN)
}

func (p *Preprocessor) Embedding(tokens []Token) []float64 {
	freq := make(map[string]float64)
	for _, tok := range tokens {
		if tok.IsAlpha && !tok.IsStop {
			freq[strings.ToLower(tok.Text)]++
		}
	}
	words := make([]string, 0, len(freq))
	for w := range freq {
		words = append(words, w)
	}
	vec := make([]float64, len(words))
	total := float64(len(tokens))
	if total == 0 {
		return vec
	}
	for i, w := range words {
		tf := freq[w] / total
		vec[i] = tf * (1 + math.Log(1+freq[w]))
	}
	return normalizeVec(vec)
}

func CosineSimilarity(a, b []float64) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	size := len(a)
	if len(b) > size {
		size = len(b)
	}
	va := padVec(a, size)
	vb := padVec(b, size)
	dot, normA, normB := 0.0, 0.0, 0.0
	for i := range va {
		dot += va[i] * vb[i]
		normA += va[i] * va[i]
		normB += vb[i] * vb[i]
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}

// IntentResult holds classification output.
type IntentResult struct {
	Intent     string
	Confidence float64
	AllScores  map[string]float64
}

type Classifier struct {
	cfg    config.NLPConfig
	client llmclient.Client
	logger *slog.Logger
}

func NewClassifier(cfg config.NLPConfig, client llmclient.Client, logger *slog.Logger) *Classifier {
	return &Classifier{cfg: cfg, client: client, logger: logger}
}

func (c *Classifier) Classify(ctx context.Context, text string, labels []string) (*IntentResult, error) {
	if c.client != nil {
		resp, err := c.client.ClassifyIntent(ctx, &llmclient.ClassifyRequest{
			Text:            text,
			CandidateLabels: labels,
		})
		if err == nil {
			return &IntentResult{
				Intent:     resp.TopLabel,
				Confidence: resp.Confidence,
				AllScores:  resp.AllScores,
			}, nil
		}
		c.logger.Warn("LLM classifier failed, using heuristic fallback", "error", err)
	}
	return c.heuristicClassify(text, labels), nil
}

func (c *Classifier) heuristicClassify(text string, labels []string) *IntentResult {
	lower := strings.ToLower(text)
	scores := make(map[string]float64, len(labels))
	for _, label := range labels {
		words := strings.Fields(strings.ToLower(label))
		score := 0.0
		for _, w := range words {
			if strings.Contains(lower, w) {
				score += 1.0 / float64(len(words))
			}
		}
		scores[label] = score
	}
	best, bestScore := "", -1.0
	for l, s := range scores {
		if s > bestScore {
			best, bestScore = l, s
		}
	}
	if bestScore < c.cfg.ConfidenceThresh {
		best = "unknown"
	}
	return &IntentResult{Intent: best, Confidence: bestScore, AllScores: scores}
}

type EntityExtractor struct {
	cfg    config.NLPConfig
	client llmclient.Client
	logger *slog.Logger
}

func NewEntityExtractor(cfg config.NLPConfig, client llmclient.Client, logger *slog.Logger) *EntityExtractor {
	return &EntityExtractor{cfg: cfg, client: client, logger: logger}
}

func (e *EntityExtractor) Extract(ctx context.Context, text string) ([]llmclient.Entity, error) {
	if e.client != nil {
		resp, err := e.client.ExtractEntities(ctx, &llmclient.ExtractRequest{Text: text})
		if err == nil {
			return resp.Entities, nil
		}
		e.logger.Warn("LLM entity extraction failed, using regex fallback", "error", err)
	}
	return e.regexExtract(text), nil
}

var (
	reEmail  = regexp.MustCompile(`\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b`)
	reURL    = regexp.MustCompile(`https?://[^\s]+`)
	rePhone  = regexp.MustCompile(`\b(\+?[0-9]{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b`)
	reNumber = regexp.MustCompile(`\b\d+(\.\d+)?\b`)
)

func (e *EntityExtractor) regexExtract(text string) []llmclient.Entity {
	var entities []llmclient.Entity
	add := func(matches [][]int, label string) {
		for _, loc := range matches {
			entities = append(entities, llmclient.Entity{
				Text: text[loc[0]:loc[1]], Label: label,
				Start: loc[0], End: loc[1], Confidence: 0.75,
			})
		}
	}
	add(reEmail.FindAllStringIndex(text, -1), "EMAIL")
	add(reURL.FindAllStringIndex(text, -1), "URL")
	add(rePhone.FindAllStringIndex(text, -1), "PHONE")
	add(reNumber.FindAllStringIndex(text, -1), "NUMBER")
	return entities
}

// --- helpers ---

func normalize(text string) string {
	text = strings.ToLower(text)
	text = rePunct.ReplaceAllString(text, " ")
	text = reMultiSp.ReplaceAllString(text, " ")
	return strings.TrimSpace(text)
}

func tokenize(text string, markStops bool) []Token {
	words := strings.Fields(text)
	tokens := make([]Token, len(words))
	for i, w := range words {
		isAlpha := true
		for _, ch := range w {
			if !unicode.IsLetter(ch) {
				isAlpha = false
				break
			}
		}
		tokens[i] = Token{
			Text: w, Lemma: strings.ToLower(w),
			IsStop:  markStops && stopwords[strings.ToLower(w)],
			IsAlpha: isAlpha, Index: i,
		}
	}
	return tokens
}

func splitSentences(text string) []string {
	parts := reSentEnd.Split(text, -1)
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if s := strings.TrimSpace(p); s != "" {
			out = append(out, s)
		}
	}
	return out
}

func countWords(text string) int { return len(strings.Fields(text)) }

func detectLanguage(text string) string {
	for _, ch := range text {
		if ch >= 0x0400 && ch <= 0x04FF {
			return "ru"
		}
	}
	return "en"
}

func topByFreq(freq map[string]int, n int) []string {
	type kv struct {
		key string
		val int
	}
	pairs := make([]kv, 0, len(freq))
	for k, v := range freq {
		pairs = append(pairs, kv{k, v})
	}
	for i := 0; i < len(pairs)-1 && i < n; i++ {
		maxIdx := i
		for j := i + 1; j < len(pairs); j++ {
			if pairs[j].val > pairs[maxIdx].val {
				maxIdx = j
			}
		}
		pairs[i], pairs[maxIdx] = pairs[maxIdx], pairs[i]
	}
	if n > len(pairs) {
		n = len(pairs)
	}
	out := make([]string, n)
	for i := range out {
		out[i] = pairs[i].key
	}
	return out
}

func normalizeVec(v []float64) []float64 {
	norm := 0.0
	for _, x := range v {
		norm += x * x
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		return v
	}
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = x / norm
	}
	return out
}

func padVec(v []float64, size int) []float64 {
	if len(v) == size {
		return v
	}
	out := make([]float64, size)
	copy(out, v)
	return out
}
