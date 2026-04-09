package agent

import (
	"sync"
	"time"

	"github.com/DennisMRitchie/go-llm-agent-framework/internal/llmclient"
	"github.com/DennisMRitchie/go-llm-agent-framework/internal/nlp"
)

// MemoryEntry is a single turn in the conversation.
type MemoryEntry struct {
	Role      string // "user" | "assistant" | "system"
	Content   string
	Timestamp time.Time
	Metadata  map[string]any
	embedding []float64 // lazy-computed
}

// Memory manages conversation history for an agent session.
type Memory struct {
	mu           sync.RWMutex
	entries      []*MemoryEntry
	maxEntries   int
	preprocessor *nlp.Preprocessor
}

func NewMemory(maxEntries int, preprocessor *nlp.Preprocessor) *Memory {
	return &Memory{
		maxEntries:   maxEntries,
		preprocessor: preprocessor,
	}
}

// Add appends a new entry, evicting the oldest non-system message if at capacity.
func (m *Memory) Add(role, content string, metadata map[string]any) {
	m.mu.Lock()
	defer m.mu.Unlock()

	entry := &MemoryEntry{
		Role:      role,
		Content:   content,
		Timestamp: time.Now(),
		Metadata:  metadata,
	}

	m.entries = append(m.entries, entry)

	// Evict oldest non-system entries when over limit
	for len(m.entries) > m.maxEntries {
		for i, e := range m.entries {
			if e.Role != "system" {
				m.entries = append(m.entries[:i], m.entries[i+1:]...)
				break
			}
		}
		// Failsafe: if all are system, evict oldest
		if len(m.entries) > m.maxEntries {
			m.entries = m.entries[1:]
		}
	}
}

// AddSystem inserts a system message at the front (persists across eviction).
func (m *Memory) AddSystem(content string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.entries = append([]*MemoryEntry{{
		Role:      "system",
		Content:   content,
		Timestamp: time.Now(),
	}}, m.entries...)
}

// Messages returns the full history as LLM client messages.
func (m *Memory) Messages() []llmclient.Message {
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := make([]llmclient.Message, len(m.entries))
	for i, e := range m.entries {
		out[i] = llmclient.Message{
			Role:      e.Role,
			Content:   e.Content,
			Timestamp: e.Timestamp.UnixMilli(),
		}
	}
	return out
}

// Search returns the top-K most semantically similar memories to the query.
func (m *Memory) Search(query string, topK int) []*MemoryEntry {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(m.entries) == 0 {
		return nil
	}

	pt := m.preprocessor.Process(query)
	qVec := m.preprocessor.Embedding(pt.Tokens)

	type scored struct {
		entry *MemoryEntry
		score float64
	}
	results := make([]scored, 0, len(m.entries))

	for _, e := range m.entries {
		if e.embedding == nil {
			ept := m.preprocessor.Process(e.Content)
			e.embedding = m.preprocessor.Embedding(ept.Tokens)
		}
		sim := nlp.CosineSimilarity(qVec, e.embedding)
		results = append(results, scored{e, sim})
	}

	// Partial sort: find top K
	for i := 0; i < len(results) && i < topK; i++ {
		maxIdx := i
		for j := i + 1; j < len(results); j++ {
			if results[j].score > results[maxIdx].score {
				maxIdx = j
			}
		}
		results[i], results[maxIdx] = results[maxIdx], results[i]
	}

	if topK > len(results) {
		topK = len(results)
	}
	out := make([]*MemoryEntry, topK)
	for i := range out {
		out[i] = results[i].entry
	}
	return out
}

// Clear wipes all memory entries.
func (m *Memory) Clear() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.entries = nil
}

// Len returns the number of stored entries.
func (m *Memory) Len() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.entries)
}

// Last returns the most recent N entries in chronological order.
func (m *Memory) Last(n int) []*MemoryEntry {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if n >= len(m.entries) {
		return m.entries
	}
	return m.entries[len(m.entries)-n:]
}
