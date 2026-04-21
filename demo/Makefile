CC ?= gcc
CXX ?= g++

SQLITE_CFLAGS := $(shell pkg-config --cflags sqlite3 2>/dev/null)
SQLITE_LIBS := $(shell pkg-config --libs sqlite3 2>/dev/null || echo -lsqlite3)

CFLAGS ?= -O2 -Wall -Wextra -Wpedantic
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wpedantic
NATIVE_RUNTIME_SRC := native_graph_runtime.c
NATIVE_RUNTIME_OBJ := native_graph_runtime.o

MAX_TOKENS ?= 3
EXTENSION_PATH ?= ../sqlite-llm/llm_ops.so
MODEL_DIR ?= /tmp/llm_sql_qwen2_05b_demo
QWEN_DB ?= model.db
PROMPT ?= hello

.PHONY: all run-qwen-c run-qwen-cpp run-qwen-str-c run-qwen-str-cpp clean

all: c_qwen_graph_tokens cpp_qwen_graph_tokens c_qwen_graph_string cpp_qwen_graph_string

$(NATIVE_RUNTIME_OBJ): $(NATIVE_RUNTIME_SRC) native_graph_runtime.h
	$(CC) $(CFLAGS) $(SQLITE_CFLAGS) -c -o $@ $(NATIVE_RUNTIME_SRC)

c_qwen_graph_tokens: c_qwen_graph_tokens.c $(NATIVE_RUNTIME_OBJ) native_graph_runtime.h
	$(CC) $(CFLAGS) $(SQLITE_CFLAGS) -o $@ c_qwen_graph_tokens.c $(NATIVE_RUNTIME_OBJ) $(SQLITE_LIBS) -ldl -lm

cpp_qwen_graph_tokens: cpp_qwen_graph_tokens.cpp $(NATIVE_RUNTIME_OBJ) native_graph_runtime.h
	$(CXX) $(CXXFLAGS) $(SQLITE_CFLAGS) -o $@ cpp_qwen_graph_tokens.cpp $(NATIVE_RUNTIME_OBJ) $(SQLITE_LIBS) -ldl -lm

c_qwen_graph_string: c_qwen_graph_string.c $(NATIVE_RUNTIME_OBJ) native_graph_runtime.h
	$(CC) $(CFLAGS) $(SQLITE_CFLAGS) -o $@ c_qwen_graph_string.c $(NATIVE_RUNTIME_OBJ) $(SQLITE_LIBS) -ldl -lm

cpp_qwen_graph_string: cpp_qwen_graph_string.cpp $(NATIVE_RUNTIME_OBJ) native_graph_runtime.h
	$(CXX) $(CXXFLAGS) $(SQLITE_CFLAGS) -o $@ cpp_qwen_graph_string.cpp $(NATIVE_RUNTIME_OBJ) $(SQLITE_LIBS) -ldl -lm

run-qwen-c: c_qwen_graph_tokens
	LLM_SQL_EXTENSION_PATH=$(EXTENSION_PATH) ./c_qwen_graph_tokens $(MODEL_DIR) "$(PROMPT)" $(MAX_TOKENS) $(QWEN_DB)

run-qwen-cpp: cpp_qwen_graph_tokens
	LLM_SQL_EXTENSION_PATH=$(EXTENSION_PATH) ./cpp_qwen_graph_tokens $(MODEL_DIR) "$(PROMPT)" $(MAX_TOKENS) $(QWEN_DB)

run-qwen-str-c: c_qwen_graph_string
	LLM_SQL_EXTENSION_PATH=$(EXTENSION_PATH) ./c_qwen_graph_string $(MODEL_DIR) "$(PROMPT)" $(MAX_TOKENS) $(QWEN_DB)

run-qwen-str-cpp: cpp_qwen_graph_string
	LLM_SQL_EXTENSION_PATH=$(EXTENSION_PATH) ./cpp_qwen_graph_string $(MODEL_DIR) "$(PROMPT)" $(MAX_TOKENS) $(QWEN_DB)

clean:
	rm -f c_qwen_graph_tokens cpp_qwen_graph_tokens c_qwen_graph_string cpp_qwen_graph_string $(NATIVE_RUNTIME_OBJ)
