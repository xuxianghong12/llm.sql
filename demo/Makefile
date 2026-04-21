CC ?= gcc
CXX ?= g++
PYTHON ?= python3

SQLITE_CFLAGS := $(shell pkg-config --cflags sqlite3 2>/dev/null)
SQLITE_LIBS := $(shell pkg-config --libs sqlite3 2>/dev/null || echo -lsqlite3)

CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wpedantic
NATIVE_RUNTIME_SRC := native_graph_runtime.c
NATIVE_RUNTIME_OBJ := native_graph_runtime.o

TOY_DB ?= /tmp/toy_model.db
START_TOKEN ?= A
MAX_TOKENS ?= 3
EXTENSION_PATH ?= ../sqlite-llm/llm_ops.so
MODEL_DIR ?= /tmp/llm_sql_qwen2_05b_demo
TOKEN_CSV ?= 14990
QWEN_DB ?= model.db
PROMPT ?= hello

.PHONY: all toy-db run-c run-cpp run-qwen-c run-qwen-cpp run-qwen-str-c run-qwen-str-cpp clean

all: c_toy_sql_infer cpp_toy_sql_infer c_qwen_graph_tokens cpp_qwen_graph_tokens c_qwen_graph_string cpp_qwen_graph_string

c_toy_sql_infer: c_toy_sql_infer.c
	$(CC) $(CFLAGS) $(SQLITE_CFLAGS) -o $@ $< $(SQLITE_LIBS) -ldl -lm

cpp_toy_sql_infer: cpp_toy_sql_infer.cpp
	$(CXX) $(CXXFLAGS) $(SQLITE_CFLAGS) -o $@ $< $(SQLITE_LIBS) -ldl -lm

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

toy-db:
	$(PYTHON) make_toy_db.py --output $(TOY_DB)

run-c: c_toy_sql_infer toy-db
	LLM_SQL_EXTENSION_PATH=$(EXTENSION_PATH) ./c_toy_sql_infer $(TOY_DB) $(START_TOKEN) $(MAX_TOKENS)

run-cpp: cpp_toy_sql_infer toy-db
	LLM_SQL_EXTENSION_PATH=$(EXTENSION_PATH) ./cpp_toy_sql_infer $(TOY_DB) $(START_TOKEN) $(MAX_TOKENS)

run-qwen-c: c_qwen_graph_tokens
	LLM_SQL_EXTENSION_PATH=$(EXTENSION_PATH) ./c_qwen_graph_tokens $(MODEL_DIR) $(TOKEN_CSV) $(MAX_TOKENS) $(QWEN_DB)

run-qwen-cpp: cpp_qwen_graph_tokens
	LLM_SQL_EXTENSION_PATH=$(EXTENSION_PATH) ./cpp_qwen_graph_tokens $(MODEL_DIR) $(TOKEN_CSV) $(MAX_TOKENS) $(QWEN_DB)

run-qwen-str-c: c_qwen_graph_string
	LLM_SQL_EXTENSION_PATH=$(EXTENSION_PATH) ./c_qwen_graph_string $(MODEL_DIR) "$(PROMPT)" $(MAX_TOKENS) $(QWEN_DB)

run-qwen-str-cpp: cpp_qwen_graph_string
	LLM_SQL_EXTENSION_PATH=$(EXTENSION_PATH) ./cpp_qwen_graph_string $(MODEL_DIR) "$(PROMPT)" $(MAX_TOKENS) $(QWEN_DB)

clean:
	rm -f c_toy_sql_infer cpp_toy_sql_infer c_qwen_graph_tokens cpp_qwen_graph_tokens c_qwen_graph_string cpp_qwen_graph_string $(NATIVE_RUNTIME_OBJ)
