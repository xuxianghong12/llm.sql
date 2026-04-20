# Makefile for sqlite-llm — SQLite LLM Operations Extension
#
# Targets:
#   all          build llm_ops.so (x86 SIMD + pthreads only)
#   test         build and run test_llm_ops
#   clean        remove build artefacts
#
# Requirements:
#   - GCC with AVX2 / FMA support (x86_64)
#   - libsqlite3-dev
#
# Configuration:
#   LLM_NUM_THREADS  number of worker threads for int8 parallel ops (default: 4)
#
# The .so is a position-independent shared library whose dynamic runtime
# dependencies are limited to libc, libpthread, libm, and libsqlite3.

CC        := gcc
# Configurable thread count for int8 parallel ops (default: 4)
LLM_NUM_THREADS ?= 4

# -march=native enables AVX2+FMA (and all other host features) automatically;
# explicit -mavx2/-mfma are redundant.  -ffast-math is intentionally omitted
# to preserve IEEE 754 semantics required by softmax, exp, etc.
CFLAGS    := -O3 -march=native \
             -Wall -Wextra -Wpedantic \
             -fPIC -fvisibility=hidden \
             -DLLM_NUM_THREADS=$(LLM_NUM_THREADS) \
			 -I. \
             -I/usr/include \
             -I/usr/include/x86_64-linux-gnu

# SQLite amalgamation header (use system-installed dev package)
SQLITE_INC := $(shell pkg-config --cflags sqlite3 2>/dev/null || echo "-I/usr/include")
SQLITE_LIB := $(shell pkg-config --libs   sqlite3 2>/dev/null || echo "-lsqlite3")

# ── Shared-library (extension) ────────────────────────────────────────────
SONAME    := llm_ops.so
SRCS      := llm_ops_x86.c

# ── Custom LLM page cache library ────────────────────────────────────────
PCACHE_SO := llm_pcache.so
PCACHE_SRC:= llm_pcache.c

# ── Parameter manager library ─────────────────────────────────────────────
PARAM_MGR_SO := llm_param_mgr.so
PARAM_MGR_SRC:= llm_param_mgr.c

# ── Tokenizer library ────────────────────────────────────────────────────
TOKENIZER_SO  := llm_tokenizer.so
TOKENIZER_SRC := llm_tokenizer.c

.PHONY: all test test-tokenizer clean print_llm_have_avx2

all: $(SONAME) $(PCACHE_SO) $(PARAM_MGR_SO) $(TOKENIZER_SO)

$(SONAME): $(SRCS) llm_ops.h
	$(CC) $(CFLAGS) $(SQLITE_INC) \
	    -shared -o $@ $(SRCS) \
	    $(SQLITE_LIB) \
	    -lm -lpthread
	@echo "Built $(SONAME)"
	@echo "Dynamic dependencies:"
	@ldd $(SONAME) 2>/dev/null || true

# ── LLM page cache shared library ────────────────────────────────────────
$(PCACHE_SO): $(PCACHE_SRC) llm_pcache.h
	$(CC) -O3 -Wall -Wextra -Wpedantic -fPIC -fvisibility=hidden \
	    $(SQLITE_INC) \
	    -shared -o $@ $(PCACHE_SRC) \
	    $(SQLITE_LIB) \
	    -lpthread
	@echo "Built $(PCACHE_SO)"
	@echo "Dynamic dependencies:"
	@ldd $(PCACHE_SO) 2>/dev/null || true

# ── Parameter manager shared library ──────────────────────────────────────
$(PARAM_MGR_SO): $(PARAM_MGR_SRC) llm_param_mgr.h llm_pcache.h $(PCACHE_SO)
	$(CC) -O3 -Wall -Wextra -Wpedantic -fPIC -fvisibility=hidden \
	    $(SQLITE_INC) \
	    -shared -o $@ $(PARAM_MGR_SRC) \
	    $(SQLITE_LIB) \
	    -L. -Wl,-rpath,'$$ORIGIN' -l:llm_pcache.so \
	    -lpthread
	@echo "Built $(PARAM_MGR_SO)"
	@echo "Dynamic dependencies:"
	@ldd $(PARAM_MGR_SO) 2>/dev/null || true

# ── Tokenizer shared library ──────────────────────────────────────────────
$(TOKENIZER_SO): $(TOKENIZER_SRC)
	$(CC) -O2 -Wall -Wextra -Wpedantic -fPIC -fvisibility=hidden \
	    $(SQLITE_INC) \
	    -shared -o $@ $(TOKENIZER_SRC) \
	    $(SQLITE_LIB)
	@echo "Built $(TOKENIZER_SO)"
	@echo "Dynamic dependencies:"
	@ldd $(TOKENIZER_SO) 2>/dev/null || true

# ── Tokenizer test binary ────────────────────────────────────────────────
TEST_TOKENIZER_BIN := test_llm_tokenizer

$(TEST_TOKENIZER_BIN): test_llm_tokenizer.c $(TOKENIZER_SO)
	$(CC) -O2 -Wall -Wextra -Wpedantic -o $@ test_llm_tokenizer.c \
	    $(SQLITE_INC) \
	    $(SQLITE_LIB)

test-tokenizer: $(TEST_TOKENIZER_BIN)
	./$(TEST_TOKENIZER_BIN)

# ── Test binary ───────────────────────────────────────────────────────────
TEST_BIN  := test_llm_ops

$(TEST_BIN): test_llm_ops.c $(SONAME)
	$(CC) -O2 -o $@ test_llm_ops.c \
	    $(SQLITE_INC) \
	    $(SQLITE_LIB) \
	    -lm

test: $(TEST_BIN)
	./$(TEST_BIN)

# ── Cleanup ───────────────────────────────────────────────────────────────
clean:
	rm -f $(SONAME) $(PCACHE_SO) $(PARAM_MGR_SO) $(TOKENIZER_SO) $(TEST_BIN) $(TEST_TOKENIZER_BIN)

# Print value of LLM_HAVE_AVX2 as seen by the preprocessor
print_llm_have_avx2:
	@echo -n "#sym:LLM_HAVE_AVX2="
	@val=`$(CC) $(CFLAGS) -E -dM llm_ops.h | awk '/^#define LLM_HAVE_AVX2 /{print $$3; exit}'`; \
	if [ -z "$$val" ]; then echo "(not defined)"; else echo "$$val"; fi
