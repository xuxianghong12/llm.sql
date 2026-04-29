# Makefile for sqlite-llm — SQLite LLM Operations Extension
#
# Targets:
#   all          build llm_ops.so (x86 split implementation)
#   arm          build llm_ops_arm.so (ARM baseline implementation)
#   test         build and run test_llm_ops_x86
#   test-arm     build and run test_llm_ops_arm
#   clean        remove build artefacts
#
# Requirements:
#   - GCC or clang
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
DEB_MULTIARCH := $(shell $(CC) -print-multiarch 2>/dev/null)

ifneq ($(strip $(DEB_MULTIARCH)),)
MULTIARCH_INC := -I/usr/include/$(DEB_MULTIARCH)
endif

# -march=native enables the host ISA automatically; x86 fast paths stay gated
# behind LLM_HAVE_AVX2 while the initial ARM port forces the portable scalar
# path until NEON-specific kernels are added.
COMMON_CFLAGS := -O3 -march=native \
				 -Wall -Wextra -Wpedantic \
				 -fPIC -fvisibility=hidden \
				 -DLLM_NUM_THREADS=$(LLM_NUM_THREADS) \
				 -I. \
				 -I/usr/include \
				 $(MULTIARCH_INC)
X86_CFLAGS   := $(COMMON_CFLAGS) -Ix86
ARM_CFLAGS   := $(COMMON_CFLAGS) -Iarm

# SQLite amalgamation header (use system-installed dev package)
SQLITE_INC := $(shell pkg-config --cflags sqlite3 2>/dev/null || echo "-I/usr/include")
SQLITE_LIB := $(shell pkg-config --libs   sqlite3 2>/dev/null || echo "-lsqlite3")

# ── Shared-library (extension) ────────────────────────────────────────────
SONAME    := llm_ops.so
SRCS      := x86/llm_ops_x86.c \
	           x86/llm_ops_x86_elementwise.c \
	           x86/llm_ops_x86_reduce.c \
	           x86/llm_ops_x86_norm.c \
	           x86/llm_ops_x86_shape.c \
	           x86/llm_ops_x86_common.c \
	           x86/llm_ops_x86_blob.c \
	           x86/llm_ops_x86_nd_layout.c \
	           x86/llm_ops_x86_nd_stats.c \
	           x86/llm_ops_x86_nd_algebra.c \
	           x86/llm_ops_x86_nd_pointwise.c \
	           x86/llm_ops_x86_nd_reduce.c \
	           x86/llm_ops_x86_nd_utility.c \
	           x86/llm_ops_x86_nd_sequence.c \
	           x86/llm_ops_x86_nd_model.c \
	           x86/llm_ops_x86_int8.c \
	           x86/llm_ops_x86_matrix.c \
	           x86/llm_ops_x86_transformer.c \
	           x86/llm_ops_x86_runtime.c \
	           x86/llm_ops_x86_lookup.c

ARM_SONAME := llm_ops_arm.so
ARM_SRCS   := arm/llm_ops_arm.c \
	             arm/llm_ops_arm_elementwise.c \
	             arm/llm_ops_arm_reduce.c \
	             arm/llm_ops_arm_norm.c \
	             arm/llm_ops_arm_shape.c \
	             arm/llm_ops_arm_common.c \
	             arm/llm_ops_arm_blob.c \
	             arm/llm_ops_arm_nd_layout.c \
	             arm/llm_ops_arm_nd_stats.c \
	             arm/llm_ops_arm_nd_algebra.c \
	             arm/llm_ops_arm_nd_pointwise.c \
	             arm/llm_ops_arm_nd_reduce.c \
	             arm/llm_ops_arm_nd_utility.c \
	             arm/llm_ops_arm_nd_sequence.c \
	             arm/llm_ops_arm_nd_model.c \
	             arm/llm_ops_arm_int8.c \
	             arm/llm_ops_arm_matrix.c \
	             arm/llm_ops_arm_transformer.c \
	             arm/llm_ops_arm_runtime.c \
	             arm/llm_ops_arm_lookup.c

# ── Custom LLM page cache library ────────────────────────────────────────
PCACHE_SO := llm_pcache.so
PCACHE_SRC:= llm_pcache.c

# ── Parameter manager library ─────────────────────────────────────────────
PARAM_MGR_SO := llm_param_mgr.so
PARAM_MGR_SRC:= llm_param_mgr.c

# ── Tokenizer library ────────────────────────────────────────────────────
TOKENIZER_SO  := llm_tokenizer.so
TOKENIZER_SRC := llm_tokenizer.c

.PHONY: all arm test test-arm test-tokenizer clean print_llm_have_avx2 print_llm_have_arm_neon

all: $(SONAME) $(PCACHE_SO) $(PARAM_MGR_SO) $(TOKENIZER_SO)

arm: $(ARM_SONAME)

$(SONAME): $(SRCS) x86/llm_ops.h
	$(CC) $(X86_CFLAGS) $(SQLITE_INC) \
	    -shared -o $@ $(SRCS) \
	    $(SQLITE_LIB) \
	    -lm -lpthread
	@echo "Built $(SONAME)"
	@echo "Dynamic dependencies:"
	@ldd $(SONAME) 2>/dev/null || true

$(ARM_SONAME): $(ARM_SRCS) arm/llm_ops.h
	$(CC) $(ARM_CFLAGS) $(SQLITE_INC) \
	    -shared -o $@ $(ARM_SRCS) \
	    $(SQLITE_LIB) \
	    -lm -lpthread
	@echo "Built $(ARM_SONAME)"
	@echo "Dynamic dependencies:"
	@ldd $(ARM_SONAME) 2>/dev/null || true

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
	$(CC) -O2 -std=c11 -Wall -Wextra -Wpedantic -fPIC -fvisibility=hidden \
	    $(SQLITE_INC) \
	    -shared -o $@ $(TOKENIZER_SRC) \
	    $(SQLITE_LIB)
	@echo "Built $(TOKENIZER_SO)"
	@echo "Dynamic dependencies:"
	@ldd $(TOKENIZER_SO) 2>/dev/null || true

# ── Tokenizer test binary ────────────────────────────────────────────────
TEST_TOKENIZER_BIN := test_llm_tokenizer

$(TEST_TOKENIZER_BIN): test_llm_tokenizer.c $(TOKENIZER_SO)
	$(CC) -O2 -std=c11 -Wall -Wextra -Wpedantic -o $@ test_llm_tokenizer.c \
	    $(SQLITE_INC) \
	    $(SQLITE_LIB)

test-tokenizer: $(TEST_TOKENIZER_BIN)
	./$(TEST_TOKENIZER_BIN)

# ── Test binary ───────────────────────────────────────────────────────────
TEST_BIN  := test_llm_ops_x86
TEST_ARM_BIN := test_llm_ops_arm

$(TEST_BIN): x86/test_llm_ops_x86.c $(SONAME)
	$(CC) -O2 -Ix86 -o $@ x86/test_llm_ops_x86.c \
	    $(SQLITE_INC) \
	    $(SQLITE_LIB) \
	    -lm

$(TEST_ARM_BIN): arm/test_llm_ops_arm.c $(ARM_SONAME)
	$(CC) -O2 -Iarm -o $@ arm/test_llm_ops_arm.c \
	    $(SQLITE_INC) \
	    $(SQLITE_LIB) \
	    -lm

test: $(TEST_BIN)
	./$(TEST_BIN)

test-arm: $(TEST_ARM_BIN)
	./$(TEST_ARM_BIN)

# ── Cleanup ───────────────────────────────────────────────────────────────
clean:
	rm -f $(SONAME) $(ARM_SONAME) $(PCACHE_SO) $(PARAM_MGR_SO) $(TOKENIZER_SO) $(TEST_BIN) $(TEST_ARM_BIN) $(TEST_TOKENIZER_BIN)

# Print value of LLM_HAVE_AVX2 as seen by the preprocessor
print_llm_have_avx2:
	@echo -n "#sym:LLM_HAVE_AVX2="
	@val=`$(CC) $(X86_CFLAGS) -E -dM x86/llm_ops.h | awk '/^#define LLM_HAVE_AVX2 /{print $$3; exit}'`; \
	if [ -z "$$val" ]; then echo "(not defined)"; else echo "$$val"; fi

print_llm_have_arm_neon:
	@echo -n "#sym:LLM_HAVE_NEON="
	@val=`$(CC) $(ARM_CFLAGS) -E -dM arm/llm_ops.h | awk '/^#define LLM_HAVE_NEON /{print $$3; exit}'`; \
	if [ -z "$$val" ]; then echo "(not defined)"; else echo "$$val"; fi
