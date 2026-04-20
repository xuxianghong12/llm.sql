/*
 * Standalone BPE tokenizer for llm.sql C/C++ demos.
 *
 * NOTE: This is a legacy implementation.  Demos now use the
 * ``llm_tokenizer`` SQLite extension (sqlite-llm/llm_tokenizer.c)
 * which provides the same functionality as SQL functions, usable
 * from any language with SQLite bindings.
 *
 * Loads vocabulary from the SQLite model database (the ``vocab`` table
 * written by ``SqliteTokenizer``) and BPE merge rules from the
 * HuggingFace ``tokenizer.json`` file that ships with the exported
 * model.  No external tokenizer library is required.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef LLM_SQL_DEMO_BPE_TOKENIZER_H
#define LLM_SQL_DEMO_BPE_TOKENIZER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct llmsql_tokenizer llmsql_tokenizer;

/**
 * Create a tokenizer by loading the vocab table from the SQLite
 * database and BPE merges from ``tokenizer.json`` in *model_dir*.
 *
 * @param model_dir   Path to the exported model directory.
 * @param db_filename Database filename inside *model_dir* (e.g.
 *                    ``"model.db"``).  When NULL the runtime
 *                    picks the first available among model.db /
 *                    model_int8.db.
 * @return Opaque tokenizer handle, or NULL on failure.
 */
llmsql_tokenizer *llmsql_tokenizer_create(
    const char *model_dir,
    const char *db_filename
);

/**
 * Encode a UTF-8 text string into a sequence of BPE token IDs.
 *
 * @param tok       Tokenizer created with llmsql_tokenizer_create().
 * @param text      Null-terminated UTF-8 input string.
 * @param out_ids   On success, receives a malloc'd array of token IDs.
 *                  Caller must free() this.
 * @param out_count On success, receives the number of IDs.
 * @return 1 on success, 0 on failure.
 */
int llmsql_tokenizer_encode(
    const llmsql_tokenizer *tok,
    const char *text,
    int **out_ids,
    int *out_count
);

/**
 * Decode a sequence of token IDs into a UTF-8 text string.
 *
 * @param tok      Tokenizer created with llmsql_tokenizer_create().
 * @param ids      Array of token IDs.
 * @param count    Number of elements in *ids*.
 * @param out_text On success, receives a malloc'd null-terminated
 *                 UTF-8 string.  Caller must free() this.
 * @return 1 on success, 0 on failure.
 */
int llmsql_tokenizer_decode(
    const llmsql_tokenizer *tok,
    const int *ids,
    int count,
    char **out_text
);

/**
 * Free all resources held by the tokenizer.
 */
void llmsql_tokenizer_free(llmsql_tokenizer *tok);

#ifdef __cplusplus
}
#endif

#endif /* LLM_SQL_DEMO_BPE_TOKENIZER_H */
