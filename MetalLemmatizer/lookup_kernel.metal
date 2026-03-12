#include <metal_stdlib>
using namespace metal;



struct GpuState {
    uint transition_start_idx;
    uint num_transitions;
    int lemma_offset;
};

struct GpuTransition {
    uchar c;
    uint next_state;
};

kernel void lookup_kernel_packed(
    const device char*           input_words   [[ buffer(0) ]],
    const device GpuState*       states        [[ buffer(1) ]],
    const device GpuTransition*  transitions   [[ buffer(2) ]],
    const device char*           lemma_buffer  [[ buffer(3) ]],
    device char*                 output_lemmas [[ buffer(4) ]],
    const device uint*           offsets       [[ buffer(5) ]],  // N+1 entries
    uint gid [[ thread_position_in_grid ]]
) {
    uint start    = offsets[gid];
    uint slot_len = offsets[gid + 1] - start;  // bytes incl. null terminator

    int state = 0;

    for (uint i = 0; i < slot_len - 1; ++i) {
        char ch = input_words[start + i];

        GpuState s = states[state];
        bool matched = false;

        for (uint k = 0; k < s.num_transitions; ++k) {
            GpuTransition t = transitions[s.transition_start_idx + k];
            if (t.c == static_cast<uchar>(ch)) {
                state = t.next_state;
                matched = true;
                break;
            }
        }

        if (!matched) {
            for (uint k = 0; k < slot_len; ++k)
                output_lemmas[start + k] = input_words[start + k];
            return;
        }
    }

    GpuState final_state = states[state];
    if (final_state.lemma_offset >= 0) {
        for (uint i = 0; i < slot_len - 1; ++i) {
            char c = lemma_buffer[final_state.lemma_offset + i];
            output_lemmas[start + i] = c;
            if (c == '\0') return;
        }
        output_lemmas[start + slot_len - 1] = '\0';  // truncated but null-terminated
    } else {
        for (uint k = 0; k < slot_len; ++k)
            output_lemmas[start + k] = input_words[start + k];
    }
}

kernel void lookup_kernel_index(
    const device char*           input_words  [[ buffer(0) ]],
    const device GpuState*       states       [[ buffer(1) ]],
    const device GpuTransition*  transitions  [[ buffer(2) ]],
    device int*                  out_indices  [[ buffer(3) ]],  // lemma_offset or -1
    const device uint*           offsets      [[ buffer(4) ]],  // N+1 entries
    uint gid [[ thread_position_in_grid ]]
) {
    uint start    = offsets[gid];
    uint slot_len = offsets[gid + 1] - start;  // bytes incl. null terminator

    int state = 0;

    for (uint i = 0; i < slot_len - 1; ++i) {
        char ch = input_words[start + i];

        GpuState s = states[state];
        bool matched = false;

        for (uint k = 0; k < s.num_transitions; ++k) {
            GpuTransition t = transitions[s.transition_start_idx + k];
            if (t.c == static_cast<uchar>(ch)) {
                state = t.next_state;
                matched = true;
                break;
            }
        }

        if (!matched) {
            out_indices[gid] = -1;
            return;
        }
    }

    GpuState final_state = states[state];
    out_indices[gid] = final_state.lemma_offset;  // >= 0 on match, -1 if non-terminal
}

kernel void lookup_kernel(
    const device char*         input_words       [[ buffer(0) ]],
    const device GpuState*     states            [[ buffer(1) ]],
    const device GpuTransition* transitions      [[ buffer(2) ]],
    const device char*         lemma_buffer      [[ buffer(3) ]],
    device char*               output_lemmas     [[ buffer(4) ]],
    const device uint&         max_word_len      [[ buffer(5) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    uint input_offset = gid * max_word_len;
    uint output_offset = gid * max_word_len;

    int state = 0;

    for (uint i = 0; i < max_word_len; ++i) {
        char ch = input_words[input_offset + i];
        if (ch == '\0') break;

        GpuState s = states[state];
        bool matched = false;

        for (uint k = 0; k < s.num_transitions; ++k) {
            GpuTransition t = transitions[s.transition_start_idx + k];
            if (t.c == static_cast<uchar>(ch)) {
                state = t.next_state;
                matched = true;
                break;
            }
        }

        if (!matched) {
            for (uint k = 0; k < max_word_len; ++k) {
                char c = input_words[input_offset + k];
                output_lemmas[output_offset + k] = c;
                if (c == '\0') break;
            }
            return;
        }
    }

    GpuState final_state = states[state];
    if (final_state.lemma_offset >= 0) {
        for (uint i = 0; i < max_word_len; ++i) {
            char c = lemma_buffer[final_state.lemma_offset + i];
            output_lemmas[output_offset + i] = c;
            if (c == '\0') break;
        }
    } else {
        for (uint k = 0; k < max_word_len; ++k) {
            char c = input_words[input_offset + k];
            output_lemmas[output_offset + k] = c;
            if (c == '\0') break;
        }
    }
}
