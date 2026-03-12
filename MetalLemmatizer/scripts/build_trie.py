import os
import struct
from collections import namedtuple

MAX_WORD_LEN = 37

State = namedtuple("State", ["transition_start_idx", "num_transitions", "lemma_offset"])
Transition = namedtuple("Transition", ["c", "next_state"])

def build_trie(words):
    root = {}
    next_state_id = 0
    states = []
    transitions = []
    lemma_buffer = bytearray()
    
    # Add a dummy state 0 for the root
    states.append(State(0, 0, 0))

    for word, lemma in words:
        current_node = root
        for i, char_code in enumerate(word):
            if char_code not in current_node:
                current_node[char_code] = {}
            current_node = current_node[char_code]
        current_node['#'] = lemma # Mark end of word and store lemma

    # Flatten the trie into states and transitions
    queue = [(root, 0)] # (node, state_id)
    
    while queue:
        node, state_id = queue.pop(0)
        
        node_transitions = []
        for char_code in sorted(node.keys()):
            if char_code == '#':
                # This is a lemma, store its offset
                lemma_offset = len(lemma_buffer)
                lemma_bytes = node['#'].encode('utf-8')
                lemma_buffer.extend(lemma_bytes)
                lemma_buffer.append(0) # Null terminator
                states[state_id] = states[state_id]._replace(lemma_offset=lemma_offset)
            else:
                next_node = node[char_code]
                next_state_id += 1
                states.append(State(0, 0, 0)) # Placeholder for new state
                node_transitions.append(Transition(char_code, next_state_id))
                queue.append((next_node, next_state_id))
        
        # Update the current state with its transitions
        start_idx = len(transitions)
        num_trans = len(node_transitions)
        states[state_id] = states[state_id]._replace(transition_start_idx=start_idx, num_transitions=num_trans)
        transitions.extend(node_transitions)

    return states, transitions, lemma_buffer

if __name__ == "__main__":
    # Example usage:
    words = [
        ("cat", "cat"),
        ("cats", "cat"),
        ("dog", "dog"),
        ("dogs", "dog"),
        ("apple", "apple"),
        ("apples", "apple"),
        ("apply", "apply"),
    ]

    states, transitions, lemma_buffer = build_trie(words)

    output_dir = "../resources"
    os.makedirs(output_dir, exist_ok=True)

    states_bytes = b"".join(struct.pack("IIi", s.transition_start_idx, s.num_transitions, s.lemma_offset) for s in states)
    transitions_bytes = b"".join(struct.pack("B I", t.c, t.next_state) for t in transitions)

    header = struct.pack("III", len(states_bytes), len(transitions_bytes), len(lemma_buffer))

    with open(os.path.join(output_dir, "trie.bin"), "wb") as f:
        f.write(header)
        f.write(states_bytes)
        f.write(transitions_bytes)
        f.write(lemma_buffer)

    print(f"Serialized {len(states)} states, {len(transitions)} transitions, {len(lemma_buffer)} bytes of lemma_buffer into trie.bin")
