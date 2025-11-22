from typing import List


class GeodesicPath:
    def __init__(
        self, start_string: str, end_string: str, operations: List[str], distance: int
    ):
        self.start_string = start_string
        self.end_string = end_string
        self.operations = operations
        self.distance = distance

    def get_operations(self) -> List[str]:
        """Returns the list of operations."""
        return self.operations

    def get_path_strings(self) -> List[str]:
        """Reconstructs and returns the list of intermediate strings."""
        current_str = self.start_string
        path_strings = [current_str]

        # We need to apply operations to transform start_string to end_string.
        # The operations list is expected to be in the order of application.
        # However, my C++ implementation returns operations in reverse order of application (from end to start).
        # Wait, let's check cped.cpp and levd.cpp again.
        # In levd.cpp:
        # path += ...
        # std::reverse(path.begin(), path.end());
        # So levd returns operations in order from start to end.

        # In cped.cpp:
        # path_ops.push_back(...)
        # for (auto it = path_ops.rbegin(); ... )
        # So cped also returns operations in order from start to end.

        # But wait, the operations are derived from backtracking from (n, m) to (0, 0).
        # So the first operation found is the LAST operation applied.
        # If I reverse them, I get the FIRST operation applied.
        # Let's verify this logic.

        # Example: kitten -> sitting
        # (0,0) -> ... -> (n,m)
        # Backtracking starts at (n,m).
        # If last op was Match 'g', we are at (n,m), go to (n-1, m-1).
        # So we find 'Match' first (which is the last op).
        # Then we reverse.
        # So the list is [First Op, Second Op, ..., Last Op].

        # Now, how to apply them?
        # If the first op is 'Substitute k->s', it applies to the first character?
        # No, Levenshtein operations are usually defined relative to the current string state.
        # But standard Levenshtein DP operations are defined on indices of the original strings.
        # Applying them sequentially to transform string A to B requires care with indices if we modify the string in place.

        # Actually, it's easier to reconstruct the string at step k from the indices.
        # But the operations returned are just "M", "S", "I", "D". They don't have indices.
        # So we need to track the current index in start_string and build the next string.

        # Let's trace:
        # s1 = "abc", s2 = "ac"
        # Path: Match 'a', Delete 'b', Match 'c'.
        # Op list: ["M", "D", "M"]

        # Step 0: "abc"
        # Op 1: "M". We consume 'a' from s1 and 'a' from s2. Output so far: "a"
        # Op 2: "D". We consume 'b' from s1. Output so far: "a"
        # Op 3: "M". We consume 'c' from s1 and 'c' from s2. Output so far: "ac"

        # Wait, `get_path_strings` should return "abc", "ac", "ac"?
        # Or "abc" -> "ac" (after deleting b) -> "ac" (match c)?
        # Usually "Match" doesn't change the string.
        # "Delete" removes a char.
        # "Insert" adds a char.
        # "Substitute" changes a char.

        # If we want to show the *process*, we should show the string after each edit.
        # But "Match" is not an edit.
        # If the user wants "Geodesic", they usually mean the sequence of strings in the metric space.
        # In Levenshtein, a Match is a 0-cost edge. It doesn't change the state in the graph of strings?
        # Actually, in the graph of strings, "abc" and "ac" are neighbors if dist is 1.
        # "abc" --(delete b)--> "ac".
        # The "Match" operations are just traversing the identical parts.
        # So "abc" -> "ac" is one step.
        # But if we have "kitten" -> "sitting".
        # k->s (Sub): "sitten"
        # i->i (Match): "sitten"
        # ...
        # If we include Matches, we have many intermediate "same" strings.
        # Maybe we should only emit a new string when the string actually changes?
        # But the user asked for "intermediate strings".

        # Let's implement a reconstruction that applies edits.
        # We maintain a cursor `read_idx` on `start_string` and build `next_string`.
        # Wait, standard edit operations are applied to the *current* string.
        # But the DP path gives us operations aligned with the *original* indices.
        # It's a sequence of alignment operations.

        # Let's try to reconstruct the sequence of strings.
        # We can think of this as building the target string `end_string` from `start_string`.
        # But strictly speaking, we want the sequence of strings $S_0, S_1, \dots, S_k$ where $S_0 = start$, $S_k = end$, and $dist(S_i, S_{i+1}) = 1$ (or cost of op).

        # If we have "M", "D", "M".
        # Start: "abc"
        # 1. M: "abc" (no change)
        # 2. D: "ac" (delete b)
        # 3. M: "ac" (no change)

        # If we filter out "M", we get "abc" -> "ac".
        # This seems reasonable.

        # However, for "Substitute", it's atomic.
        # "abc" -> "abd" (Sub c->d).

        # Implementation:
        # We can't easily apply ops one by one to the whole string if we don't track indices.
        # But we have the full list of ops.
        # We can reconstruct the string at step `k` by applying the first `k` operations?
        # No, that's inefficient.

        # Better approach:
        # We can simulate the editing process.
        # We start with `current_s = start_string`.
        # We process operations.
        # But wait, the operations "M", "I", "D" from DP are "alignment" operations.
        # M: x[i] == y[j]
        # S: x[i] != y[j] -> replace x[i] with y[j]
        # I: insert y[j]
        # D: delete x[i]

        # These operations are effectively instructions to build Y from X, *reading X from left to right*.
        # But to generate intermediate strings $S_1, S_2...$, we need to apply them.
        # If we apply them strictly left-to-right, the indices of future characters in X shift.
        # But since we process X left-to-right, we don't care about indices of *processed* chars, only *unprocessed* chars.
        # And unprocessed chars of X remain at the suffix.

        # So:
        # processed_prefix = ""
        # remaining_suffix = start_string
        # For op in operations:
        #   if M:
        #     char = remaining_suffix[0]
        #     processed_prefix += char
        #     remaining_suffix = remaining_suffix[1:]
        #   if S:
        #     char = y_char (we need y_char!)
        #     processed_prefix += char
        #     remaining_suffix = remaining_suffix[1:]
        #     yield processed_prefix + remaining_suffix
        #   if I:
        #     char = y_char
        #     processed_prefix += char
        #     yield processed_prefix + remaining_suffix
        #   if D:
        #     remaining_suffix = remaining_suffix[1:]
        #     yield processed_prefix + remaining_suffix

        # Problem: We don't have `y_char` in the operations list!
        # The operations list is just ["M", "S", "I", "D"].
        # We need `end_string` to know what to insert/substitute.
        # We also need `start_string` to know what to match/delete.

        # So we need to track indices in `start_string` (i) and `end_string` (j).

        # But wait, `current_s` changes.
        # If we use `processed_prefix` (list of chars) and `start_string[remaining_suffix_idx:]`, we can reconstruct the current string.

        # Let's refine the loop.

        i = 0  # index in start_string
        j = 0  # index in end_string

        processed = []

        # We need to handle CPED ops too: "C:len", "D:len".

        for op in self.operations:
            if op == "M":
                # Copy char from start_string to processed
                processed.append(self.start_string[i])
                i += 1
                j += 1
                # No string change
            elif op == "S":
                # Substitute: take char from end_string
                processed.append(self.end_string[j])
                i += 1
                j += 1
                # String changed
                path_strings.append("".join(processed) + self.start_string[i:])
            elif op == "I":
                # Insert: take char from end_string
                processed.append(self.end_string[j])
                j += 1
                # String changed
                path_strings.append("".join(processed) + self.start_string[i:])
            elif op == "D":
                # Delete: skip char in start_string
                i += 1
                # String changed
                path_strings.append("".join(processed) + self.start_string[i:])
            elif op.startswith("D:"):
                # Block delete
                count = int(op.split(":")[1])
                i += count
                path_strings.append("".join(processed) + self.start_string[i:])
            elif op.startswith("C:"):
                # Copy
                length = int(op.split(":")[1])
                # Copy `length` chars from end_string[j:j+length]
                # But wait, CPED Copy copies from *already generated* Y.
                # The definition of Copy in CPED is copying from Y[0:j].
                # So we just take from end_string.
                segment = self.end_string[j : j + length]
                processed.extend(list(segment))
                j += length
                path_strings.append("".join(processed) + self.start_string[i:])

        return path_strings
