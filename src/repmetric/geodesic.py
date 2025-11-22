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

        # Operations are in application order (C++ implementation reverses backtracked path).
        # Each operation is an alignment instruction between start_string and end_string:
        #   M: Match (no change), S: Substitute, I: Insert, D: Delete
        # We track indices i (start_string) and j (end_string) and build intermediate strings.
        # Only emit new strings when actual edits occur (S/I/D), not on Match operations.

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
