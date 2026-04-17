import json
import re

def clean_json_string(s: str) -> str:
    # Aggressive backslash doubling
    # Double every \ that is NOT followed by \ or "
    return re.sub(r'\\(?![\\"])', r'\\\\', s)

def test():
    test_cases = [
        (r'{"math": "\lambda"}', r'{"math": "\\lambda"}'),
        (r'{"math": "\rho"}', r'{"math": "\\rho"}'),
        (r'{"quote": "He said \"hello\""}', r'{"quote": "He said \"hello\""}'),
        (r'{"path": "C:\\Windows"}', r'{"path": "C:\\Windows"}'),
        (r'{"broken": "\1"}', r'{"broken": "\\1"}'),
    ]
    
    for inp, exp in test_cases:
        out = clean_json_string(inp)
        print(f"Input:    {inp}")
        print(f"Expected: {exp}")
        print(f"Output:   {out}")
        try:
            json.loads(out)
            print("Status:   SUCCESS")
        except Exception as e:
            print(f"Status:   FAILED ({e})")
        print("-" * 20)

if __name__ == "__main__":
    test()
