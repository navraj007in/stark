from parser import parse_stark_code
from interpreter import simulate_execution
import sys

def run_stark_file(file_path):
    with open(file_path, "r") as f:
        code = f.read()
    ast = parse_stark_code(code)
    simulate_execution(ast)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python starkvm.py <program.st>")
        sys.exit(1)

    run_stark_file(sys.argv[1])
