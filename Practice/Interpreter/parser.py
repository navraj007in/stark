def parse_stark_code(code: str):
    lines = [line.strip() for line in code.split("\n") if line.strip()]
    ast = []
    for line in lines:
        if line.startswith("struct"):
            ast.append({"type": "struct_def", "definition": line})
        elif line.startswith("fn"):
            ast.append({"type": "fn_def", "definition": line})
        elif line.startswith("let "):
            ast.append({"type": "var_decl", "line": line})
        elif line.startswith("print("):
            ast.append({"type": "print", "line": line})
        elif line.startswith("if "):
            ast.append({"type": "if_stmt", "line": line})
        elif "return" in line:
            ast.append({"type": "stmt", "line": line})
        else:
            ast.append({"type": "stmt", "line": line})
    return ast
