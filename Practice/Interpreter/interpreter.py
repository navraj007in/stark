class Function:
    def __init__(self, name, params, body):
        self.name = name
        self.params = params
        self.body = body

class StackFrame:
    def __init__(self, name, parent=None):
        self.name = name
        self.variables = {}
        self.parent = parent

class STARKVM:
    def __init__(self, ast):
        self.ast = ast
        self.functions = {}
        self.global_frame = StackFrame("global")
        self.current_frame = self.global_frame

    def run(self):
        self._index_functions()
        if "main" not in self.functions:
            raise Exception("main() function not found.")
        print("\n--- STARKVM Phase 2.1 Execution Trace ---\n")
        self._call_function("main", [])

    def _index_functions(self):
        current_func = None
        for node in self.ast:
            if node["type"] == "fn_def":
                header = node["definition"]
                name = header.split()[1].split("(")[0]
                params_part = header.split("(")[1].split(")")[0]
                params = [p.strip().split(":")[0] for p in params_part.split(",") if p]
                current_func = Function(name, params, [])
                self.functions[name] = current_func
            elif current_func:
                current_func.body.append(node)

    def _call_function(self, name, args):
        print(f"> Entering function: {name}()")
        func = self.functions.get(name)
        if not func:
            raise Exception(f"Function {name} not defined")

        local_frame = StackFrame(name, parent=self.current_frame)
        self.current_frame = local_frame

        for i in range(len(func.params)):
            if i < len(args):
                local_frame.variables[func.params[i]] = args[i]

        return_val = None
        for node in func.body:
            if node["type"] == "var_decl" or ("=" in node.get("line", "")):
                self._execute_statement(node["line"])
            elif node["type"] == "print":
                self._handle_print(node["line"])
            elif node["type"] == "stmt" and "(" in node["line"]:
                return_val = self._handle_function_call(node["line"])
            elif node["type"] == "stmt" and node["line"].strip().startswith("return"):
                return_val = self._evaluate_expression(node["line"].split("return", 1)[1].strip())
                break

        self.current_frame = local_frame.parent
        print(f"> Exiting function: {name}()")
        return return_val

    def _execute_statement(self, line):
        if line.startswith("let "):
            line = line.replace("let ", "")
        var, expr = [x.strip() for x in line.split("=", 1)]
        val = self._evaluate_expression(expr)
        self.current_frame.variables[var] = val
        print(f"  Assigned {var} = {val}")

    def _evaluate_expression(self, expr):
        tokens = expr.split()
        if len(tokens) == 1:
            if tokens[0].isdigit():
                return int(tokens[0])
            return self._resolve_variable(tokens[0])
        elif len(tokens) == 3:
            left = self._resolve_variable(tokens[0])
            op = tokens[1]
            right = self._resolve_variable(tokens[2])
            if op == '+':
                return left + right
            elif op == '-':
                return left - right
            elif op == '*':
                return left * right
            elif op == '/':
                return left // right
        return expr

    def _resolve_variable(self, val):
        if val.isdigit():
            return int(val)
        return self.current_frame.variables.get(val, 0)

    def _handle_print(self, line):
        content = line[6:-1].strip()
        val = self.current_frame.variables.get(content, content)
        print(f"Output: {val}")

    def _handle_function_call(self, line):
        name = line.split("(")[0].strip()
        args_part = line.split("(")[1].split(")")[0]
        args = [self._evaluate_expression(arg.strip()) for arg in args_part.split(",") if arg.strip()]
        return self._call_function(name, args)

def simulate_execution(ast):
    vm = STARKVM(ast)
    vm.run()
