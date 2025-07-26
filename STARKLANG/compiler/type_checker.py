"""
STARK Type Checker

This module implements the main type checking logic for STARK programs.
It walks the AST and performs type checking with inference.
"""

from typing import List, Dict, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum, auto

from type_system import (
    Type, TypeEnvironment, TypeInferenceEngine,
    PrimitiveType, ArrayType, ListType, TupleType, FunctionType,
    TensorType, TensorShape, ModelType, DatasetType,
    TypeVariable, StructType, EnumType, TraitType,
    TypeKind, BUILTIN_TYPES
)


class ASTNodeKind(Enum):
    """AST node types"""
    PROGRAM = auto()
    FUNCTION_DECL = auto()
    STRUCT_DECL = auto()
    ENUM_DECL = auto()
    TRAIT_DECL = auto()
    IMPL_DECL = auto()
    ACTOR_DECL = auto()
    MODEL_DECL = auto()
    PIPELINE_DECL = auto()
    
    # Statements
    LET_STMT = auto()
    ASSIGN_STMT = auto()
    IF_STMT = auto()
    WHILE_STMT = auto()
    FOR_STMT = auto()
    MATCH_STMT = auto()
    RETURN_STMT = auto()
    BREAK_STMT = auto()
    CONTINUE_STMT = auto()
    EXPR_STMT = auto()
    BLOCK = auto()
    
    # Expressions
    LITERAL = auto()
    IDENTIFIER = auto()
    BINARY_OP = auto()
    UNARY_OP = auto()
    CALL = auto()
    INDEX = auto()
    FIELD_ACCESS = auto()
    ARRAY_LITERAL = auto()
    TUPLE_LITERAL = auto()
    STRUCT_LITERAL = auto()
    TENSOR_LITERAL = auto()
    LAMBDA = auto()
    IF_EXPR = auto()
    MATCH_EXPR = auto()
    AWAIT_EXPR = auto()
    SPAWN_EXPR = auto()
    SEND_EXPR = auto()
    
    # Patterns
    LITERAL_PATTERN = auto()
    IDENTIFIER_PATTERN = auto()
    WILDCARD_PATTERN = auto()
    TUPLE_PATTERN = auto()
    STRUCT_PATTERN = auto()
    ENUM_PATTERN = auto()


@dataclass
class ASTNode:
    """Base AST node"""
    kind: ASTNodeKind
    type: Optional[Type] = None  # Inferred type
    children: List['ASTNode'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


@dataclass
class LiteralNode(ASTNode):
    """Literal value node"""
    value: Any = None
    
    def __post_init__(self):
        super().__post_init__()
        self.kind = ASTNodeKind.LITERAL


@dataclass
class IdentifierNode(ASTNode):
    """Identifier node"""
    name: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        self.kind = ASTNodeKind.IDENTIFIER


@dataclass
class BinaryOpNode(ASTNode):
    """Binary operation node"""
    operator: str = ""
    left: ASTNode = None
    right: ASTNode = None
    
    def __post_init__(self):
        super().__post_init__()
        self.kind = ASTNodeKind.BINARY_OP
        if self.left and self.right:
            self.children = [self.left, self.right]


@dataclass
class UnaryOpNode(ASTNode):
    """Unary operation node"""
    operator: str = ""
    operand: ASTNode = None
    
    def __post_init__(self):
        super().__post_init__()
        self.kind = ASTNodeKind.UNARY_OP
        if self.operand:
            self.children = [self.operand]


@dataclass
class CallNode(ASTNode):
    """Function call node"""
    function: ASTNode = None
    arguments: List[ASTNode] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.kind = ASTNodeKind.CALL
        if self.arguments is None:
            self.arguments = []
        if self.function:
            self.children = [self.function] + self.arguments


@dataclass
class LetStmtNode(ASTNode):
    """Let statement node"""
    name: str = ""
    type_annotation: Optional[Type] = None
    value: ASTNode = None
    is_immutable: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        self.kind = ASTNodeKind.LET_STMT
        if self.value:
            self.children = [self.value]


@dataclass
class FunctionDeclNode(ASTNode):
    """Function declaration node"""
    name: str = ""
    parameters: List[Tuple[str, Type]] = None
    return_type: Optional[Type] = None
    body: ASTNode = None
    is_async: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        self.kind = ASTNodeKind.FUNCTION_DECL
        if self.body:
            self.children = [self.body]
        if self.parameters is None:
            self.parameters = []


@dataclass
class TensorLiteralNode(ASTNode):
    """Tensor literal node"""
    elements: List[ASTNode] = None
    shape: Optional[TensorShape] = None
    device: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.kind = ASTNodeKind.TENSOR_LITERAL
        if self.elements:
            self.children = self.elements
        else:
            self.elements = []


class TypeChecker:
    """Main type checker for STARK"""
    
    def __init__(self):
        self.env = TypeEnvironment()
        self.inference_engine = TypeInferenceEngine()
        self.errors: List[str] = []
        
        # Initialize built-in types
        for name, type in BUILTIN_TYPES.items():
            self.env.bind(name, type)
        
        # Initialize built-in functions
        self._init_builtin_functions()
    
    def _init_builtin_functions(self):
        """Initialize built-in function types"""
        # print function
        print_type = FunctionType([BUILTIN_TYPES["str"]], BUILTIN_TYPES["void"])
        self.env.bind("print", print_type)
        
        # tensor operations
        tensor_f32 = TypeVariable("T", [BUILTIN_TYPES["f32"]])
        shape_var = TypeVariable("S")
        
        # tensor.rand(shape) -> Tensor<f32, shape>
        rand_type = FunctionType(
            [ArrayType(BUILTIN_TYPES["i32"])],
            TensorType(BUILTIN_TYPES["f32"], TensorShape([None]))
        )
        self.env.bind("tensor.rand", rand_type)
        
        # tensor.zeros(shape) -> Tensor<f32, shape>
        zeros_type = FunctionType(
            [ArrayType(BUILTIN_TYPES["i32"])],
            TensorType(BUILTIN_TYPES["f32"], TensorShape([None]))
        )
        self.env.bind("tensor.zeros", zeros_type)
    
    def error(self, message: str) -> None:
        """Report a type error"""
        self.errors.append(message)
    
    def check_program(self, program: ASTNode) -> bool:
        """Type check an entire program"""
        self.errors.clear()
        self.check_node(program)
        return len(self.errors) == 0
    
    def check_node(self, node: ASTNode) -> Type:
        """Type check a single AST node"""
        if node.kind == ASTNodeKind.LITERAL:
            return self.check_literal(node)
        elif node.kind == ASTNodeKind.IDENTIFIER:
            return self.check_identifier(node)
        elif node.kind == ASTNodeKind.BINARY_OP:
            return self.check_binary_op(node)
        elif node.kind == ASTNodeKind.UNARY_OP:
            return self.check_unary_op(node)
        elif node.kind == ASTNodeKind.CALL:
            return self.check_call(node)
        elif node.kind == ASTNodeKind.LET_STMT:
            return self.check_let_stmt(node)
        elif node.kind == ASTNodeKind.FUNCTION_DECL:
            return self.check_function_decl(node)
        elif node.kind == ASTNodeKind.TENSOR_LITERAL:
            return self.check_tensor_literal(node)
        elif node.kind == ASTNodeKind.BLOCK:
            return self.check_block(node)
        else:
            self.error(f"Unknown node kind: {node.kind}")
            return BUILTIN_TYPES["void"]
    
    def check_literal(self, node: LiteralNode) -> Type:
        """Type check a literal value"""
        value = node.value
        
        if isinstance(value, bool):
            node.type = BUILTIN_TYPES["bool"]
        elif isinstance(value, int):
            # Default to i32 for integer literals
            node.type = BUILTIN_TYPES["i32"]
        elif isinstance(value, float):
            # Default to f64 for float literals
            node.type = BUILTIN_TYPES["f64"]
        elif isinstance(value, str):
            node.type = BUILTIN_TYPES["str"]
        elif value is None:
            # null literal - will need context to determine type
            node.type = TypeVariable("null")
        else:
            self.error(f"Unknown literal type: {type(value)}")
            node.type = BUILTIN_TYPES["void"]
        
        return node.type
    
    def check_identifier(self, node: IdentifierNode) -> Type:
        """Type check an identifier"""
        type = self.env.lookup(node.name)
        if type is None:
            self.error(f"Undefined variable: {node.name}")
            node.type = TypeVariable(f"undefined_{node.name}")
        else:
            node.type = type
        
        return node.type
    
    def check_binary_op(self, node: BinaryOpNode) -> Type:
        """Type check a binary operation"""
        left_type = self.check_node(node.left)
        right_type = self.check_node(node.right)
        
        # Tensor operations
        if (isinstance(left_type, TensorType) and isinstance(right_type, TensorType)):
            result_type = self.inference_engine.infer_tensor_op_type(
                node.operator, left_type, right_type
            )
            if result_type:
                node.type = result_type
                return result_type
            else:
                self.error(f"Incompatible tensor shapes for operation {node.operator}")
                node.type = left_type
                return node.type
        
        # Arithmetic operations
        if node.operator in ["+", "-", "*", "/", "%", "**"]:
            if self.is_numeric_type(left_type) and self.is_numeric_type(right_type):
                # Promote to the "larger" type
                result_type = self.promote_numeric_types(left_type, right_type)
                node.type = result_type
                return result_type
            else:
                self.error(f"Arithmetic operation {node.operator} requires numeric types")
        
        # Comparison operations
        elif node.operator in ["==", "!=", "<", "<=", ">", ">="]:
            if self.can_compare(left_type, right_type):
                node.type = BUILTIN_TYPES["bool"]
                return node.type
            else:
                self.error(f"Cannot compare types {left_type} and {right_type}")
        
        # Logical operations
        elif node.operator in ["and", "or"]:
            if (left_type.equals(BUILTIN_TYPES["bool"]) and 
                right_type.equals(BUILTIN_TYPES["bool"])):
                node.type = BUILTIN_TYPES["bool"]
                return node.type
            else:
                self.error(f"Logical operation {node.operator} requires bool types")
        
        # Default to left type if unification fails
        node.type = left_type
        return node.type
    
    def check_unary_op(self, node: UnaryOpNode) -> Type:
        """Type check a unary operation"""
        operand_type = self.check_node(node.operand)
        
        if node.operator == "-":
            if self.is_numeric_type(operand_type):
                node.type = operand_type
            else:
                self.error(f"Unary minus requires numeric type, got {operand_type}")
                node.type = operand_type
        elif node.operator == "not":
            if operand_type.equals(BUILTIN_TYPES["bool"]):
                node.type = BUILTIN_TYPES["bool"]
            else:
                self.error(f"Logical not requires bool type, got {operand_type}")
                node.type = BUILTIN_TYPES["bool"]
        else:
            self.error(f"Unknown unary operator: {node.operator}")
            node.type = operand_type
        
        return node.type
    
    def check_call(self, node: CallNode) -> Type:
        """Type check a function call"""
        func_type = self.check_node(node.function)
        arg_types = [self.check_node(arg) for arg in node.arguments]
        
        if isinstance(func_type, FunctionType):
            # Check parameter count
            if len(arg_types) != len(func_type.param_types):
                self.error(f"Function expects {len(func_type.param_types)} arguments, got {len(arg_types)}")
                node.type = func_type.return_type
                return node.type
            
            # Check parameter types
            for i, (arg_type, param_type) in enumerate(zip(arg_types, func_type.param_types)):
                if not self.inference_engine.unify(arg_type, param_type):
                    self.error(f"Argument {i+1} type mismatch: expected {param_type}, got {arg_type}")
            
            node.type = func_type.return_type
            return node.type
        else:
            self.error(f"Cannot call non-function type: {func_type}")
            node.type = TypeVariable("call_result")
            return node.type
    
    def check_let_stmt(self, node: LetStmtNode) -> Type:
        """Type check a let statement"""
        value_type = self.check_node(node.value)
        
        if node.type_annotation:
            # Check that value type matches annotation
            if not self.inference_engine.unify(value_type, node.type_annotation):
                self.error(f"Type mismatch in let statement: expected {node.type_annotation}, got {value_type}")
            node.type = node.type_annotation
        else:
            # Infer type from value
            node.type = value_type
        
        # Bind variable in environment
        self.env.bind(node.name, node.type)
        
        return BUILTIN_TYPES["void"]
    
    def check_function_decl(self, node: FunctionDeclNode) -> Type:
        """Type check a function declaration"""
        # Create function type
        param_types = [param[1] for param in node.parameters]
        return_type = node.return_type or BUILTIN_TYPES["void"]
        func_type = FunctionType(param_types, return_type, node.is_async)
        
        # Bind function name
        self.env.bind(node.name, func_type)
        
        # Check function body in new environment
        body_env = self.env.extend()
        old_env = self.env
        self.env = body_env
        
        # Bind parameters
        for param_name, param_type in node.parameters:
            self.env.bind(param_name, param_type)
        
        # Check body
        body_type = self.check_node(node.body)
        
        # Check return type compatibility
        if not self.inference_engine.unify(body_type, return_type):
            self.error(f"Function body type {body_type} doesn't match declared return type {return_type}")
        
        # Restore environment
        self.env = old_env
        
        node.type = func_type
        return func_type
    
    def check_tensor_literal(self, node: TensorLiteralNode) -> Type:
        """Type check a tensor literal"""
        if not node.elements:
            self.error("Empty tensor literal")
            node.type = TensorType(BUILTIN_TYPES["f32"], TensorShape([0]))
            return node.type
        
        # Check all elements have the same type
        element_types = [self.check_node(elem) for elem in node.elements]
        first_type = element_types[0]
        
        for i, elem_type in enumerate(element_types[1:], 1):
            if not self.inference_engine.unify(elem_type, first_type):
                self.error(f"Tensor element {i} type mismatch: expected {first_type}, got {elem_type}")
        
        # Infer shape
        if node.shape:
            shape = node.shape
        else:
            # Simple 1D tensor for now
            shape = TensorShape([len(node.elements)])
        
        node.type = TensorType(first_type, shape, node.device)
        return node.type
    
    def check_block(self, node: ASTNode) -> Type:
        """Type check a block of statements"""
        block_env = self.env.extend()
        old_env = self.env
        self.env = block_env
        
        last_type = BUILTIN_TYPES["void"]
        for child in node.children:
            last_type = self.check_node(child)
        
        self.env = old_env
        node.type = last_type
        return last_type
    
    def is_numeric_type(self, type: Type) -> bool:
        """Check if a type is numeric"""
        if isinstance(type, PrimitiveType):
            return type.kind in [
                TypeKind.I8, TypeKind.I16, TypeKind.I32, TypeKind.I64, TypeKind.I128,
                TypeKind.U8, TypeKind.U16, TypeKind.U32, TypeKind.U64, TypeKind.U128,
                TypeKind.F32, TypeKind.F64
            ]
        return False
    
    def can_compare(self, t1: Type, t2: Type) -> bool:
        """Check if two types can be compared"""
        return (self.is_numeric_type(t1) and self.is_numeric_type(t2)) or t1.equals(t2)
    
    def promote_numeric_types(self, t1: Type, t2: Type) -> Type:
        """Promote numeric types to a common type"""
        if not (isinstance(t1, PrimitiveType) and isinstance(t2, PrimitiveType)):
            return t1
        
        # Simple promotion rules - float beats int, larger beats smaller
        type_priority = {
            TypeKind.I8: 1, TypeKind.U8: 1,
            TypeKind.I16: 2, TypeKind.U16: 2,
            TypeKind.I32: 3, TypeKind.U32: 3,
            TypeKind.I64: 4, TypeKind.U64: 4,
            TypeKind.I128: 5, TypeKind.U128: 5,
            TypeKind.F32: 10,
            TypeKind.F64: 11
        }
        
        p1 = type_priority.get(t1.kind, 0)
        p2 = type_priority.get(t2.kind, 0)
        
        return t1 if p1 >= p2 else t2


def create_sample_ast() -> ASTNode:
    """Create a sample AST for testing"""
    # let a = tensor.rand([1000, 1000])
    # let b = tensor.rand([1000, 1000])  
    # let c = a @ b
    
    program = ASTNode(ASTNodeKind.PROGRAM)
    
    # tensor.rand([1000, 1000])
    rand_call1 = CallNode(
        function=IdentifierNode("tensor.rand"),
        arguments=[
            TensorLiteralNode([
                LiteralNode(1000),
                LiteralNode(1000)
            ])
        ]
    )
    
    # let a = tensor.rand([1000, 1000])
    let_a = LetStmtNode("a", None, rand_call1)
    
    # tensor.rand([1000, 1000])
    rand_call2 = CallNode(
        function=IdentifierNode("tensor.rand"),
        arguments=[
            TensorLiteralNode([
                LiteralNode(1000),
                LiteralNode(1000)
            ])
        ]
    )
    
    # let b = tensor.rand([1000, 1000])
    let_b = LetStmtNode("b", None, rand_call2)
    
    # a @ b
    matmul = BinaryOpNode("@", IdentifierNode("a"), IdentifierNode("b"))
    
    # let c = a @ b
    let_c = LetStmtNode("c", None, matmul)
    
    program.children = [let_a, let_b, let_c]
    return program


if __name__ == "__main__":
    # Test the type checker
    checker = TypeChecker()
    
    # Create and check sample AST
    ast = create_sample_ast()
    success = checker.check_program(ast)
    
    if success:
        print("✅ Type checking passed!")
        print("\nInferred types:")
        for node in ast.children:
            if hasattr(node, 'name'):
                print(f"  {node.name}: {node.type}")
    else:
        print("❌ Type checking failed!")
        for error in checker.errors:
            print(f"  Error: {error}")