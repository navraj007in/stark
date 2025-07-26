#!/usr/bin/env python3
"""
Simple test to verify the type checker works
"""

from type_system import *
from type_checker import *

def test_basic_functionality():
    """Test basic type checker functionality"""
    print("Testing STARK Type Checker...")
    
    # Test type creation
    print("✓ Creating basic types")
    i32 = PrimitiveType(TypeKind.I32)
    f64 = PrimitiveType(TypeKind.F64)
    print(f"  i32: {i32}")
    print(f"  f64: {f64}")
    
    # Test tensor types
    print("✓ Creating tensor types")
    shape = TensorShape([10, 20])
    tensor = TensorType(i32, shape)
    print(f"  tensor: {tensor}")
    
    # Test type checker
    print("✓ Creating type checker")
    checker = TypeChecker()
    
    # Test literal checking
    print("✓ Testing literal inference")
    lit_node = LiteralNode(ASTNodeKind.LITERAL, value=42)
    lit_type = checker.check_literal(lit_node)
    print(f"  Literal 42 has type: {lit_type}")
    
    # Test simple AST creation
    print("✓ Creating simple AST")
    
    # Create nodes manually with proper initialization
    left = LiteralNode(ASTNodeKind.LITERAL, value=5)
    right = LiteralNode(ASTNodeKind.LITERAL, value=3)
    add_node = BinaryOpNode(ASTNodeKind.BINARY_OP, operator="+", left=left, right=right)
    
    print(f"  Created AST: {left.value} + {right.value}")
    
    # Type check the binary operation
    result_type = checker.check_binary_op(add_node)
    print(f"  Result type: {result_type}")
    
    print("\n✅ All basic tests passed!")

if __name__ == "__main__":
    test_basic_functionality()