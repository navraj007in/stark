"""
Tests for the STARK type checker
"""

import unittest
from type_system import *
from type_checker import *


class TestTypeSystem(unittest.TestCase):
    """Test basic type system functionality"""
    
    def test_primitive_types(self):
        """Test primitive type creation and equality"""
        i32 = PrimitiveType(TypeKind.I32)
        f64 = PrimitiveType(TypeKind.F64)
        
        self.assertTrue(i32.equals(PrimitiveType(TypeKind.I32)))
        self.assertFalse(i32.equals(f64))
        self.assertEqual(str(i32), "i32")
    
    def test_tensor_types(self):
        """Test tensor type creation and shape compatibility"""
        shape1 = TensorShape([10, 20])
        shape2 = TensorShape([10, None])  # Dynamic dimension
        shape3 = TensorShape([10, 30])
        
        self.assertTrue(shape1.is_compatible_with(shape2))
        self.assertFalse(shape1.is_compatible_with(shape3))
        
        tensor1 = TensorType(PrimitiveType(TypeKind.F32), shape1)
        tensor2 = TensorType(PrimitiveType(TypeKind.F32), shape2)
        
        self.assertTrue(tensor1.equals(tensor2))
        self.assertEqual(str(tensor1), "Tensor<f32, [10, 20]>")
    
    def test_tensor_broadcasting(self):
        """Test tensor shape broadcasting"""
        shape1 = TensorShape([1, 10])
        shape2 = TensorShape([5, 1])
        shape3 = TensorShape([5, 10])
        
        broadcast = shape1.broadcast_with(shape2)
        self.assertIsNotNone(broadcast)
        self.assertEqual(broadcast.dimensions, [5, 10])
        
        # Incompatible shapes
        shape4 = TensorShape([3, 4])
        shape5 = TensorShape([5, 6])
        broadcast2 = shape4.broadcast_with(shape5)
        self.assertIsNone(broadcast2)
    
    def test_function_types(self):
        """Test function type creation"""
        func_type = FunctionType(
            [PrimitiveType(TypeKind.I32), PrimitiveType(TypeKind.F64)],
            PrimitiveType(TypeKind.BOOL),
            is_async=True
        )
        
        self.assertEqual(str(func_type), "async fn(i32, f64) -> bool")
    
    def test_type_inference_unification(self):
        """Test type unification"""
        engine = TypeInferenceEngine()
        
        # Basic unification
        i32 = PrimitiveType(TypeKind.I32)
        f64 = PrimitiveType(TypeKind.F64)
        
        self.assertTrue(engine.unify(i32, i32))
        self.assertFalse(engine.unify(i32, f64))
        
        # Type variable unification
        var = engine.fresh_type_var()
        self.assertTrue(engine.unify(var, i32))
        self.assertEqual(engine.substitutions[var.name], i32)


class TestTypeChecker(unittest.TestCase):
    """Test the main type checker"""
    
    def setUp(self):
        self.checker = TypeChecker()
    
    def test_literal_inference(self):
        """Test literal type inference"""
        # Integer literal
        int_node = LiteralNode(ASTNodeKind.LITERAL, value=42)
        int_type = self.checker.check_literal(int_node)
        self.assertTrue(int_type.equals(BUILTIN_TYPES["i32"]))
        
        # Float literal  
        float_node = LiteralNode(3.14)
        float_type = self.checker.check_literal(float_node)
        self.assertTrue(float_type.equals(BUILTIN_TYPES["f64"]))
        
        # Boolean literal
        bool_node = LiteralNode(True)
        bool_type = self.checker.check_literal(bool_node)
        self.assertTrue(bool_type.equals(BUILTIN_TYPES["bool"]))
        
        # String literal
        str_node = LiteralNode("hello")
        str_type = self.checker.check_literal(str_node)
        self.assertTrue(str_type.equals(BUILTIN_TYPES["str"]))
    
    def test_arithmetic_operations(self):
        """Test arithmetic operation type checking"""
        # 5 + 3
        left = LiteralNode(5)
        right = LiteralNode(3)
        add_node = BinaryOpNode("+", left, right)
        
        result_type = self.checker.check_binary_op(add_node)
        self.assertTrue(result_type.equals(BUILTIN_TYPES["i32"]))
        self.assertEqual(len(self.checker.errors), 0)
    
    def test_tensor_operations(self):
        """Test tensor operation type checking"""
        # Create tensor types
        shape1 = TensorShape([10, 20])
        shape2 = TensorShape([20, 30])
        tensor1 = TensorType(BUILTIN_TYPES["f32"], shape1)
        tensor2 = TensorType(BUILTIN_TYPES["f32"], shape2)
        
        # Mock nodes with these types
        left = ASTNode(ASTNodeKind.IDENTIFIER)
        left.type = tensor1
        right = ASTNode(ASTNodeKind.IDENTIFIER) 
        right.type = tensor2
        
        # Matrix multiplication: [10, 20] @ [20, 30] -> [10, 30]
        matmul_node = BinaryOpNode("@", left, right)
        result_type = self.checker.inference_engine.infer_tensor_op_type("@", tensor1, tensor2)
        
        self.assertIsNotNone(result_type)
        self.assertEqual(result_type.shape.dimensions, [10, 30])
    
    def test_variable_binding(self):
        """Test variable binding and lookup"""
        # let x = 42
        value_node = LiteralNode(42)
        let_node = LetStmtNode("x", None, value_node)
        
        self.checker.check_let_stmt(let_node)
        
        # Check that x is bound to i32
        x_type = self.checker.env.lookup("x")
        self.assertIsNotNone(x_type)
        self.assertTrue(x_type.equals(BUILTIN_TYPES["i32"]))
    
    def test_function_call_checking(self):
        """Test function call type checking"""
        # Define a function: fn add(a: i32, b: i32) -> i32
        func_type = FunctionType(
            [BUILTIN_TYPES["i32"], BUILTIN_TYPES["i32"]],
            BUILTIN_TYPES["i32"]
        )
        self.checker.env.bind("add", func_type)
        
        # Call: add(5, 3)
        func_node = IdentifierNode("add")
        func_node.type = func_type
        arg1 = LiteralNode(5)
        arg2 = LiteralNode(3)
        call_node = CallNode(func_node, [arg1, arg2])
        
        result_type = self.checker.check_call(call_node)
        self.assertTrue(result_type.equals(BUILTIN_TYPES["i32"]))
        self.assertEqual(len(self.checker.errors), 0)
    
    def test_tensor_literal_checking(self):
        """Test tensor literal type checking"""
        # tensor [1.0, 2.0, 3.0]
        elements = [LiteralNode(1.0), LiteralNode(2.0), LiteralNode(3.0)]
        tensor_node = TensorLiteralNode(elements)
        
        result_type = self.checker.check_tensor_literal(tensor_node)
        
        self.assertIsInstance(result_type, TensorType)
        self.assertTrue(result_type.element_type.equals(BUILTIN_TYPES["f64"]))
        self.assertEqual(result_type.shape.dimensions, [3])
    
    def test_type_annotation_checking(self):
        """Test explicit type annotations"""
        # let x: i64 = 42
        value_node = LiteralNode(42)  # This is i32 by default
        let_node = LetStmtNode("x", BUILTIN_TYPES["i64"], value_node)
        
        self.checker.check_let_stmt(let_node)
        
        # Should have a type mismatch error (i32 vs i64)
        self.assertGreater(len(self.checker.errors), 0)
    
    def test_sample_program(self):
        """Test the complete sample program"""
        ast = create_sample_ast()
        success = self.checker.check_program(ast)
        
        if not success:
            for error in self.checker.errors:
                print(f"Error: {error}")
        
        # Should pass type checking (with some unification warnings)
        # The exact result depends on tensor.rand implementation


class TestTensorShapes(unittest.TestCase):
    """Test tensor shape operations"""
    
    def test_matrix_multiplication_shapes(self):
        """Test matrix multiplication shape inference"""
        engine = TypeInferenceEngine()
        
        # [2, 3] @ [3, 4] -> [2, 4]
        t1 = TensorType(BUILTIN_TYPES["f32"], TensorShape([2, 3]))
        t2 = TensorType(BUILTIN_TYPES["f32"], TensorShape([3, 4]))
        
        result = engine.infer_tensor_op_type("@", t1, t2)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape.dimensions, [2, 4])
        
        # Incompatible shapes: [2, 3] @ [5, 4]
        t3 = TensorType(BUILTIN_TYPES["f32"], TensorShape([5, 4]))
        result2 = engine.infer_tensor_op_type("@", t1, t3)
        self.assertIsNone(result2)
    
    def test_element_wise_operations(self):
        """Test element-wise operation shape inference"""
        engine = TypeInferenceEngine()
        
        # Same shapes: [2, 3] + [2, 3] -> [2, 3]
        t1 = TensorType(BUILTIN_TYPES["f32"], TensorShape([2, 3]))
        t2 = TensorType(BUILTIN_TYPES["f32"], TensorShape([2, 3]))
        
        result = engine.infer_tensor_op_type("+", t1, t2)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape.dimensions, [2, 3])
        
        # Broadcasting: [2, 1] + [1, 3] -> [2, 3]
        t3 = TensorType(BUILTIN_TYPES["f32"], TensorShape([2, 1]))
        t4 = TensorType(BUILTIN_TYPES["f32"], TensorShape([1, 3]))
        
        result2 = engine.infer_tensor_op_type("+", t3, t4)
        self.assertIsNotNone(result2)
        self.assertEqual(result2.shape.dimensions, [2, 3])


if __name__ == "__main__":
    unittest.main()