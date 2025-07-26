"""
STARK Type System Implementation

This module implements the core type system for STARK, including:
- Basic types (primitives, composites)
- AI types (Tensor, Model, Dataset)
- Type inference with unification
- Shape inference for tensor operations
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple, Any
from enum import Enum, auto
from abc import ABC, abstractmethod


class TypeKind(Enum):
    """Enumeration of all type kinds in STARK"""
    # Primitives
    I8 = auto()
    I16 = auto()
    I32 = auto()
    I64 = auto()
    I128 = auto()
    U8 = auto()
    U16 = auto()
    U32 = auto()
    U64 = auto()
    U128 = auto()
    F32 = auto()
    F64 = auto()
    BOOL = auto()
    CHAR = auto()
    STR = auto()
    
    # Composites
    ARRAY = auto()
    LIST = auto()
    TUPLE = auto()
    MAP = auto()
    SET = auto()
    OPTIONAL = auto()
    
    # References
    REF = auto()
    MUT_REF = auto()
    
    # Functions
    FUNCTION = auto()
    
    # AI Types
    TENSOR = auto()
    MODEL = auto()
    DATASET = auto()
    GRAPH = auto()
    
    # Special
    VOID = auto()
    NEVER = auto()
    TYPE_VAR = auto()
    UNKNOWN = auto()
    
    # User-defined
    STRUCT = auto()
    ENUM = auto()
    TRAIT = auto()
    ACTOR = auto()


class Type(ABC):
    """Base class for all types in STARK"""
    
    @abstractmethod
    def __str__(self) -> str:
        pass
    
    @abstractmethod
    def equals(self, other: 'Type') -> bool:
        pass
    
    @abstractmethod
    def substitute(self, subst: Dict[str, 'Type']) -> 'Type':
        """Apply type variable substitution"""
        pass


@dataclass
class PrimitiveType(Type):
    """Primitive types like i32, f64, bool, etc."""
    kind: TypeKind
    
    def __str__(self) -> str:
        return self.kind.name.lower()
    
    def equals(self, other: Type) -> bool:
        return isinstance(other, PrimitiveType) and self.kind == other.kind
    
    def substitute(self, subst: Dict[str, Type]) -> Type:
        return self


@dataclass
class ArrayType(Type):
    """Fixed-size array type"""
    element_type: Type
    size: Optional[int] = None  # None means dynamic size
    
    def __str__(self) -> str:
        if self.size is not None:
            return f"[{self.element_type}; {self.size}]"
        return f"[{self.element_type}]"
    
    def equals(self, other: Type) -> bool:
        return (isinstance(other, ArrayType) and 
                self.element_type.equals(other.element_type) and
                self.size == other.size)
    
    def substitute(self, subst: Dict[str, Type]) -> Type:
        return ArrayType(self.element_type.substitute(subst), self.size)


@dataclass
class ListType(Type):
    """Dynamic list type"""
    element_type: Type
    
    def __str__(self) -> str:
        return f"List<{self.element_type}>"
    
    def equals(self, other: Type) -> bool:
        return (isinstance(other, ListType) and 
                self.element_type.equals(other.element_type))
    
    def substitute(self, subst: Dict[str, Type]) -> Type:
        return ListType(self.element_type.substitute(subst))


@dataclass
class TupleType(Type):
    """Tuple type with fixed elements"""
    element_types: List[Type]
    
    def __str__(self) -> str:
        elements = ", ".join(str(t) for t in self.element_types)
        return f"({elements})"
    
    def equals(self, other: Type) -> bool:
        return (isinstance(other, TupleType) and 
                len(self.element_types) == len(other.element_types) and
                all(a.equals(b) for a, b in zip(self.element_types, other.element_types)))
    
    def substitute(self, subst: Dict[str, Type]) -> Type:
        return TupleType([t.substitute(subst) for t in self.element_types])


@dataclass
class MapType(Type):
    """Map/dictionary type"""
    key_type: Type
    value_type: Type
    
    def __str__(self) -> str:
        return f"Map<{self.key_type}, {self.value_type}>"
    
    def equals(self, other: Type) -> bool:
        return (isinstance(other, MapType) and 
                self.key_type.equals(other.key_type) and
                self.value_type.equals(other.value_type))
    
    def substitute(self, subst: Dict[str, Type]) -> Type:
        return MapType(
            self.key_type.substitute(subst),
            self.value_type.substitute(subst)
        )


@dataclass
class OptionalType(Type):
    """Optional/nullable type"""
    inner_type: Type
    
    def __str__(self) -> str:
        return f"?{self.inner_type}"
    
    def equals(self, other: Type) -> bool:
        return (isinstance(other, OptionalType) and 
                self.inner_type.equals(other.inner_type))
    
    def substitute(self, subst: Dict[str, Type]) -> Type:
        return OptionalType(self.inner_type.substitute(subst))


@dataclass
class FunctionType(Type):
    """Function type with parameters and return type"""
    param_types: List[Type]
    return_type: Type
    is_async: bool = False
    
    def __str__(self) -> str:
        params = ", ".join(str(t) for t in self.param_types)
        async_prefix = "async " if self.is_async else ""
        return f"{async_prefix}fn({params}) -> {self.return_type}"
    
    def equals(self, other: Type) -> bool:
        return (isinstance(other, FunctionType) and 
                len(self.param_types) == len(other.param_types) and
                all(a.equals(b) for a, b in zip(self.param_types, other.param_types)) and
                self.return_type.equals(other.return_type) and
                self.is_async == other.is_async)
    
    def substitute(self, subst: Dict[str, Type]) -> Type:
        return FunctionType(
            [t.substitute(subst) for t in self.param_types],
            self.return_type.substitute(subst),
            self.is_async
        )


@dataclass
class TensorShape:
    """Represents tensor shape with support for dynamic dimensions"""
    dimensions: List[Optional[int]]  # None represents dynamic dimension
    
    def __str__(self) -> str:
        dims = [str(d) if d is not None else "?" for d in self.dimensions]
        return f"[{', '.join(dims)}]"
    
    def rank(self) -> int:
        return len(self.dimensions)
    
    def is_compatible_with(self, other: 'TensorShape') -> bool:
        """Check if shapes are compatible (considering dynamic dims)"""
        if self.rank() != other.rank():
            return False
        for d1, d2 in zip(self.dimensions, other.dimensions):
            if d1 is not None and d2 is not None and d1 != d2:
                return False
        return True
    
    def broadcast_with(self, other: 'TensorShape') -> Optional['TensorShape']:
        """Compute broadcast shape if possible"""
        # Implement NumPy-style broadcasting rules
        if self.rank() == 0 or other.rank() == 0:
            return other if self.rank() == 0 else self
        
        # Pad shorter shape with 1s on the left
        dims1 = [1] * (max(self.rank(), other.rank()) - self.rank()) + self.dimensions
        dims2 = [1] * (max(self.rank(), other.rank()) - other.rank()) + other.dimensions
        
        result_dims = []
        for d1, d2 in zip(dims1, dims2):
            if d1 is None or d2 is None:
                result_dims.append(None)
            elif d1 == 1:
                result_dims.append(d2)
            elif d2 == 1:
                result_dims.append(d1)
            elif d1 == d2:
                result_dims.append(d1)
            else:
                return None  # Incompatible shapes
        
        return TensorShape(result_dims)


@dataclass
class TensorType(Type):
    """Tensor type with element type and shape"""
    element_type: Type
    shape: TensorShape
    device: Optional[str] = None  # cpu, gpu, tpu
    
    def __str__(self) -> str:
        device_str = f"@{self.device}" if self.device else ""
        return f"Tensor<{self.element_type}, {self.shape}>{device_str}"
    
    def equals(self, other: Type) -> bool:
        return (isinstance(other, TensorType) and 
                self.element_type.equals(other.element_type) and
                self.shape.is_compatible_with(other.shape) and
                self.device == other.device)
    
    def substitute(self, subst: Dict[str, Type]) -> Type:
        return TensorType(
            self.element_type.substitute(subst),
            self.shape,
            self.device
        )


@dataclass
class ModelType(Type):
    """Model type with input and output specifications"""
    input_type: Type
    output_type: Type
    
    def __str__(self) -> str:
        return f"Model<{self.input_type}, {self.output_type}>"
    
    def equals(self, other: Type) -> bool:
        return (isinstance(other, ModelType) and 
                self.input_type.equals(other.input_type) and
                self.output_type.equals(other.output_type))
    
    def substitute(self, subst: Dict[str, Type]) -> Type:
        return ModelType(
            self.input_type.substitute(subst),
            self.output_type.substitute(subst)
        )


@dataclass
class DatasetType(Type):
    """Dataset type for ML data"""
    element_type: Type
    
    def __str__(self) -> str:
        return f"Dataset<{self.element_type}>"
    
    def equals(self, other: Type) -> bool:
        return (isinstance(other, DatasetType) and 
                self.element_type.equals(other.element_type))
    
    def substitute(self, subst: Dict[str, Type]) -> Type:
        return DatasetType(self.element_type.substitute(subst))


@dataclass
class TypeVariable(Type):
    """Type variable for generics and type inference"""
    name: str
    constraints: List[Type] = field(default_factory=list)
    
    def __str__(self) -> str:
        if self.constraints:
            constraints_str = " + ".join(str(c) for c in self.constraints)
            return f"{self.name}: {constraints_str}"
        return self.name
    
    def equals(self, other: Type) -> bool:
        return isinstance(other, TypeVariable) and self.name == other.name
    
    def substitute(self, subst: Dict[str, Type]) -> Type:
        if self.name in subst:
            return subst[self.name]
        return self


@dataclass
class StructType(Type):
    """User-defined struct type"""
    name: str
    fields: Dict[str, Type]
    type_params: List[TypeVariable] = field(default_factory=list)
    
    def __str__(self) -> str:
        if self.type_params:
            params = ", ".join(str(p) for p in self.type_params)
            return f"{self.name}<{params}>"
        return self.name
    
    def equals(self, other: Type) -> bool:
        return isinstance(other, StructType) and self.name == other.name
    
    def substitute(self, subst: Dict[str, Type]) -> Type:
        new_fields = {k: v.substitute(subst) for k, v in self.fields.items()}
        return StructType(self.name, new_fields, self.type_params)


@dataclass
class EnumType(Type):
    """User-defined enum type"""
    name: str
    variants: Dict[str, Optional[Type]]
    type_params: List[TypeVariable] = field(default_factory=list)
    
    def __str__(self) -> str:
        if self.type_params:
            params = ", ".join(str(p) for p in self.type_params)
            return f"{self.name}<{params}>"
        return self.name
    
    def equals(self, other: Type) -> bool:
        return isinstance(other, EnumType) and self.name == other.name
    
    def substitute(self, subst: Dict[str, Type]) -> Type:
        new_variants = {
            k: v.substitute(subst) if v else None 
            for k, v in self.variants.items()
        }
        return EnumType(self.name, new_variants, self.type_params)


@dataclass
class TraitType(Type):
    """Trait/interface type"""
    name: str
    methods: Dict[str, FunctionType]
    
    def __str__(self) -> str:
        return self.name
    
    def equals(self, other: Type) -> bool:
        return isinstance(other, TraitType) and self.name == other.name
    
    def substitute(self, subst: Dict[str, Type]) -> Type:
        new_methods = {k: v.substitute(subst) for k, v in self.methods.items()}
        return TraitType(self.name, new_methods)


class TypeEnvironment:
    """Type environment for variable and function bindings"""
    
    def __init__(self, parent: Optional['TypeEnvironment'] = None):
        self.bindings: Dict[str, Type] = {}
        self.parent = parent
    
    def bind(self, name: str, type: Type) -> None:
        """Bind a variable to a type"""
        self.bindings[name] = type
    
    def lookup(self, name: str) -> Optional[Type]:
        """Look up a variable's type"""
        if name in self.bindings:
            return self.bindings[name]
        if self.parent:
            return self.parent.lookup(name)
        return None
    
    def extend(self) -> 'TypeEnvironment':
        """Create a child environment"""
        return TypeEnvironment(self)


class TypeInferenceEngine:
    """Type inference engine using unification"""
    
    def __init__(self):
        self.type_var_counter = 0
        self.substitutions: Dict[str, Type] = {}
    
    def fresh_type_var(self, prefix: str = "T") -> TypeVariable:
        """Generate a fresh type variable"""
        name = f"{prefix}{self.type_var_counter}"
        self.type_var_counter += 1
        return TypeVariable(name)
    
    def unify(self, t1: Type, t2: Type) -> bool:
        """Unify two types, updating substitutions"""
        t1 = self.apply_substitutions(t1)
        t2 = self.apply_substitutions(t2)
        
        # Same types unify trivially
        if t1.equals(t2):
            return True
        
        # Type variable unification
        if isinstance(t1, TypeVariable):
            return self.unify_var(t1, t2)
        if isinstance(t2, TypeVariable):
            return self.unify_var(t2, t1)
        
        # Structural unification
        if type(t1) != type(t2):
            return False
        
        if isinstance(t1, ArrayType):
            return self.unify(t1.element_type, t2.element_type)
        
        if isinstance(t1, ListType):
            return self.unify(t1.element_type, t2.element_type)
        
        if isinstance(t1, TupleType):
            if len(t1.element_types) != len(t2.element_types):
                return False
            return all(self.unify(a, b) for a, b in zip(t1.element_types, t2.element_types))
        
        if isinstance(t1, FunctionType):
            if len(t1.param_types) != len(t2.param_types):
                return False
            params_unify = all(self.unify(a, b) for a, b in zip(t1.param_types, t2.param_types))
            return params_unify and self.unify(t1.return_type, t2.return_type)
        
        if isinstance(t1, TensorType):
            return (self.unify(t1.element_type, t2.element_type) and
                    t1.shape.is_compatible_with(t2.shape))
        
        return False
    
    def unify_var(self, var: TypeVariable, type: Type) -> bool:
        """Unify a type variable with a type"""
        if var.name in self.substitutions:
            return self.unify(self.substitutions[var.name], type)
        
        # Occurs check
        if self.occurs_check(var.name, type):
            return False
        
        self.substitutions[var.name] = type
        return True
    
    def occurs_check(self, var_name: str, type: Type) -> bool:
        """Check if a type variable occurs in a type"""
        if isinstance(type, TypeVariable):
            return type.name == var_name
        # Implement for other type constructors
        return False
    
    def apply_substitutions(self, type: Type) -> Type:
        """Apply current substitutions to a type"""
        return type.substitute(self.substitutions)
    
    def infer_tensor_op_type(self, op: str, t1: TensorType, t2: TensorType) -> Optional[TensorType]:
        """Infer result type for tensor operations"""
        if op == "@":  # Matrix multiplication
            # Check shape compatibility: [a, b] @ [b, c] -> [a, c]
            if t1.shape.rank() == 2 and t2.shape.rank() == 2:
                d1 = t1.shape.dimensions
                d2 = t2.shape.dimensions
                if d1[1] == d2[0] or d1[1] is None or d2[0] is None:
                    result_shape = TensorShape([d1[0], d2[1]])
                    return TensorType(t1.element_type, result_shape, t1.device)
            return None
        
        elif op in ["+", "-", "*", "/"]:  # Element-wise operations
            # Try broadcasting
            broadcast_shape = t1.shape.broadcast_with(t2.shape)
            if broadcast_shape:
                return TensorType(t1.element_type, broadcast_shape, t1.device)
            return None
        
        return None


# Built-in types
BUILTIN_TYPES = {
    "i8": PrimitiveType(TypeKind.I8),
    "i16": PrimitiveType(TypeKind.I16),
    "i32": PrimitiveType(TypeKind.I32),
    "i64": PrimitiveType(TypeKind.I64),
    "i128": PrimitiveType(TypeKind.I128),
    "u8": PrimitiveType(TypeKind.U8),
    "u16": PrimitiveType(TypeKind.U16),
    "u32": PrimitiveType(TypeKind.U32),
    "u64": PrimitiveType(TypeKind.U64),
    "u128": PrimitiveType(TypeKind.U128),
    "f32": PrimitiveType(TypeKind.F32),
    "f64": PrimitiveType(TypeKind.F64),
    "bool": PrimitiveType(TypeKind.BOOL),
    "char": PrimitiveType(TypeKind.CHAR),
    "str": PrimitiveType(TypeKind.STR),
    "void": PrimitiveType(TypeKind.VOID),
}