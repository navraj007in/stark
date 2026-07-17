//! Expression precedence and parenthesization.
//!
//! The AST does not represent grouping parentheses (`02-Syntax-Grammar.md`:
//! `(expr)` parses to the inner expression); precedence is instead baked
//! into tree shape by the parser's 16-level recursive-descent table
//! (`parser.rs`, `expr_inner` through `primary_expr`). Printing that tree
//! back to text must therefore re-derive, from tree shape alone, exactly
//! where the original source needed parentheses — otherwise the printed
//! text can silently reparse to a *different* tree than the one printed.
//!
//! Levels below mirror the parser precedence table 1:1 (0 = loosest).

use crate::ast::{AssignOp, Ast, BinOp, ExprId, ExprKind};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Level(pub u8);

pub const ASSIGN: Level = Level(0);
pub const RANGE: Level = Level(1);
pub const OR: Level = Level(2);
pub const AND: Level = Level(3);
pub const EQUALITY: Level = Level(4);
pub const RELATIONAL: Level = Level(5);
pub const BITOR: Level = Level(6);
pub const BITXOR: Level = Level(7);
pub const BITAND: Level = Level(8);
pub const SHIFT: Level = Level(9);
pub const ADDITIVE: Level = Level(10);
pub const MULTIPLICATIVE: Level = Level(11);
pub const POW: Level = Level(12);
pub const CAST: Level = Level(13);
pub const UNARY: Level = Level(14);
pub const POSTFIX: Level = Level(15);
pub const PRIMARY: Level = Level(16);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Assoc {
    Left,
    Right,
    /// The parser rejects chaining outright (equality, relational, range).
    /// A same-level child in either position only arises from explicit
    /// source parens, so it is always reparenthesized defensively.
    None,
}

pub fn bin_op_level(op: BinOp) -> (Level, Assoc) {
    match op {
        BinOp::Or => (OR, Assoc::Left),
        BinOp::And => (AND, Assoc::Left),
        BinOp::Eq | BinOp::Ne => (EQUALITY, Assoc::None),
        BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => (RELATIONAL, Assoc::None),
        BinOp::BitOr => (BITOR, Assoc::Left),
        BinOp::BitXor => (BITXOR, Assoc::Left),
        BinOp::BitAnd => (BITAND, Assoc::Left),
        BinOp::Shl | BinOp::Shr => (SHIFT, Assoc::Left),
        BinOp::Add | BinOp::Sub => (ADDITIVE, Assoc::Left),
        BinOp::Mul | BinOp::Div | BinOp::Rem => (MULTIPLICATIVE, Assoc::Left),
        BinOp::Pow => (POW, Assoc::Right),
    }
}

pub fn bin_op_symbol(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Rem => "%",
        BinOp::Pow => "**",
        BinOp::Eq => "==",
        BinOp::Ne => "!=",
        BinOp::Lt => "<",
        BinOp::Le => "<=",
        BinOp::Gt => ">",
        BinOp::Ge => ">=",
        BinOp::And => "&&",
        BinOp::Or => "||",
        BinOp::BitAnd => "&",
        BinOp::BitOr => "|",
        BinOp::BitXor => "^",
        BinOp::Shl => "<<",
        BinOp::Shr => ">>",
    }
}

pub fn assign_op_symbol(op: AssignOp) -> &'static str {
    match op {
        AssignOp::Assign => "=",
        AssignOp::AddAssign => "+=",
        AssignOp::SubAssign => "-=",
        AssignOp::MulAssign => "*=",
        AssignOp::DivAssign => "/=",
        AssignOp::RemAssign => "%=",
        AssignOp::PowAssign => "**=",
        AssignOp::BitAndAssign => "&=",
        AssignOp::BitOrAssign => "|=",
        AssignOp::BitXorAssign => "^=",
        AssignOp::ShlAssign => "<<=",
        AssignOp::ShrAssign => ">>=",
    }
}

/// The precedence level of `id`'s outermost form. Matches the parser's
/// levels; forms parsed only via `primary_expr` (literals, paths, `if`,
/// `match`, blocks, tuples, arrays, struct literals, ...) are atomic from
/// the outside and sit at [`PRIMARY`], the loosest level that never needs
/// parenthesization as someone else's child.
pub fn level(ast: &Ast, id: ExprId) -> Level {
    match &ast.expr(id).kind {
        ExprKind::Assign { .. } => ASSIGN,
        ExprKind::Range { .. } => RANGE,
        ExprKind::Binary { op, .. } => bin_op_level(*op).0,
        ExprKind::Cast { .. } => CAST,
        ExprKind::Unary { .. } => UNARY,
        ExprKind::Call { .. }
        | ExprKind::Field { .. }
        | ExprKind::TupleField { .. }
        | ExprKind::Index { .. }
        | ExprKind::Try(_) => POSTFIX,
        _ => PRIMARY,
    }
}

/// Should `child` be wrapped in parentheses when printed at `position`
/// (lhs/rhs of a binary form, or the sole operand of a prefix form) whose
/// enclosing operator has `parent_level` and `assoc`?
pub fn needs_parens(
    child: &Ast,
    child_id: ExprId,
    parent_level: Level,
    assoc: Assoc,
    position: Position,
) -> bool {
    let child_level = level(child, child_id);
    if child_level > parent_level {
        return false;
    }
    if child_level < parent_level {
        return true;
    }
    // Equal precedence: only the "natural recursion" side of the parser's
    // associativity avoids parens.
    !matches!(
        (assoc, position),
        (Assoc::Left, Position::Lhs) | (Assoc::Right, Position::Rhs)
    )
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Position {
    Lhs,
    Rhs,
}

/// True if `id`'s leftmost/"head" subexpression is a bare struct literal
/// that would be swallowed as a block-start `{` when printed directly in
/// `if`/`while`/`for`/`match` condition position (`02-Syntax-Grammar.md`'s
/// struct-literal restriction; parser.rs `Restrictions::no_struct_literal`).
/// Mirrors exactly which recursive positions the parser leaves restricted:
/// binary/unary/cast/range/assign left-recursion and postfix bases, but not
/// call args, index arguments, array/tuple elements, or anything else the
/// parser re-parses with `DEFAULT` restrictions.
pub fn head_is_struct_lit(ast: &Ast, id: ExprId) -> bool {
    match &ast.expr(id).kind {
        ExprKind::StructLit { .. } => true,
        ExprKind::Binary { lhs, .. } => head_is_struct_lit(ast, *lhs),
        ExprKind::Assign { lhs, .. } => head_is_struct_lit(ast, *lhs),
        ExprKind::Range { lo, .. } => head_is_struct_lit(ast, *lo),
        ExprKind::Cast { expr, .. } => head_is_struct_lit(ast, *expr),
        ExprKind::Unary { operand, .. } => head_is_struct_lit(ast, *operand),
        ExprKind::Field { base, .. } => head_is_struct_lit(ast, *base),
        ExprKind::TupleField { base, .. } => head_is_struct_lit(ast, *base),
        ExprKind::Index { base, .. } => head_is_struct_lit(ast, *base),
        ExprKind::Call { callee, .. } => head_is_struct_lit(ast, *callee),
        ExprKind::Try(inner) => head_is_struct_lit(ast, *inner),
        _ => false,
    }
}
